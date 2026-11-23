"""Train the experimental MLP only after diversity and grouped-split gates pass."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from common import (
    FEATURE_TRANSFORM_VERSION,
    POSTURE_LABELS,
    diversity_reasons,
    feature_metadata,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def grouped_splits(features, labels, groups, class_count: int, seed: int = 2026):
    """Return grouped train/validation/test indices with every label represented."""
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        import numpy as np
    except ImportError as error:
        raise RuntimeError("Scikit-learn e NumPy precisam estar instalados no ambiente virtual.") from error

    wanted = set(range(class_count))
    for attempt_seed in range(seed, seed + 128):
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=attempt_seed)
        try:
            folds = list(splitter.split(features, labels, groups))
        except ValueError:
            continue
        if len(folds) < 5:
            continue
        test_indices = folds[0][1]
        validation_indices = folds[1][1]
        train_indices = np.concatenate([fold[1] for fold in folds[2:]])
        candidate = {
            "train": train_indices,
            "validation": validation_indices,
            "test": test_indices,
        }
        if any(set(labels[index] for index in indices) != wanted for indices in candidate.values()):
            continue
        split_groups = {name: set(groups[index] for index in indices) for name, indices in candidate.items()}
        if (split_groups["train"] & split_groups["validation"] or
                split_groups["train"] & split_groups["test"] or
                split_groups["validation"] & split_groups["test"]):
            continue
        return candidate, attempt_seed
    raise ValueError(
        "Não foi possível criar treino/validação/teste agrupados com as quatro classes. "
        "Colete mais participantes e sessões para evitar vazamento entre conjuntos."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepared", type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "processed" / "pose-features-v1" / "dataset.npz",
    )
    parser.add_argument(
        "--metadata", type=Path,
        help="Metadados gerados por prepare_dataset.py (padrão: metadata.json ao lado do NPZ)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts",
        help="Diretório local ignorado pelo Git para os resultados do treino",
    )
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    metadata_path = args.metadata or args.prepared.with_name("metadata.json")
    if not args.prepared.is_file() or not metadata_path.is_file():
        print("Dataset preparado não encontrado. Execute prepare_dataset.py após validar dados reais.")
        return 2

    try:
        import numpy as np
    except ImportError:
        print("NumPy não está instalado no ambiente virtual do laboratório.")
        return 2

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("feature_transform", {}).get("version") != FEATURE_TRANSFORM_VERSION:
        print("Versão de transformação incompatível com este script.")
        return 2
    with np.load(args.prepared, allow_pickle=False) as archive:
        features = archive["features"].astype(np.float32)
        labels = archive["labels"].astype(np.int64)
        classes = archive["posture_classes"].tolist()
        sample_ids = archive["sample_ids"].astype(str).tolist()
        subject_ids = archive["subject_ids"].astype(str).tolist()
        session_ids = archive["session_ids"].astype(str).tolist()
        group_ids = archive["group_ids"].astype(str).tolist()

    if tuple(classes) != POSTURE_LABELS or features.ndim != 2 or features.shape[1] != len(feature_metadata()["feature_order"]):
        print("O vetor ou a ordem de classes do dataset preparado não corresponde à configuração v1.")
        return 2
    records = [
        {"sample_id": sample_id, "subject_id": subject_id, "session_id": session_id,
         "posture_label": classes[int(label)]}
        for sample_id, subject_id, session_id, label in zip(sample_ids, subject_ids, session_ids, labels)
    ]
    reasons = diversity_reasons(records)
    if reasons:
        print("Treino recusado: diversidade insuficiente. Nenhum modelo foi criado.")
        for reason in reasons:
            print(f"- {reason}")
        return 2
    if len(set(group_ids)) < 5:
        print("Treino recusado: são necessários ao menos cinco grupos independentes.")
        return 2

    try:
        splits, split_seed = grouped_splits(features, labels, group_ids, len(classes), args.seed)
    except (RuntimeError, ValueError) as error:
        print(f"Treino recusado: {error}")
        return 2

    try:
        import tensorflow as tf
        from sklearn.utils.class_weight import compute_class_weight
    except ImportError as error:
        print(f"Dependência ausente no ambiente virtual: {error.name}.")
        return 2

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    train_indices = splits["train"]
    validation_indices = splits["validation"]
    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(classes)),
        y=labels[train_indices],
    )
    class_weights = {index: float(weight) for index, weight in enumerate(class_weights_array)}

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(features.shape[1],), name="pose_features_v1"),
        tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.30),
        tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.20),
        tf.keras.layers.Dense(len(classes), activation="softmax", name="posture_output"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(run_dir / "best.keras"), monitor="val_loss", save_best_only=True
        ),
    ]
    history = model.fit(
        features[train_indices], labels[train_indices],
        validation_data=(features[validation_indices], labels[validation_indices]),
        epochs=200, batch_size=32, class_weight=class_weights,
        callbacks=callbacks, verbose=2,
    )

    keras_path = run_dir / "posture_model.keras"
    h5_path = run_dir / "posture_model.h5"
    model.save(keras_path)
    model.save(h5_path)
    history_data = {key: [float(value) for value in values] for key, values in history.history.items()}
    (run_dir / "training-history.json").write_text(
        json.dumps(history_data, indent=2) + "\n", encoding="utf-8"
    )
    split_sample_ids = {
        name: [sample_ids[int(index)] for index in indices]
        for name, indices in splits.items()
    }
    run_metadata = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_format": "Keras MLP; HDF5 copy is prepared for TensorFlow.js Layers conversion",
        "model_files": {
            keras_path.name: sha256_file(keras_path),
            h5_path.name: sha256_file(h5_path),
        },
        "dataset_fingerprint": metadata["dataset_fingerprint"],
        "dataset_metadata_path": str(metadata_path.resolve()),
        "feature_transform": feature_metadata(),
        "class_order": list(classes),
        "split_seed": split_seed,
        "split_sample_ids": split_sample_ids,
        "split_grouped_by_participant_and_session": True,
        "split_counts": {name: len(ids) for name, ids in split_sample_ids.items()},
        "class_weights": {classes[index]: weight for index, weight in class_weights.items()},
        "training_config": {
            "architecture": [128, 64, len(classes)],
            "dropout": [0.30, 0.20],
            "l2": 0.001,
            "max_epochs": 200,
            "batch_size": 32,
            "early_stopping_patience": 15,
            "seed": args.seed,
        },
        "training_and_validation_history": history_data,
        "evaluation_status": "PENDING_GROUPED_HOLDOUT_EVALUATION",
    }
    (run_dir / "training-metadata.json").write_text(
        json.dumps(run_metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Treino experimental concluído. Avalie o conjunto holdout antes de exportar: {run_dir}")
    print("As métricas de treino/validação não são métricas finais de generalização.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
