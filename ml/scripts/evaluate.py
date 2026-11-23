"""Evaluate a trained model on the participant/session-grouped holdout only."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from common import CHALLENGE_LABELS, POSTURE_LABELS, FEATURE_TRANSFORM_VERSION, validate_split_disjoint


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sklearn_metrics(y_true, y_pred, classes: list[str]) -> dict:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    import numpy as np

    indices = np.arange(len(classes))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=indices, average="macro", zero_division=0)),
        "per_class": classification_report(
            y_true, y_pred, labels=indices, target_names=classes,
            output_dict=True, zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=indices).tolist(),
        "class_order": classes,
        "sample_count": int(len(y_true)),
    }


def metric_breakdowns(y_true, y_pred, grouping: list[str], classes: list[str]) -> dict:
    import numpy as np

    output: dict[str, dict] = {}
    for key in sorted(set(grouping)):
        indices = np.asarray([index for index, value in enumerate(grouping) if value == key], dtype=np.int64)
        output[key] = sklearn_metrics(y_true[indices], y_pred[indices], classes)
    return output


def geometric_baseline(true_labels, geometric_predictions, viewpoints, classes: list[str]) -> dict:
    total = len(true_labels)
    selected_indices = [
        index for index, prediction in enumerate(geometric_predictions)
        if prediction in classes
    ]
    if not selected_indices:
        return {"coverage": 0.0, "covered_samples": 0, "total_samples": total, "metrics": None, "by_viewpoint": {}}
    y_true = [true_labels[index] for index in selected_indices]
    y_pred = [classes.index(geometric_predictions[index]) for index in selected_indices]
    selected_viewpoints = [viewpoints[index] for index in selected_indices]
    return {
        "coverage": len(selected_indices) / total if total else 0.0,
        "covered_samples": len(selected_indices),
        "total_samples": total,
        "metrics": sklearn_metrics(y_true, y_pred, classes),
        "by_viewpoint": metric_breakdowns(y_true, y_pred, selected_viewpoints, classes),
        "inconclusive_predictions_excluded_from_accuracy": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Diretório criado por train_posture.py")
    parser.add_argument("--prepared", type=Path, required=True, help="dataset.npz original usado no treino")
    parser.add_argument("--metadata", type=Path, help="metadata.json ao lado do NPZ, se não estiver no local padrão")
    args = parser.parse_args()

    run_metadata_path = args.run_dir / "training-metadata.json"
    if not run_metadata_path.is_file() or not args.prepared.is_file():
        print("Metadados do treino ou dataset preparado não encontrados.")
        return 2
    run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
    dataset_metadata_path = args.metadata or args.prepared.with_name("metadata.json")
    if not dataset_metadata_path.is_file():
        print("Metadados do dataset preparado não encontrados.")
        return 2
    dataset_metadata = json.loads(dataset_metadata_path.read_text(encoding="utf-8"))
    if dataset_metadata.get("dataset_fingerprint") != run_metadata.get("dataset_fingerprint"):
        print("O fingerprint do dataset não corresponde ao modelo treinado.")
        return 2
    if run_metadata.get("feature_transform", {}).get("version") != FEATURE_TRANSFORM_VERSION:
        print("A transformação de características não corresponde a esta versão do avaliador.")
        return 2

    model_path = args.run_dir / "posture_model.keras"
    expected_hash = run_metadata.get("model_files", {}).get(model_path.name)
    if not model_path.is_file() or sha256_file(model_path) != expected_hash:
        print("O arquivo do modelo não corresponde ao hash registrado no treinamento.")
        return 2

    try:
        import numpy as np
        import tensorflow as tf
    except ImportError as error:
        print(f"Dependência ausente no ambiente virtual: {error.name}.")
        return 2
    try:
        from sklearn.metrics import classification_report  # noqa: F401
    except ImportError:
        print("Scikit-learn não está instalado no ambiente virtual do laboratório.")
        return 2

    with np.load(args.prepared, allow_pickle=False) as archive:
        features = archive["features"].astype(np.float32)
        label_indices = archive["labels"].astype(np.int64)
        classes = archive["posture_classes"].astype(str).tolist()
        sample_ids = archive["sample_ids"].astype(str).tolist()
        subjects = archive["subject_ids"].astype(str).tolist()
        sessions = archive["session_ids"].astype(str).tolist()
        viewpoints = archive["viewpoint_labels"].astype(str).tolist()
        geometric_predictions = archive["geometric_predictions"].astype(str).tolist()
        challenge_labels = (
            archive["challenge_labels"].astype(str).tolist()
            if "challenge_labels" in archive.files
            else ["unspecified"] * len(sample_ids)
        )
        groups = archive["group_ids"].astype(str).tolist()

    if tuple(classes) != POSTURE_LABELS:
        print("Ordem das classes no dataset incompatível.")
        return 2
    index_by_id = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    if len(index_by_id) != len(sample_ids):
        print("sample_id duplicado no dataset preparado; avaliação interrompida.")
        return 2
    split_ids = run_metadata.get("split_sample_ids", {})
    if set(split_ids) != {"train", "validation", "test"}:
        print("Metadados não contêm as três partições agrupadas.")
        return 2
    try:
        test_indices = np.asarray([index_by_id[sample_id] for sample_id in split_ids["test"]], dtype=np.int64)
    except KeyError as error:
        print(f"Amostra do holdout ausente no dataset: {error.args[0]}")
        return 2

    reconstructed = {}
    for name, ids in split_ids.items():
        indices = [index_by_id[sample_id] for sample_id in ids]
        reconstructed[name] = [
            {"subject_id": subjects[index], "session_id": sessions[index]}
            for index in indices
        ]
    split_errors = validate_split_disjoint(reconstructed)
    if split_errors or not run_metadata.get("split_grouped_by_participant_and_session"):
        print("Avaliação interrompida: partições não estão isoladas por participante/sessão.")
        for reason in split_errors:
            print(f"- {reason}")
        return 2

    y_test = label_indices[test_indices]
    if set(y_test.tolist()) != set(range(len(classes))):
        print("O conjunto holdout precisa conter todas as classes antes da avaliação.")
        return 2
    model = tf.keras.models.load_model(model_path)
    probabilities = model.predict(features[test_indices], verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    test_viewpoints = [viewpoints[index] for index in test_indices]
    test_subjects = [subjects[index] for index in test_indices]
    test_geometric = [geometric_predictions[index] for index in test_indices]
    truth_labels = [classes[int(index)] for index in y_test]
    prediction_labels = [classes[int(index)] for index in predictions]

    viewpoint_metrics = metric_breakdowns(y_test, predictions, test_viewpoints, classes)
    participant_metrics = metric_breakdowns(y_test, predictions, test_subjects, classes)
    test_challenges = [challenge_labels[index] for index in test_indices]
    challenge_metrics = {}
    for challenge in CHALLENGE_LABELS:
        selected = [index for index, value in enumerate(test_challenges) if value == challenge]
        if not selected:
            challenge_metrics[challenge] = {"sample_count": 0, "status": "NOT_OBSERVED_IN_HOLDOUT"}
            continue
        challenge_metrics[challenge] = {
            "sample_count": len(selected),
            "status": "REPORTED_DESCRIPTIVELY",
            "metrics": sklearn_metrics(y_test[selected], predictions[selected], classes),
        }
    result = {
        "report_version": 1,
        "evaluation_completed": True,
        "evaluation_scope": "participant/session-grouped holdout test partition",
        "human_review_status": "REVIEW_REQUIRED",
        "model_sha256": expected_hash,
        "dataset_fingerprint": run_metadata["dataset_fingerprint"],
        "feature_transform_version": FEATURE_TRANSFORM_VERSION,
        "test_metrics": sklearn_metrics(y_test, predictions, classes),
        "by_viewpoint": viewpoint_metrics,
        "by_participant": participant_metrics,
        "by_challenge_condition": challenge_metrics,
        "geometric_classifier_comparison": geometric_baseline(
            truth_labels, test_geometric, test_viewpoints, classes
        ),
        "split_sample_counts": run_metadata["split_counts"],
        "split_grouped_by_participant_and_session": True,
        "limitations": [
            "Resultados descrevem apenas o holdout deste dataset; não comprovam desempenho clínico ou de segurança.",
            "Métricas por perspectiva e participante dependem da quantidade e diversidade de amostras disponíveis.",
            "Revisão humana dos dados, das métricas e dos vieses é obrigatória antes da exportação.",
        ],
    }
    result_path = args.run_dir / "evaluation.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    run_metadata["evaluation_status"] = "EVALUATED_REVIEW_REQUIRED"
    run_metadata_path.write_text(json.dumps(run_metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "test_accuracy": result["test_metrics"]["accuracy"],
        "test_macro_f1": result["test_metrics"]["macro_f1"],
        "test_samples": result["test_metrics"]["sample_count"],
        "human_review_status": result["human_review_status"],
        "report": str(result_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
