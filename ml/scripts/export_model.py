"""Convert a human-reviewed Keras run into a local TensorFlow.js Layers model."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from common import FEATURE_TRANSFORM_VERSION, feature_metadata
from common import POSTURE_LABELS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_eligibility(run_dir: Path, human_review_confirmed: bool) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    training_path = run_dir / "training-metadata.json"
    evaluation_path = run_dir / "evaluation.json"
    model_path = run_dir / "posture_model.h5"
    if not training_path.is_file():
        return False, ["metadados de treinamento ausentes"]
    if not evaluation_path.is_file():
        return False, ["avaliação holdout ausente"]
    if not model_path.is_file():
        return False, ["modelo Keras treinado ausente"]
    training = json.loads(training_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    model_hash = sha256_file(model_path)
    expected_hash = training.get("model_files", {}).get(model_path.name)
    if model_hash != expected_hash:
        reasons.append("hash do modelo diverge dos metadados de treinamento")
    if not evaluation.get("evaluation_completed"):
        reasons.append("avaliação holdout não foi concluída")
    if training.get("evaluation_status") != "EVALUATED_REVIEW_REQUIRED":
        reasons.append("metadados do treino ainda não registram avaliação holdout concluída")
    if training.get("class_order") != list(POSTURE_LABELS):
        reasons.append("ordem das classes no treino não corresponde ao esquema esperado")
    if training.get("feature_transform", {}).get("version") != FEATURE_TRANSFORM_VERSION:
        reasons.append("metadados do treino usam outra versão de transformação")
    if evaluation.get("human_review_status") != "REVIEW_REQUIRED":
        reasons.append("relatório não tem estado de revisão esperado")
    if not evaluation.get("split_grouped_by_participant_and_session"):
        reasons.append("partições não comprovam separação por participante e sessão")
    if evaluation.get("model_sha256") != model_hash:
        reasons.append("avaliação não corresponde ao arquivo de modelo atual")
    if evaluation.get("dataset_fingerprint") != training.get("dataset_fingerprint"):
        reasons.append("avaliação e treinamento usam fingerprints de dataset diferentes")
    if evaluation.get("feature_transform_version") != FEATURE_TRANSFORM_VERSION:
        reasons.append("avaliação usa outra versão de transformação")
    metrics = evaluation.get("test_metrics", {})
    if metrics.get("class_order") != list(POSTURE_LABELS):
        reasons.append("ordem das classes na avaliação não corresponde ao esquema esperado")
    if metrics.get("sample_count", 0) <= 0:
        reasons.append("conjunto holdout sem amostras avaliadas")
    per_class = metrics.get("per_class", {})
    missing_classes = [
        label for label in POSTURE_LABELS
        if label not in per_class or per_class[label].get("support", 0) <= 0
    ]
    if missing_classes:
        reasons.append(f"holdout sem suporte para classes: {', '.join(missing_classes)}")
    if not human_review_confirmed:
        reasons.append("confirme explicitamente que revisou as métricas holdout")
    return not reasons, reasons


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="Padrão: <run-dir>/tfjs_model")
    parser.add_argument(
        "--confirm-evaluation-reviewed", action="store_true",
        help="Confirma que as métricas holdout foram revisadas antes da conversão.",
    )
    args = parser.parse_args()

    eligible, reasons = export_eligibility(args.run_dir, args.confirm_evaluation_reviewed)
    if not eligible:
        print("Exportação recusada:")
        for reason in reasons:
            print(f"- {reason}")
        return 2
    if platform.system() == "Windows":
        print("O conversor está configurado para Linux/WSL2 porque TF-DF não mantém wheels nativos do Windows.")
        print("Execute este comando no ambiente virtual Linux/WSL2 apontando para os mesmos artefatos locais.")
        return 2

    converter = shutil.which("tensorflowjs_converter")
    if not converter:
        print("tensorflowjs_converter ausente. Instale ml/requirements.txt no ambiente virtual Linux/WSL2.")
        return 2
    output_dir = args.output_dir or args.run_dir / "tfjs_model"
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"O destino já contém arquivos: {output_dir}")
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.run_dir / "posture_model.h5"
    command = [
        converter,
        "--input_format=keras",
        "--output_format=tfjs_layers_model",
        str(model_path.resolve()),
        str(output_dir.resolve()),
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        print("TensorFlow.js Converter falhou; consulte as mensagens acima antes de repetir a exportação.")
        return completed.returncode or 1

    model_json = output_dir / "model.json"
    if not model_json.is_file():
        print("O conversor terminou sem criar model.json.")
        return 1
    converted = json.loads(model_json.read_text(encoding="utf-8"))
    manifest = converted.get("weightsManifest")
    if not isinstance(manifest, list) or not manifest:
        print("model.json não contém weightsManifest; exportação inválida.")
        return 1
    missing_weights = [
        path for group in manifest for path in group.get("paths", [])
        if not (output_dir / path).is_file()
    ]
    if missing_weights:
        print(f"Arquivos de pesos ausentes: {missing_weights}")
        return 1

    training = json.loads((args.run_dir / "training-metadata.json").read_text(encoding="utf-8"))
    evaluation = json.loads((args.run_dir / "evaluation.json").read_text(encoding="utf-8"))
    metadata = {
        "schema_version": 1,
        "model_version": training["run_id"],
        "model_format": "tfjs_layers_model",
        "tfjs_converter_version": "4.22.0",
        "feature_transform": feature_metadata(),
        "class_order": training["class_order"],
        "confidence_thresholds": {
            "keypoint_observed": 0.25,
            "missing_keypoint_mask": 0.0,
        },
        "evaluation_metrics": evaluation["test_metrics"],
        "evaluation_breakdowns": {
            "by_viewpoint": evaluation["by_viewpoint"],
            "by_participant": evaluation["by_participant"],
            "by_challenge_condition": evaluation.get("by_challenge_condition", {}),
            "geometric_classifier_comparison": evaluation["geometric_classifier_comparison"],
        },
        "source_model_sha256": sha256_file(model_path),
        "dataset_fingerprint": training["dataset_fingerprint"],
        "human_review_confirmation": True,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "browser_smoke_test": "Abra ml/tools/verify-tfjs-export.html e carregue este model.json com tf.loadLayersModel().",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Modelo exportado localmente para: {output_dir}")
    print("A exportação ainda precisa passar pelo smoke test tf.loadLayersModel no navegador.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
