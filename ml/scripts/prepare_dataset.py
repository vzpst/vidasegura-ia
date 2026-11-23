"""Validate camera JSONL and build the versioned fixed-length feature matrix."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from common import (
    CHALLENGE_LABELS,
    POSTURE_LABELS,
    connected_group_ids,
    dataset_fingerprint,
    duplicate_groups,
    extract_features,
    feature_metadata,
    read_jsonl_files,
    training_sample_reasons,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Arquivos JSONL exportados pela ferramenta de captura")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "processed" / "pose-features-v1",
    )
    parser.add_argument(
        "--allow-invalid-records", action="store_true",
        help="Processa as linhas utilizáveis e grava cada rejeição no relatório local.",
    )
    args = parser.parse_args()

    records, read_issues = read_jsonl_files(args.inputs)
    usable: list[dict] = []
    rejected: list[dict] = []
    for record in records:
        reasons = training_sample_reasons(record)
        if reasons:
            rejected.append({
                "sample_id": record.get("sample_id") if isinstance(record, dict) else None,
                "source_file": record.get("_source_file") if isinstance(record, dict) else None,
                "source_line": record.get("_source_line") if isinstance(record, dict) else None,
                "reasons": reasons,
            })
        else:
            usable.append(record)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "report_version": 1,
        "records_read": len(records),
        "usable_for_training": len(usable),
        "rejected_records": rejected,
        "read_issues": read_issues,
        "possible_duplicates": duplicate_groups(usable),
        "dataset_fingerprint": dataset_fingerprint(usable),
    }
    report_path = args.output_dir / "preparation-report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if (rejected or read_issues) and not args.allow_invalid_records:
        print(f"Preparação interrompida: há rejeições. Consulte {report_path}.")
        print("Use --allow-invalid-records para processar somente registros utilizáveis após revisar o relatório.")
        return 1
    if not usable:
        print("Nenhuma amostra utilizável foi encontrada; nenhum vetor foi salvo.")
        return 1

    try:
        import numpy as np
    except ImportError as error:
        raise SystemExit("Instale as dependências no ambiente virtual do laboratório: python -m pip install -r ml/requirements.txt") from error

    features = [extract_features(record) for record in usable]
    label_indices = [POSTURE_LABELS.index(record["posture_label"]) for record in usable]
    group_ids = connected_group_ids(usable)
    output_path = args.output_dir / "dataset.npz"
    np.savez_compressed(
        output_path,
        features=np.asarray(features, dtype=np.float32),
        labels=np.asarray(label_indices, dtype=np.int64),
        posture_classes=np.asarray(POSTURE_LABELS),
        sample_ids=np.asarray([record["sample_id"] for record in usable]),
        subject_ids=np.asarray([record["subject_id"] for record in usable]),
        session_ids=np.asarray([record["session_id"] for record in usable]),
        viewpoint_labels=np.asarray([record["viewpoint_label"] for record in usable]),
        geometric_predictions=np.asarray([
            record.get("geometric_prediction") or "inconclusive" for record in usable
        ]),
        challenge_labels=np.asarray([
            record.get("challenge_label") or "unspecified" for record in usable
        ]),
        group_ids=np.asarray(group_ids),
    )
    metadata = {
        "dataset_schema_version": 1,
        "feature_transform": feature_metadata(),
        "posture_class_order": list(POSTURE_LABELS),
        "sample_count": len(usable),
        "class_counts": {label: Counter(record["posture_label"] for record in usable)[label] for label in POSTURE_LABELS},
        "viewpoint_counts": {
            label: sum(record["viewpoint_label"] == label for record in usable)
            for label in ("front", "left_profile", "right_profile", "oblique")
        },
        "challenge_counts": {
            label: sum(record.get("challenge_label", "unspecified") == label for record in usable)
            for label in CHALLENGE_LABELS
        },
        "participant_count": len({record["subject_id"] for record in usable}),
        "session_count": len({record["session_id"] for record in usable}),
        "independent_group_count": len(set(group_ids)),
        "dataset_fingerprint": report["dataset_fingerprint"],
        "source_files": [str(path) for path in args.inputs],
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"prepared": len(usable), "features": len(features[0]), "output": str(output_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
