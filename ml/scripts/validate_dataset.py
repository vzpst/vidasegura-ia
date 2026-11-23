"""Validate camera-captured JSONL files and write an explicit rejection report."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from common import (
    POSTURE_LABELS,
    VIEWPOINT_LABELS,
    dataset_fingerprint,
    diversity_reasons,
    duplicate_groups,
    read_jsonl_files,
    training_sample_reasons,
    validate_record,
)


def build_report(input_paths: list[Path]) -> dict:
    records, read_issues = read_jsonl_files(input_paths)
    rejected: list[dict] = []
    schema_valid: list[dict] = []
    training_usable: list[dict] = []

    for record in records:
        source = {key: record.get(key) for key in ("_source_file", "_source_line")}
        sample_id = record.get("sample_id") if isinstance(record, dict) else None
        schema_errors = validate_record(record)
        if schema_errors:
            rejected.append({**source, "sample_id": sample_id, "reasons": schema_errors})
            continue
        schema_valid.append(record)
        training_errors = training_sample_reasons(record)
        if training_errors:
            rejected.append({**source, "sample_id": sample_id, "reasons": training_errors})
        else:
            training_usable.append(record)

    posture_counts = Counter(record["posture_label"] for record in schema_valid)
    viewpoint_counts = Counter(record["viewpoint_label"] for record in schema_valid)
    diversity = diversity_reasons(training_usable)
    return {
        "report_version": 1,
        "schema_version": 1,
        "input_files": [str(path) for path in input_paths],
        "records_read": len(records),
        "schema_valid_records": len(schema_valid),
        "usable_for_training": len(training_usable),
        "rejected_records": len(rejected),
        "posture_counts": {label: posture_counts[label] for label in POSTURE_LABELS},
        "viewpoint_counts": {label: viewpoint_counts[label] for label in VIEWPOINT_LABELS},
        "participant_count": len({record["subject_id"] for record in schema_valid}),
        "participants": sorted({record["subject_id"] for record in schema_valid}),
        "session_count": len({record["session_id"] for record in schema_valid}),
        "sessions": sorted({record["session_id"] for record in schema_valid}),
        "possible_duplicates": duplicate_groups(schema_valid),
        "dataset_fingerprint": dataset_fingerprint(schema_valid),
        "training_diversity_ready": not diversity,
        "training_diversity_reasons": diversity,
        "rejections": rejected,
        "read_issues": read_issues,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Um ou mais arquivos JSONL locais")
    parser.add_argument(
        "--report", type=Path,
        default=Path(__file__).resolve().parents[1] / "reports" / "dataset-validation.json",
        help="Destino do relatório JSON (padrão: ml/reports/dataset-validation.json)",
    )
    args = parser.parse_args()

    report = build_report(args.inputs)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "records_read": report["records_read"],
        "schema_valid_records": report["schema_valid_records"],
        "usable_for_training": report["usable_for_training"],
        "rejected_records": report["rejected_records"],
        "possible_duplicate_groups": len(report["possible_duplicates"]),
        "training_diversity_ready": report["training_diversity_ready"],
        "report": str(args.report),
    }, indent=2, ensure_ascii=False))
    return 1 if report["rejected_records"] or report["read_issues"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
