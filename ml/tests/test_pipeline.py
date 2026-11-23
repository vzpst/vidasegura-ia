"""Synthetic fixtures here are unit-test inputs only and never training datasets."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ROOT / "scripts"))

from common import (  # noqa: E402
    CHALLENGE_LABELS,
    FEATURE_TRANSFORM_VERSION,
    GEOMETRY_PAIRS,
    KEYPOINT_NAMES,
    POSTURE_LABELS,
    accuracy_by_group,
    balanced_class_weights,
    connected_group_ids,
    diversity_reasons,
    duplicate_groups,
    extract_features,
    feature_metadata,
    feature_order,
    read_jsonl_files,
    training_sample_reasons,
    validate_record,
    validate_split_disjoint,
)
from export_model import export_eligibility, sha256_file  # noqa: E402


def camera_sample(
    sample_id: str = "sample_1",
    subject_id: str = "P001",
    session_id: str = "S001",
    posture: str = "standing",
    viewpoint: str = "front",
) -> dict:
    points = {
        "left_shoulder": [0.42, 0.20], "right_shoulder": [0.58, 0.20],
        "left_hip": [0.45, 0.50], "right_hip": [0.55, 0.50],
        "left_elbow": [0.38, 0.32], "right_elbow": [0.62, 0.32],
        "left_wrist": [0.36, 0.45], "right_wrist": [0.64, 0.45],
        "left_knee": [0.45, 0.70], "right_knee": [0.55, 0.70],
        "left_ankle": [0.45, 0.90], "right_ankle": [0.55, 0.90],
    }
    return {
        "schema_version": 1,
        "sample_id": sample_id,
        "subject_id": subject_id,
        "session_id": session_id,
        "timestamp_ms": 1_700_000_000_000,
        "image_width": 640,
        "image_height": 480,
        "data_origin": "camera",
        "capture_method": "movenet_singlepose_lightning",
        "posture_label": posture,
        "viewpoint_label": viewpoint,
        "geometric_prediction": "standing",
        "keypoints": {
            name: {"x": x, "y": y, "score": 0.9, "validity": "observed"}
            for name, (x, y) in points.items()
        },
    }


class DatasetTests(unittest.TestCase):
    def test_schema_accepts_camera_capture_and_has_all_12_points(self):
        sample = camera_sample()
        self.assertEqual(validate_record(sample), [])
        self.assertEqual(tuple(sample["keypoints"]), KEYPOINT_NAMES)

    def test_boolean_is_not_accepted_as_schema_version_one(self):
        sample = camera_sample()
        sample["schema_version"] = True
        self.assertTrue(any("schema_version" in error for error in validate_record(sample)))

    def test_optional_challenge_condition_is_validated_separately_from_posture(self):
        sample = camera_sample(posture="lying", viewpoint="left_profile")
        sample["challenge_label"] = "side_lying"
        self.assertEqual(validate_record(sample), [])
        sample["challenge_label"] = "fall_confirmed"
        self.assertTrue(any("challenge_label" in error for error in validate_record(sample)))

    def test_jsonl_reader_loads_records_and_reports_bad_lines(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "capture.jsonl"
            path.write_text(json.dumps(camera_sample()) + "\nnot-json\n", encoding="utf-8")
            records, issues = read_jsonl_files([path])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["_source_line"], 1)
        self.assertEqual(issues[0]["line"], 2)
        self.assertIn("JSON inválido", issues[0]["reason"])

    def test_rejects_out_of_range_normalized_coordinates_and_scores(self):
        sample = camera_sample()
        sample["keypoints"]["left_shoulder"]["x"] = 1.2
        sample["keypoints"]["right_shoulder"]["score"] = 1.5
        errors = validate_record(sample)
        self.assertTrue(any("coordenadas normalizadas" in error for error in errors))
        self.assertTrue(any("score precisa" in error for error in errors))

    def test_missing_keypoints_are_explicit_and_keep_mask_zero(self):
        sample = camera_sample()
        sample["keypoints"]["left_wrist"] = {
            "x": None, "y": None, "score": None, "validity": "missing"
        }
        self.assertEqual(validate_record(sample), [])
        vector = extract_features(sample)
        mask_index = KEYPOINT_NAMES.index("left_wrist") * 4 + 3
        self.assertEqual(vector[mask_index], 0.0)
        self.assertEqual(vector[mask_index - 1], 0.0)

    def test_low_confidence_point_keeps_score_but_uses_neutral_coordinates_and_mask(self):
        sample = camera_sample()
        sample["keypoints"]["left_wrist"] = {
            "x": 0.36, "y": 0.45, "score": 0.1, "validity": "low_confidence"
        }
        self.assertEqual(validate_record(sample), [])
        vector = extract_features(sample)
        start = KEYPOINT_NAMES.index("left_wrist") * 4
        self.assertEqual(vector[start:start + 4], [0.0, 0.0, 0.1, 0.0])

    def test_missing_joint_is_not_silently_inferred_or_used_as_coordinate(self):
        sample = camera_sample()
        sample["keypoints"]["left_knee"] = {
            "x": None, "y": None, "score": 0.7, "validity": "missing"
        }
        self.assertEqual(validate_record(sample), [])
        vector = extract_features(sample)
        start = KEYPOINT_NAMES.index("left_knee") * 4
        self.assertEqual(vector[start:start + 4], [0.0, 0.0, 0.7, 0.0])

    def test_insufficiently_visible_capture_is_not_training_usable(self):
        sample = camera_sample()
        for name, point in sample["keypoints"].items():
            point.update(x=None, y=None, score=None, validity="missing")
        self.assertTrue(any("articulações observadas" in reason for reason in training_sample_reasons(sample)))

    def test_synthetic_origin_is_rejected_for_training(self):
        sample = camera_sample()
        sample["data_origin"] = "synthetic"
        self.assertTrue(any("amostras sintéticas" in error for error in validate_record(sample)))

    def test_possible_duplicates_are_reported(self):
        first = camera_sample(sample_id="sample_1")
        second = camera_sample(sample_id="sample_2")
        self.assertEqual(duplicate_groups([first, second]), [["sample_1", "sample_2"]])


class FeatureTests(unittest.TestCase):
    def test_feature_transform_has_a_versioned_fixed_order(self):
        sample = camera_sample()
        vector = extract_features(sample)
        order = feature_order()
        self.assertEqual(FEATURE_TRANSFORM_VERSION, "pose-features-v1")
        self.assertEqual(len(vector), len(order))
        self.assertEqual(len(order), 12 * 4 + len(GEOMETRY_PAIRS) * 5)
        self.assertEqual(order[:4], [
            "left_shoulder.x_centered_scaled", "left_shoulder.y_centered_scaled",
            "left_shoulder.confidence", "left_shoulder.mask",
        ])
        self.assertEqual(order[48], "left_shoulder__right_shoulder.dx_scaled")

    def test_relative_geometry_and_angles_are_extracted(self):
        vector = extract_features(camera_sample())
        self.assertAlmostEqual(vector[0], -0.08 / 0.7)
        self.assertAlmostEqual(vector[1], -0.30 / 0.7)
        self.assertAlmostEqual(vector[48], 0.16 / 0.7)
        self.assertEqual(vector[51], 1.0)  # 0 vertical, 1 horizontal.
        self.assertEqual(vector[48 + 4], 1.0)
        self.assertEqual(feature_metadata()["feature_count"], len(vector))
        self.assertIn("0 vertical", feature_metadata()["angle_convention"])

    def test_feature_order_is_stable_and_no_values_are_non_finite(self):
        samples = [camera_sample() for _ in range(3)]
        vectors = [extract_features(sample) for sample in samples]
        self.assertEqual(vectors[0], vectors[1])
        self.assertTrue(all(value == value for value in vectors[0]))


class SplitAndTrainingGateTests(unittest.TestCase):
    def test_same_subject_across_sessions_is_one_group(self):
        records = [
            {"subject_id": "P001", "session_id": "S001"},
            {"subject_id": "P001", "session_id": "S002"},
            {"subject_id": "P002", "session_id": "S003"},
        ]
        groups = connected_group_ids(records)
        self.assertEqual(groups[0], groups[1])
        self.assertNotEqual(groups[0], groups[2])

    def test_shared_session_connects_participants_and_prevents_leakage(self):
        records = [
            {"subject_id": "P001", "session_id": "S_SHARED"},
            {"subject_id": "P002", "session_id": "S_SHARED"},
        ]
        self.assertEqual(connected_group_ids(records)[0], connected_group_ids(records)[1])

    def test_train_validation_test_reject_shared_subject_or_session(self):
        good = {
            "train": [{"subject_id": "P001", "session_id": "S001"}],
            "validation": [{"subject_id": "P002", "session_id": "S002"}],
            "test": [{"subject_id": "P003", "session_id": "S003"}],
        }
        self.assertEqual(validate_split_disjoint(good), [])
        bad = dict(good)
        bad["test"] = [{"subject_id": "P001", "session_id": "S003"}]
        self.assertTrue(any("subject_id" in error for error in validate_split_disjoint(bad)))

    def test_class_weights_compensate_for_imbalance(self):
        weights = balanced_class_weights(["standing"] * 10 + ["other"] * 2)
        self.assertAlmostEqual(weights["standing"], 0.6)
        self.assertAlmostEqual(weights["other"], 3.0)

    def test_small_dataset_is_rejected_before_training(self):
        records = [
            {"subject_id": "P001", "session_id": "S001", "posture_label": "standing"},
            {"subject_id": "P002", "session_id": "S002", "posture_label": "sitting"},
        ]
        reasons = diversity_reasons(records)
        self.assertTrue(any("participantes" in reason for reason in reasons))
        self.assertTrue(any("classe lying" in reason for reason in reasons))

    def test_class_presence_is_checked_across_independent_groups(self):
        records = []
        for subject_index in range(5):
            for label in POSTURE_LABELS:
                for sample_index in range(4):
                    records.append({
                        "subject_id": f"P{subject_index:03}",
                        "session_id": f"S{subject_index:03}",
                        "posture_label": label,
                        "sample_id": f"sample_{subject_index}_{label}_{sample_index}",
                    })
        self.assertEqual(diversity_reasons(records), [])


class EvaluationTests(unittest.TestCase):
    def test_viewpoint_breakdown_reports_independent_scores(self):
        summary = accuracy_by_group(
            ["standing", "sitting", "standing"],
            ["standing", "standing", "lying"],
            ["front", "front", "left_profile"],
        )
        self.assertEqual(summary["front"]["samples"], 2)
        self.assertEqual(summary["front"]["accuracy"], 0.5)
        self.assertEqual(summary["left_profile"]["accuracy"], 0.0)

    def test_export_requires_model_evaluation_and_explicit_human_review(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            model = run_dir / "posture_model.h5"
            model.write_bytes(b"unit-test-placeholder")
            model_hash = sha256_file(model)
            training = {
                "run_id": "test-run",
                "evaluation_status": "EVALUATED_REVIEW_REQUIRED",
                "dataset_fingerprint": "abc123",
                "class_order": list(POSTURE_LABELS),
                "feature_transform": {"version": FEATURE_TRANSFORM_VERSION},
                "model_files": {model.name: model_hash},
            }
            evaluation = {
                "evaluation_completed": True,
                "human_review_status": "REVIEW_REQUIRED",
                "split_grouped_by_participant_and_session": True,
                "model_sha256": model_hash,
                "dataset_fingerprint": "abc123",
                "feature_transform_version": FEATURE_TRANSFORM_VERSION,
                "test_metrics": {
                    "sample_count": 4,
                    "class_order": list(POSTURE_LABELS),
                    "per_class": {label: {"support": 1} for label in POSTURE_LABELS},
                },
                "by_challenge_condition": {label: {"sample_count": 0} for label in CHALLENGE_LABELS},
            }
            (run_dir / "training-metadata.json").write_text(json.dumps(training), encoding="utf-8")
            (run_dir / "evaluation.json").write_text(json.dumps(evaluation), encoding="utf-8")
            allowed_without_review, reasons = export_eligibility(run_dir, False)
            self.assertFalse(allowed_without_review)
            self.assertTrue(any("confirme explicitamente" in reason for reason in reasons))
            allowed_after_review, reasons = export_eligibility(run_dir, True)
            self.assertTrue(allowed_after_review, reasons)

            evaluation["feature_transform_version"] = "pose-features-v0"
            (run_dir / "evaluation.json").write_text(json.dumps(evaluation), encoding="utf-8")
            incompatible, reasons = export_eligibility(run_dir, True)
            self.assertFalse(incompatible)
            self.assertTrue(any("outra versão de transformação" in reason for reason in reasons))

    def test_export_gate_rejects_holdout_missing_a_class(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            model = run_dir / "posture_model.h5"
            model.write_bytes(b"placeholder")
            model_hash = sha256_file(model)
            training = {
                "evaluation_status": "EVALUATED_REVIEW_REQUIRED",
                "dataset_fingerprint": "abc",
                "class_order": list(POSTURE_LABELS),
                "feature_transform": {"version": FEATURE_TRANSFORM_VERSION},
                "model_files": {model.name: model_hash},
            }
            evaluation = {
                "evaluation_completed": True,
                "human_review_status": "REVIEW_REQUIRED",
                "split_grouped_by_participant_and_session": True,
                "model_sha256": model_hash,
                "dataset_fingerprint": "abc",
                "feature_transform_version": FEATURE_TRANSFORM_VERSION,
                "test_metrics": {
                    "sample_count": 3,
                    "class_order": list(POSTURE_LABELS),
                    "per_class": {"standing": {"support": 3}},
                },
            }
            (run_dir / "training-metadata.json").write_text(json.dumps(training), encoding="utf-8")
            (run_dir / "evaluation.json").write_text(json.dumps(evaluation), encoding="utf-8")
            allowed, reasons = export_eligibility(run_dir, True)
        self.assertFalse(allowed)
        self.assertTrue(any("holdout sem suporte" in reason for reason in reasons))


if __name__ == "__main__":
    unittest.main()
