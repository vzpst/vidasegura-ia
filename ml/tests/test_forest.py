"""Testes da oclusão simulada e do treino Random Forest (dados sintéticos só como entrada de teste)."""

from __future__ import annotations

import copy
import random
import sys
import tempfile
import unittest
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ROOT / "scripts"))

from common import KEYPOINT_NAMES, extract_features, training_sample_reasons, validate_record  # noqa: E402
from occlusion import (  # noqa: E402
    LEFT_SIDE,
    OCCLUSION_SCENARIOS,
    RIGHT_SIDE,
    augment_records,
    hide,
    lower_confidence,
)
from synthetic_demo import make_dataset, make_record  # noqa: E402


def camera_record(posture: str = "standing") -> dict:
    record = make_record(posture, "front", 1, 1, random.Random(1))
    record["data_origin"] = "camera"
    return record


class OcclusionTests(unittest.TestCase):
    def test_hide_marks_joints_missing_without_mutating_original(self):
        record = camera_record()
        original = copy.deepcopy(record)
        hidden = hide(record, RIGHT_SIDE)
        self.assertEqual(record, original)
        for name in RIGHT_SIDE:
            self.assertEqual(hidden["keypoints"][name], {"x": None, "y": None, "score": None, "validity": "missing"})
        for name in LEFT_SIDE:
            self.assertEqual(hidden["keypoints"][name], record["keypoints"][name])
        self.assertEqual(validate_record(hidden), [])

    def test_hidden_joints_get_mask_zero_in_features(self):
        hidden = hide(camera_record(), ["left_knee"])
        vector = extract_features(hidden)
        index = KEYPOINT_NAMES.index("left_knee") * 4
        self.assertEqual(vector[index:index + 4], [0.0, 0.0, 0.0, 0.0])

    def test_lower_confidence_reclassifies_validity(self):
        lowered = lower_confidence(camera_record(), KEYPOINT_NAMES, 0.1)
        self.assertTrue(all(point["validity"] == "low_confidence" for point in lowered["keypoints"].values()))
        self.assertEqual(validate_record(lowered), [])

    def test_every_scenario_produces_schema_valid_records(self):
        rng = random.Random(3)
        for name, scenario in OCCLUSION_SCENARIOS.items():
            with self.subTest(name=name):
                self.assertEqual(validate_record(scenario(camera_record(), rng)), [])

    def test_augmentation_is_deterministic_and_keeps_only_usable_copies(self):
        records = [camera_record(label) for label in ("standing", "sitting", "lying", "other")]
        first = augment_records(records, copies_per_record=3, seed=11)
        second = augment_records(records, copies_per_record=3, seed=11)
        self.assertEqual(first, second)
        self.assertGreater(len(first), len(records))
        self.assertEqual(first[:len(records)], records)
        self.assertTrue(all(not training_sample_reasons(record) for record in first))


class SyntheticDemoTests(unittest.TestCase):
    def test_synthetic_records_are_refused_by_the_real_pipeline(self):
        record = make_dataset(subjects=1, per_class_view=1)[0]
        self.assertTrue(any("data_origin" in reason for reason in validate_record(record)))


class ForestTrainingTests(unittest.TestCase):
    def test_demo_run_produces_report_with_baselines_and_occlusion_scenarios(self):
        import train_forest

        with tempfile.TemporaryDirectory() as directory:
            exit_code = train_forest.main(["--demo-sintetico", "--folds", "3", "--output-dir", directory])
            self.assertEqual(exit_code, 0)
            run_dirs = list(Path(directory).iterdir())
            self.assertEqual(len(run_dirs), 1)
            import json

            report = json.loads((run_dirs[0] / "relatorio.json").read_text(encoding="utf-8"))
            self.assertTrue(report["dados_sinteticos"])
            self.assertIn("NÃO usar", report["aviso"])
            self.assertEqual(set(report["macro_f1"]), {"regressao_logistica", "random_forest"})
            self.assertIn("aleatorio_50", report["macro_f1"]["random_forest"])
            self.assertTrue((run_dirs[0] / "relatorio.md").is_file())
            self.assertTrue((run_dirs[0] / "random_forest.joblib").is_file())
            for fold in report["folds"]:
                self.assertTrue(fold["grupos_teste"])

    def test_training_refuses_small_real_dataset(self):
        import json
        import train_forest

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "small.jsonl"
            path.write_text("\n".join(json.dumps(camera_record(label)) for label in
                                      ("standing", "sitting", "lying", "other")) + "\n", encoding="utf-8")
            with self.assertRaises(SystemExit) as raised:
                train_forest.main([str(path), "--output-dir", directory])
            self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
