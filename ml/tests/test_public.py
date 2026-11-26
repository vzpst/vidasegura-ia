"""Testes da conversão do MPII e da exportação da floresta para o navegador."""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ROOT / "scripts"))

from common import POSTURE_LABELS, validate_record  # noqa: E402
from import_mpii import activity_allows, geometric_label, to_record  # noqa: E402
from train_public import export_forest, public_gate, to_matrix  # noqa: E402


def body(points: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float, bool]]:
    return {name: (x, y, True) for name, (x, y) in points.items()}


STANDING = body({
    "left_shoulder": (120, 100), "right_shoulder": (80, 100), "left_hip": (115, 220), "right_hip": (85, 220),
    "left_knee": (115, 320), "right_knee": (85, 320), "left_ankle": (115, 420), "right_ankle": (85, 420),
    "left_elbow": (125, 160), "right_elbow": (75, 160), "left_wrist": (125, 210), "right_wrist": (75, 210),
})
LYING = body({
    "left_shoulder": (100, 300), "right_shoulder": (100, 280), "left_hip": (220, 300), "right_hip": (220, 280),
    "left_knee": (320, 302), "right_knee": (320, 282), "left_ankle": (420, 305), "right_ankle": (420, 285),
})
SITTING = body({
    "left_shoulder": (120, 100), "right_shoulder": (80, 100), "left_hip": (115, 220), "right_hip": (85, 220),
    "left_knee": (215, 225), "right_knee": (185, 225), "left_ankle": (215, 320), "right_ankle": (185, 320),
})


class GeometricLabelTests(unittest.TestCase):
    def test_strict_geometry_labels(self):
        self.assertEqual(geometric_label(STANDING), "standing")
        self.assertEqual(geometric_label(LYING), "lying")
        self.assertEqual(geometric_label(SITTING), "sitting")

    def test_missing_legs_is_ambiguous(self):
        torso_only = {name: value for name, value in STANDING.items() if "hip" in name or "shoulder" in name}
        self.assertIsNone(geometric_label(torso_only))

    def test_activity_must_agree_for_standing_and_sitting(self):
        self.assertTrue(activity_allows("standing", "standing, talking in person"))
        self.assertFalse(activity_allows("standing", "eating, sitting"))
        self.assertFalse(activity_allows("sitting", "sitting or standing"))
        self.assertTrue(activity_allows("lying", "anything"))


class PublicRecordTests(unittest.TestCase):
    def test_public_record_passes_only_the_public_gate(self):
        record = to_record(STANDING, "standing", "front", "mpii_00001_00", "MPII_V1", random.Random(1))
        self.assertEqual(public_gate(record), [])
        self.assertTrue(any("data_origin" in reason for reason in validate_record(record)))
        coordinates = [p["x"] for p in record["keypoints"].values()] + [p["y"] for p in record["keypoints"].values()]
        self.assertTrue(all(0 <= value <= 1 for value in coordinates))

    def test_export_forest_format(self):
        from sklearn.ensemble import RandomForestClassifier

        rng = random.Random(2)
        records = [to_record(skeleton, label, "front", f"mpii_{i:05d}", f"MPII_V{i}", rng)
                   for i in range(12) for skeleton, label in ((STANDING, "standing"), (LYING, "lying"),
                                                               (SITTING, "sitting"), (STANDING, "other"))]
        features, labels = to_matrix(records)
        model = RandomForestClassifier(n_estimators=3, max_depth=3, random_state=0).fit(features, labels)
        exported = export_forest(model, {})
        self.assertEqual(exported["labels"], list(POSTURE_LABELS))
        self.assertEqual(exported["feature_count"], 118)
        self.assertEqual(len(exported["trees"]), 3)
        tree = exported["trees"][0]
        self.assertEqual(len(tree["feature"]), len(tree["value"]))
        self.assertAlmostEqual(sum(tree["value"][0]), 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
