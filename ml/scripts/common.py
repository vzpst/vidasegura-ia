"""Shared, dependency-light utilities for the isolated posture ML lab."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
FEATURE_TRANSFORM_VERSION = "pose-features-v1"
VISIBILITY_CONFIDENCE_THRESHOLD = 0.25

POSTURE_LABELS = ("standing", "sitting", "lying", "other")
VIEWPOINT_LABELS = ("front", "left_profile", "right_profile", "oblique")
CHALLENGE_LABELS = (
    "unspecified", "standard", "side_lying", "squat", "kneeling", "leaning", "other",
)

# Mantém a mesma seleção de 12 articulações corporais de js/visibility.js.
KEYPOINT_NAMES = (
    "left_shoulder", "right_shoulder", "left_hip", "right_hip",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
TORSO_NAMES = frozenset(("left_shoulder", "right_shoulder", "left_hip", "right_hip"))
LEG_NAMES = frozenset(("left_knee", "right_knee", "left_ankle", "right_ankle"))
KEYPOINT_VALIDITY = ("observed", "low_confidence", "missing")
GEOMETRIC_PREDICTIONS = (*POSTURE_LABELS[:3], "inconclusive")

# Ordem versionada das relações geométricas. Cada par gera dx, dy, distância,
# ângulo em relação à vertical e máscara de disponibilidade.
GEOMETRY_PAIRS = (
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "left_knee"),
    ("right_hip", "right_knee"),
    ("left_knee", "left_ankle"),
    ("right_knee", "right_ankle"),
    ("left_shoulder", "left_elbow"),
    ("right_shoulder", "right_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "right_hip"),
    ("right_shoulder", "left_hip"),
)

MIN_SUBJECTS_FOR_TRAINING = 5
MIN_SESSIONS_FOR_TRAINING = 5
MIN_GROUPS_FOR_TRAINING = 5
MIN_SAMPLES_PER_CLASS = 20
MIN_GROUPS_PER_CLASS = 3
MIN_OBSERVED_KEYPOINTS_FOR_SAMPLE = 4

_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _safe_id(value: Any) -> bool:
    return isinstance(value, str) and bool(_SAFE_ID.fullmatch(value))


def validate_record(record: Any) -> list[str]:
    """Return structural/schema errors for one JSONL sample."""
    errors: list[str] = []
    if not isinstance(record, dict):
        return ["linha JSON precisa ser um objeto"]

    required = (
        "schema_version", "sample_id", "subject_id", "session_id", "timestamp_ms",
        "image_width", "image_height", "posture_label", "viewpoint_label",
        "data_origin", "capture_method", "keypoints",
    )
    allowed = {*required, "geometric_prediction", "challenge_label", "_source_file", "_source_line"}
    for field in required:
        if field not in record:
            errors.append(f"campo obrigatório ausente: {field}")
    unknown_fields = sorted(set(record) - allowed)
    if unknown_fields:
        errors.append(f"campos fora do esquema: {', '.join(unknown_fields)}")

    if (not isinstance(record.get("schema_version"), int)
            or isinstance(record.get("schema_version"), bool)
            or record.get("schema_version") != SCHEMA_VERSION):
        errors.append(f"schema_version precisa ser {SCHEMA_VERSION}")
    for field in ("sample_id", "subject_id", "session_id"):
        if not _safe_id(record.get(field)):
            errors.append(f"{field} precisa ser um identificador anônimo alfanumérico")
    if not isinstance(record.get("timestamp_ms"), int) or isinstance(record.get("timestamp_ms"), bool) or record.get("timestamp_ms", 0) <= 0:
        errors.append("timestamp_ms precisa ser um inteiro positivo")
    for field in ("image_width", "image_height"):
        if not isinstance(record.get(field), int) or isinstance(record.get(field), bool) or record.get(field, 0) <= 0:
            errors.append(f"{field} precisa ser um inteiro positivo")
    if record.get("posture_label") not in POSTURE_LABELS:
        errors.append(f"posture_label inválida; classes permitidas: {', '.join(POSTURE_LABELS)}")
    if record.get("viewpoint_label") not in VIEWPOINT_LABELS:
        errors.append(f"viewpoint_label inválida; classes permitidas: {', '.join(VIEWPOINT_LABELS)}")
    if "challenge_label" in record and record.get("challenge_label") not in CHALLENGE_LABELS:
        errors.append(f"challenge_label inválida; opções permitidas: {', '.join(CHALLENGE_LABELS)}")
    if record.get("data_origin") != "camera":
        errors.append("data_origin precisa ser 'camera'; amostras sintéticas não podem treinar o modelo")
    if record.get("capture_method") != "movenet_singlepose_lightning":
        errors.append("capture_method precisa ser 'movenet_singlepose_lightning'")

    prediction = record.get("geometric_prediction")
    if "geometric_prediction" in record and prediction is not None and prediction not in GEOMETRIC_PREDICTIONS:
        errors.append("geometric_prediction inválida")

    keypoints = record.get("keypoints")
    if not isinstance(keypoints, dict):
        errors.append("keypoints precisa ser um objeto com as 12 articulações obrigatórias")
        return errors

    for name in KEYPOINT_NAMES:
        point = keypoints.get(name)
        if not isinstance(point, dict):
            errors.append(f"articulação ausente do esquema: {name}")
            continue
        for field in ("x", "y", "score", "validity"):
            if field not in point:
                errors.append(f"{name}: campo obrigatório ausente: {field}")

        validity = point.get("validity")
        if validity not in KEYPOINT_VALIDITY:
            errors.append(f"{name}: validity inválido")
            continue
        x, y, score = point.get("x"), point.get("y"), point.get("score")
        if validity == "missing":
            if x is not None or y is not None:
                errors.append(f"{name}: articulação missing deve ter x e y nulos")
            if score is not None and (not _finite_number(score) or not 0 <= score <= 1):
                errors.append(f"{name}: score original precisa estar entre 0 e 1 ou ser nulo")
            continue

        if not _finite_number(x) or not _finite_number(y):
            errors.append(f"{name}: coordenadas observadas precisam ser números finitos")
        elif not (0 <= x <= 1 and 0 <= y <= 1):
            errors.append(f"{name}: coordenadas normalizadas precisam estar entre 0 e 1")
        if not _finite_number(score) or not 0 <= score <= 1:
            errors.append(f"{name}: score precisa estar entre 0 e 1")
        elif validity == "observed" and score < VISIBILITY_CONFIDENCE_THRESHOLD:
            errors.append(f"{name}: observed exige score >= {VISIBILITY_CONFIDENCE_THRESHOLD}")
        elif validity == "low_confidence" and score >= VISIBILITY_CONFIDENCE_THRESHOLD:
            errors.append(f"{name}: low_confidence exige score < {VISIBILITY_CONFIDENCE_THRESHOLD}")

    unknown = sorted(set(keypoints) - set(KEYPOINT_NAMES))
    if unknown:
        errors.append(f"keypoints contém nomes fora do esquema: {', '.join(unknown)}")
    return errors


def point_is_observed(point: Any) -> bool:
    return bool(
        isinstance(point, dict)
        and point.get("validity") == "observed"
        and _finite_number(point.get("x"))
        and _finite_number(point.get("y"))
        and _finite_number(point.get("score"))
        and point["score"] >= VISIBILITY_CONFIDENCE_THRESHOLD
    )


def training_sample_reasons(record: dict[str, Any]) -> list[str]:
    """Sample-level usability gate; absent joints remain maskable features."""
    reasons = validate_record(record)
    if reasons:
        return reasons
    keypoints = record["keypoints"]
    observed = [name for name in KEYPOINT_NAMES if point_is_observed(keypoints[name])]
    if len(observed) < MIN_OBSERVED_KEYPOINTS_FOR_SAMPLE:
        reasons.append(f"menos de {MIN_OBSERVED_KEYPOINTS_FOR_SAMPLE} articulações observadas")
    if not TORSO_NAMES.intersection(observed):
        reasons.append("nenhuma articulação observada do tronco")
    if not LEG_NAMES.intersection(observed):
        reasons.append("nenhuma articulação observada das pernas")
    return reasons


def read_jsonl_files(paths: Iterable[str | Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    read_issues: list[dict[str, Any]] = []
    for path_value in paths:
        path = Path(path_value)
        try:
            stream = path.open("r", encoding="utf-8")
        except OSError as error:
            read_issues.append({"file": str(path), "line": None, "reason": f"não foi possível ler: {error}"})
            continue
        with stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    read_issues.append({
                        "file": str(path), "line": line_number,
                        "reason": f"JSON inválido: {error.msg}",
                    })
                    continue
                if isinstance(value, dict):
                    value = dict(value)
                    value["_source_file"] = str(path)
                    value["_source_line"] = line_number
                records.append(value)
    return records, read_issues


def observed_keypoint_count(record: dict[str, Any]) -> int:
    keypoints = record.get("keypoints", {}) if isinstance(record, dict) else {}
    if not isinstance(keypoints, dict):
        return 0
    return sum(point_is_observed(keypoints.get(name)) for name in KEYPOINT_NAMES)


def duplicate_groups(records: Iterable[dict[str, Any]]) -> list[list[str]]:
    fingerprints: dict[str, list[str]] = defaultdict(list)
    for record in records:
        if not isinstance(record, dict):
            continue
        identity = {
            "subject_id": record.get("subject_id"),
            "session_id": record.get("session_id"),
            "posture_label": record.get("posture_label"),
            "viewpoint_label": record.get("viewpoint_label"),
            "keypoints": record.get("keypoints"),
        }
        fingerprint = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        fingerprints[fingerprint].append(str(record.get("sample_id", "sem_id")))
    return [sample_ids for sample_ids in fingerprints.values() if len(sample_ids) > 1]


def dataset_fingerprint(records: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        clean = {key: value for key, value in record.items() if not key.startswith("_")}
        digest.update(json.dumps(clean, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def feature_order() -> list[str]:
    names: list[str] = []
    for point_name in KEYPOINT_NAMES:
        names.extend((
            f"{point_name}.x_centered_scaled",
            f"{point_name}.y_centered_scaled",
            f"{point_name}.confidence",
            f"{point_name}.mask",
        ))
    for first, second in GEOMETRY_PAIRS:
        pair = f"{first}__{second}"
        names.extend((
            f"{pair}.dx_scaled", f"{pair}.dy_scaled", f"{pair}.distance_scaled",
            f"{pair}.angle_from_vertical_norm", f"{pair}.mask",
        ))
    return names


def extract_features(record: dict[str, Any]) -> list[float]:
    """Create the fixed v1 vector without fabricating coordinates for missing joints."""
    keypoints = record.get("keypoints", {})
    observed = {
        name: keypoints.get(name)
        for name in KEYPOINT_NAMES
        if point_is_observed(keypoints.get(name))
    }

    center_points = [observed[name] for name in ("left_hip", "right_hip") if name in observed]
    if not center_points:
        center_points = [observed[name] for name in ("left_shoulder", "right_shoulder") if name in observed]
    if not center_points:
        center_points = list(observed.values())
    if center_points:
        center_x = sum(point["x"] for point in center_points) / len(center_points)
        center_y = sum(point["y"] for point in center_points) / len(center_points)
        xs = [point["x"] for point in observed.values()]
        ys = [point["y"] for point in observed.values()]
        scale = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)
    else:
        center_x = center_y = 0.5
        scale = 1.0

    vector: list[float] = []
    for name in KEYPOINT_NAMES:
        point = observed.get(name)
        if point is None:
            vector.extend((0.0, 0.0, _safe_confidence(keypoints.get(name)), 0.0))
        else:
            vector.extend(((point["x"] - center_x) / scale,
                           (point["y"] - center_y) / scale,
                           float(point["score"]), 1.0))

    for first_name, second_name in GEOMETRY_PAIRS:
        first, second = observed.get(first_name), observed.get(second_name)
        if first is None or second is None:
            vector.extend((0.0, 0.0, 0.0, 0.0, 0.0))
            continue
        dx = second["x"] - first["x"]
        dy = second["y"] - first["y"]
        length = math.hypot(dx, dy)
        angle = math.atan2(abs(dx), abs(dy)) / (math.pi / 2)
        vector.extend((dx / scale, dy / scale, length / scale, angle, 1.0))

    if len(vector) != len(feature_order()):
        raise AssertionError("feature vector and documented order diverged")
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("feature transform generated a non-finite value")
    return vector


def _safe_confidence(point: Any) -> float:
    if isinstance(point, dict) and _finite_number(point.get("score")) and 0 <= point["score"] <= 1:
        return float(point["score"])
    return 0.0


def feature_metadata() -> dict[str, Any]:
    return {
        "version": FEATURE_TRANSFORM_VERSION,
        "schema_version": SCHEMA_VERSION,
        "keypoint_order": list(KEYPOINT_NAMES),
        "geometry_pair_order": [list(pair) for pair in GEOMETRY_PAIRS],
        "features_per_keypoint": ["x_centered_scaled", "y_centered_scaled", "confidence", "mask"],
        "features_per_pair": ["dx_scaled", "dy_scaled", "distance_scaled", "angle_from_vertical_norm", "mask"],
        "angle_convention": "0 vertical, 90 degrees horizontal; stored normalized to [0, 1]",
        "confidence_threshold": VISIBILITY_CONFIDENCE_THRESHOLD,
        "feature_order": feature_order(),
        "feature_count": len(feature_order()),
        "missing_coordinate_policy": "zero neutral values with mask=0; no coordinate is estimated",
        "center_policy": "mean of observed hips, otherwise observed shoulders, otherwise observed points",
        "scale_policy": "maximum observed x/y extent, lower bounded by 1e-6",
    }


def connected_group_ids(records: list[dict[str, Any]]) -> list[str]:
    """Join rows sharing a participant or session to prevent train/test leakage."""
    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        root_first, root_second = find(first), find(second)
        if root_first != root_second:
            parent[root_second] = root_first

    owner: dict[tuple[str, str], int] = {}
    for index, record in enumerate(records):
        for kind in ("subject_id", "session_id"):
            value = record.get(kind)
            if value is None:
                continue
            key = (kind, str(value))
            if key in owner:
                union(index, owner[key])
            else:
                owner[key] = index

    root_names: dict[int, str] = {}
    output = []
    for index in range(len(records)):
        root = find(index)
        if root not in root_names:
            root_names[root] = f"group_{len(root_names) + 1:04d}"
        output.append(root_names[root])
    return output


def diversity_reasons(records: list[dict[str, Any]]) -> list[str]:
    """Conservative preflight; thresholds are procedural, not scientific claims."""
    reasons: list[str] = []
    subjects = {record.get("subject_id") for record in records}
    sessions = {record.get("session_id") for record in records}
    groups = connected_group_ids(records)
    label_counts = Counter(record.get("posture_label") for record in records)
    groups_by_label: dict[str, set[str]] = defaultdict(set)
    for record, group_id in zip(records, groups):
        groups_by_label[str(record.get("posture_label"))].add(group_id)

    if len(subjects) < MIN_SUBJECTS_FOR_TRAINING:
        reasons.append(f"exige ao menos {MIN_SUBJECTS_FOR_TRAINING} participantes distintos")
    if len(sessions) < MIN_SESSIONS_FOR_TRAINING:
        reasons.append(f"exige ao menos {MIN_SESSIONS_FOR_TRAINING} sessões distintas")
    if len(set(groups)) < MIN_GROUPS_FOR_TRAINING:
        reasons.append(f"exige ao menos {MIN_GROUPS_FOR_TRAINING} grupos independentes de participante/sessão")
    for label in POSTURE_LABELS:
        if label_counts[label] < MIN_SAMPLES_PER_CLASS:
            reasons.append(f"classe {label} tem {label_counts[label]} amostras; mínimo provisório: {MIN_SAMPLES_PER_CLASS}")
        if len(groups_by_label[label]) < MIN_GROUPS_PER_CLASS:
            reasons.append(f"classe {label} aparece em menos de {MIN_GROUPS_PER_CLASS} grupos independentes")
    return reasons


def balanced_class_weights(labels: Iterable[str]) -> dict[str, float]:
    counts = Counter(labels)
    present = len(counts)
    total = sum(counts.values())
    return {label: total / (present * count) for label, count in sorted(counts.items()) if count > 0}


def accuracy_by_group(truth: list[str], predictions: list[str], groups: list[str]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    by_group: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for actual, predicted, group in zip(truth, predictions, groups):
        by_group[group].append((actual, predicted))
    for group, rows in sorted(by_group.items()):
        correct = sum(actual == predicted for actual, predicted in rows)
        summary[group] = {"samples": len(rows), "accuracy": correct / len(rows)}
    return summary


def validate_split_disjoint(splits: dict[str, list[dict[str, Any]]]) -> list[str]:
    errors: list[str] = []
    split_names = list(splits)
    for index, first_name in enumerate(split_names):
        for second_name in split_names[index + 1:]:
            for field in ("subject_id", "session_id"):
                first_values = {record.get(field) for record in splits[first_name]}
                second_values = {record.get(field) for record in splits[second_name]}
                overlap = first_values & second_values
                if overlap:
                    errors.append(f"{field} compartilhado entre {first_name} e {second_name}: {sorted(overlap)}")
    return errors
