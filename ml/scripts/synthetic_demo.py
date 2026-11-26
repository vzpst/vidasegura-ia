"""Gerador de esqueletos SINTÉTICOS para testar o pipeline de ponta a ponta.

ATENÇÃO: estes dados são bonecos de palito gerados por fórmula. Servem apenas
para verificar que o código roda (testes automatizados e ensaio da apresentação).
Qualquer métrica obtida com eles NÃO diz nada sobre pessoas reais e nunca deve
ser apresentada como resultado do modelo. Os registros saem com
``data_origin = "synthetic_demo"`` e por isso são recusados pelo pipeline real.
"""

from __future__ import annotations

import math
import random
from typing import Any

from common import KEYPOINT_NAMES, POSTURE_LABELS, SCHEMA_VERSION, VIEWPOINT_LABELS

# Posições-base (x relativo ao centro, y absoluto) de um lado do corpo.
_TEMPLATES = {
    "standing": {"shoulder": (0, .25), "elbow": (.02, .38), "wrist": (.02, .50),
                 "hip": (0, .52), "knee": (0, .71), "ankle": (0, .90)},
    "sitting": {"shoulder": (0, .30), "elbow": (.04, .43), "wrist": (.10, .52),
                "hip": (0, .56), "knee": (.17, .58), "ankle": (.18, .80)},
    "lying": {"shoulder": (-.25, .70), "elbow": (-.15, .72), "wrist": (-.05, .72),
              "hip": (0, .71), "knee": (.18, .72), "ankle": (.36, .73)},
    "other": {"shoulder": (0, .33), "elbow": (.03, .46), "wrist": (.03, .56),  # ajoelhado
              "hip": (0, .58), "knee": (.02, .80), "ankle": (-.18, .82)},
}


def _point(x: float, y: float, score: float) -> dict[str, Any]:
    x = min(1.0, max(0.0, x))
    y = min(1.0, max(0.0, y))
    return {"x": x, "y": y, "score": score, "validity": "observed" if score >= 0.25 else "low_confidence"}


def make_record(posture: str, viewpoint: str, subject: int, index: int, rng: random.Random) -> dict[str, Any]:
    template = _TEMPLATES[posture]
    center_x = rng.uniform(.35, .65)
    scale = rng.uniform(.8, 1.1)
    shift_y = rng.uniform(-.05, .05)
    tilt = math.radians(rng.uniform(-8, 8))
    # Largura entre lados: frente = afastados; perfil = quase sobrepostos.
    half_width = {"front": .07, "oblique": .04, "left_profile": .008, "right_profile": .008}[viewpoint]
    hidden_side = {"left_profile": "right", "right_profile": "left"}.get(viewpoint)
    keypoints: dict[str, Any] = {}
    for name in KEYPOINT_NAMES:
        side, joint = name.split("_", 1)
        base_x, base_y = template[joint]
        offset = -half_width if side == "left" else half_width
        if posture == "lying":
            px, py = base_x * scale, base_y + offset * .6
        else:
            px, py = base_x * scale + offset, base_y * scale
        rx = px * math.cos(tilt) - (py - .5) * math.sin(tilt)
        ry = px * math.sin(tilt) + (py - .5) * math.cos(tilt) + .5
        score = rng.uniform(.55, .98)
        if hidden_side == side:
            score = rng.uniform(.15, .45)  # lado de trás no perfil
        keypoints[name] = _point(center_x + rx + rng.gauss(0, .008), ry + shift_y + rng.gauss(0, .008), score)
    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": f"demo_P{subject:02d}_{index:05d}",
        "subject_id": f"DEMO_P{subject:02d}",
        "session_id": f"DEMO_S{subject:02d}",
        "timestamp_ms": 1_700_000_000_000 + index,
        "image_width": 640,
        "image_height": 480,
        "posture_label": posture,
        "viewpoint_label": viewpoint,
        "challenge_label": "unspecified",
        "data_origin": "synthetic_demo",
        "capture_method": "movenet_singlepose_lightning",
        "geometric_prediction": None,
        "keypoints": keypoints,
    }


def make_dataset(subjects: int = 6, per_class_view: int = 6, seed: int = 7) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    records = []
    index = 0
    for subject in range(1, subjects + 1):
        for posture in POSTURE_LABELS:
            for viewpoint in VIEWPOINT_LABELS:
                for _ in range(per_class_view):
                    index += 1
                    records.append(make_record(posture, viewpoint, subject, index, rng))
    return records
