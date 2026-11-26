"""Oclusão simulada para ensinar o classificador a lidar com corpo parcialmente visível.

Cada função recebe um registro JSONL (dict) e devolve uma CÓPIA com algumas
articulações escondidas, sem alterar o original. As articulações escondidas
ficam com ``validity = "missing"`` e coordenadas nulas, exatamente como a
ferramenta de coleta registra um ponto que o MoveNet não encontrou.

Uso correto: aplicar somente ao conjunto de TREINO de cada fold. No teste a
oclusão serve apenas para medir robustez, nunca para treinar.
"""

from __future__ import annotations

import copy
import random
from typing import Any, Callable, Iterable

from common import KEYPOINT_NAMES, VISIBILITY_CONFIDENCE_THRESHOLD, training_sample_reasons

LEFT_SIDE = tuple(name for name in KEYPOINT_NAMES if name.startswith("left_"))
RIGHT_SIDE = tuple(name for name in KEYPOINT_NAMES if name.startswith("right_"))
LOWER_LEGS = ("left_knee", "right_knee", "left_ankle", "right_ankle")
ARMS = ("left_elbow", "right_elbow", "left_wrist", "right_wrist")


def hide(record: dict[str, Any], names: Iterable[str]) -> dict[str, Any]:
    """Return a copy where the given joints are marked as missing."""
    result = copy.deepcopy(record)
    for name in names:
        if name in result["keypoints"]:
            result["keypoints"][name] = {"x": None, "y": None, "score": None, "validity": "missing"}
    return result


def hide_random(record: dict[str, Any], fraction: float, rng: random.Random) -> dict[str, Any]:
    names = [name for name in KEYPOINT_NAMES if record["keypoints"][name].get("validity") != "missing"]
    count = int(round(len(names) * fraction))
    return hide(record, rng.sample(names, min(count, len(names))))


def lower_confidence(record: dict[str, Any], names: Iterable[str], factor: float) -> dict[str, Any]:
    """Simula baixa iluminação ou borrão: reduz a confiança e reclassifica a validade."""
    result = copy.deepcopy(record)
    for name in names:
        point = result["keypoints"].get(name)
        if not point or point.get("validity") == "missing" or point.get("score") is None:
            continue
        point["score"] = max(0.0, min(1.0, point["score"] * factor))
        point["validity"] = "observed" if point["score"] >= VISIBILITY_CONFIDENCE_THRESHOLD else "low_confidence"
    return result


# Cenários nomeados usados tanto no aumento de treino quanto na avaliação.
OCCLUSION_SCENARIOS: dict[str, Callable[[dict[str, Any], random.Random], dict[str, Any]]] = {
    "sem_oclusao": lambda record, rng: copy.deepcopy(record),
    "lado_esquerdo_oculto": lambda record, rng: hide(record, LEFT_SIDE),
    "lado_direito_oculto": lambda record, rng: hide(record, RIGHT_SIDE),
    "pernas_ocultas_parcial": lambda record, rng: hide(record, rng.sample(LOWER_LEGS, 2)),
    "bracos_ocultos": lambda record, rng: hide(record, ARMS),
    "aleatorio_25": lambda record, rng: hide_random(record, 0.25, rng),
    "aleatorio_50": lambda record, rng: hide_random(record, 0.50, rng),
    "baixa_confianca": lambda record, rng: lower_confidence(record, KEYPOINT_NAMES, rng.uniform(0.4, 0.8)),
}

TRAINING_SCENARIOS = (
    "lado_esquerdo_oculto", "lado_direito_oculto", "pernas_ocultas_parcial",
    "bracos_ocultos", "aleatorio_25", "aleatorio_50", "baixa_confianca",
)


def augment_records(
    records: list[dict[str, Any]],
    copies_per_record: int = 2,
    seed: int = 2026,
    gate: Callable[[dict[str, Any]], list[str]] = training_sample_reasons,
) -> list[dict[str, Any]]:
    """Original records plus occluded copies that still pass the training gate.

    Cópias que ficariam inutilizáveis (por exemplo, sem nenhum ponto do tronco)
    são descartadas, porque o pipeline real também as rejeitaria.
    """
    rng = random.Random(seed)
    output = list(records)
    for record in records:
        for scenario in rng.sample(TRAINING_SCENARIOS, min(copies_per_record, len(TRAINING_SCENARIOS))):
            candidate = OCCLUSION_SCENARIOS[scenario](record, rng)
            if not gate(candidate):
                candidate["sample_id"] = f"{record['sample_id']}_aug_{scenario}"[:64]
                output.append(candidate)
    return output
