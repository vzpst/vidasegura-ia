"""Converte as anotações públicas do MPII Human Pose para o JSONL do projeto.

Fonte: MPII Human Pose Dataset (Andriluka et al., CVPR 2014). As anotações são
distribuídas sob a licença BSD simplificada; as imagens NÃO são usadas nem
baixadas. Uso acadêmico, citando o artigo (ver ml/README.md).

Como os rótulos são obtidos (é importante explicar isso na apresentação):
  * O MPII traz articulações marcadas à mão e uma ATIVIDADE por imagem
    (ex.: "sitting quietly"), não a postura de cada pessoa.
  * "standing" e "sitting" exigem as DUAS coisas: atividade compatível e
    geometria estrita do esqueleto completo.
  * "lying" e "other" (ajoelhado / inclinado para frente) vêm da geometria
    estrita do esqueleto completo, porque o MPII quase não tem atividades de
    pessoa deitada. Pessoas ambíguas são descartadas.
  * O ponto de vista é ESTIMADO pela largura dos ombros e pela visibilidade.
  * A confiança de cada ponto é simulada (o MPII não tem confiança): pontos
    visíveis recebem 0,50–0,95 e pontos ocultos 0,15–0,45, imitando o MoveNet.

Os registros saem com ``data_origin = "public_mpii"``. O pipeline de coleta
continua recusando essa origem; só scripts que pedem dados públicos de forma
explícita (``train_public.py``) os aceitam.

Uso:
    python ml/scripts/import_mpii.py ml/data/public/mpii/.../mpii_human_pose_v1_u12_1.mat \
        --saida ml/data/public/mpii_posturas.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

from common import KEYPOINT_NAMES, SCHEMA_VERSION, VISIBILITY_CONFIDENCE_THRESHOLD

PUBLIC_ORIGIN = "public_mpii"
PUBLIC_CAPTURE_METHOD = "manual_annotation_mpii"

# Índices das articulações no MPII (direita/esquerda da própria pessoa, como no MoveNet).
MPII_JOINTS = {
    "right_ankle": 0, "right_knee": 1, "right_hip": 2, "left_hip": 3, "left_knee": 4, "left_ankle": 5,
    "right_wrist": 10, "right_elbow": 11, "right_shoulder": 12,
    "left_shoulder": 13, "left_elbow": 14, "left_wrist": 15,
}
FRAME_WIDTH, FRAME_HEIGHT = 640, 480  # quadro virtual 4:3, igual ao da coleta


def _angle_from_vertical(top, bottom) -> float:
    dx, dy = bottom[0] - top[0], bottom[1] - top[1]
    return math.degrees(math.atan2(abs(dx), abs(dy)))


def _mid(points):
    points = [point for point in points if point is not None]
    if not points:
        return None
    return (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))


def geometric_label(joints: dict[str, tuple[float, float, bool]]) -> str | None:
    """Rótulo estrito a partir do esqueleto anotado; None quando ambíguo."""
    xy = {name: (value[0], value[1]) for name, value in joints.items()}
    shoulders = _mid([xy.get("left_shoulder"), xy.get("right_shoulder")])
    hips = _mid([xy.get("left_hip"), xy.get("right_hip")])
    if shoulders is None or hips is None:
        return None
    torso_length = math.dist(shoulders, hips)
    if torso_length < 1:
        return None
    torso = _angle_from_vertical(shoulders, hips)
    upright = shoulders[1] < hips[1]

    thighs, shins, knees_below_hips = [], [], []
    for side in ("left", "right"):
        hip, knee, ankle = xy.get(f"{side}_hip"), xy.get(f"{side}_knee"), xy.get(f"{side}_ankle")
        if hip and knee:
            thighs.append(_angle_from_vertical(hip, knee))
            knees_below_hips.append(knee[1] > hip[1])
            if ankle:
                shins.append(_angle_from_vertical(knee, ankle))
    if not thighs:
        return None

    if torso > 60 and min(thighs) > 50:
        return "lying"
    if not upright:
        return None
    if torso < 20 and max(thighs) < 25 and shins and max(shins) < 25 and all(knees_below_hips):
        return "standing"
    if torso < 35 and min(thighs) > 55:
        return "sitting"
    if torso < 35 and max(thighs) < 35 and shins and min(shins) > 55:
        return "other"  # ajoelhado
    if torso > 50 and max(thighs) < 30 and shins and max(shins) < 30:
        return "other"  # tronco bem inclinado para frente, pernas estendidas
    return None


def activity_allows(label: str, activity: str) -> bool:
    activity = activity.lower()
    if label == "standing":
        return "standing" in activity and "sitting" not in activity
    if label == "sitting":
        return "sitting" in activity and "standing" not in activity and "reclin" not in activity
    return True  # lying / other dependem só da geometria estrita


def estimate_viewpoint(joints: dict[str, tuple[float, float, bool]]) -> str:
    xy = {name: (value[0], value[1]) for name, value in joints.items()}
    if "left_shoulder" not in xy or "right_shoulder" not in xy:
        return "oblique"
    shoulders = _mid([xy["left_shoulder"], xy["right_shoulder"]])
    hips = _mid([xy.get("left_hip"), xy.get("right_hip")])
    torso_length = math.dist(shoulders, hips) if hips else 0
    if torso_length < 1:
        return "oblique"
    ratio = math.dist(xy["left_shoulder"], xy["right_shoulder"]) / torso_length
    if ratio >= 0.55:
        return "front"
    if ratio <= 0.25:
        left = sum(1 for name, value in joints.items() if name.startswith("left_") and value[2])
        right = sum(1 for name, value in joints.items() if name.startswith("right_") and value[2])
        if left != right:
            return "left_profile" if left > right else "right_profile"
    return "oblique"


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    try:
        return list(value)
    except TypeError:
        return [value]


def read_people(annotation) -> dict[str, tuple[float, float, bool]] | None:
    points = getattr(getattr(annotation, "annopoints", None), "point", None)
    if points is None or isinstance(points, (list, tuple)) and not points:
        return None
    joints: dict[str, tuple[float, float, bool]] = {}
    by_id = {}
    for point in _as_list(points):
        try:
            by_id[int(point.id)] = point
        except (AttributeError, TypeError, ValueError):
            continue
    for name, index in MPII_JOINTS.items():
        point = by_id.get(index)
        if point is None:
            continue
        visible = getattr(point, "is_visible", 1)
        try:
            visible = bool(int(visible))
        except (TypeError, ValueError):
            visible = True  # o campo vem vazio em parte das anotações
        joints[name] = (float(point.x), float(point.y), visible)
    return joints


def to_record(joints, label, viewpoint, sample_id, group_id, rng) -> dict[str, Any] | None:
    xs = [value[0] for value in joints.values()]
    ys = [value[1] for value in joints.values()]
    width, height = max(xs) - min(xs), max(ys) - min(ys)
    # Enquadra a pessoa num quadro 4:3 com margem, como alguém filmado pela câmera.
    frame_h = max(height, width * FRAME_HEIGHT / FRAME_WIDTH) * rng.uniform(1.15, 1.6)
    frame_w = frame_h * FRAME_WIDTH / FRAME_HEIGHT
    left = (min(xs) + max(xs)) / 2 - frame_w * rng.uniform(0.4, 0.6)
    top = (min(ys) + max(ys)) / 2 - frame_h * rng.uniform(0.4, 0.6)
    keypoints = {}
    for name in KEYPOINT_NAMES:
        if name not in joints:
            keypoints[name] = {"x": None, "y": None, "score": None, "validity": "missing"}
            continue
        x, y, visible = joints[name]
        score = rng.uniform(0.50, 0.95) if visible else rng.uniform(0.15, 0.45)
        keypoints[name] = {
            "x": min(1.0, max(0.0, (x - left) / frame_w)),
            "y": min(1.0, max(0.0, (y - top) / frame_h)),
            "score": score,
            "validity": "observed" if score >= VISIBILITY_CONFIDENCE_THRESHOLD else "low_confidence",
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": sample_id,
        "subject_id": group_id,
        "session_id": group_id,
        "timestamp_ms": 1,  # MPII não tem horário de captura
        "image_width": FRAME_WIDTH,
        "image_height": FRAME_HEIGHT,
        "data_origin": PUBLIC_ORIGIN,
        "capture_method": PUBLIC_CAPTURE_METHOD,
        "posture_label": label,
        "viewpoint_label": viewpoint,
        "challenge_label": "unspecified",
        "geometric_prediction": None,
        "keypoints": keypoints,
    }


def convert(mat_path: Path, seed: int = 2026, max_per_class: int | None = None) -> tuple[list[dict], dict]:
    from scipy.io import loadmat

    release = loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)["RELEASE"]
    rng = random.Random(seed)
    stats: collections.Counter = collections.Counter()
    by_label: dict[str, list[dict]] = collections.defaultdict(list)
    for image_index, (image, train_flag, activity) in enumerate(
            zip(release.annolist, release.img_train, release.act)):
        if not int(train_flag):
            continue  # o conjunto de teste oficial não tem articulações
        activity_name = activity.act_name if isinstance(activity.act_name, str) else ""
        video = getattr(image, "vididx", None)
        group_id = f"MPII_V{int(video)}" if isinstance(video, (int, float)) and not isinstance(video, bool) \
            else f"MPII_I{image_index}"
        for person_index, annotation in enumerate(_as_list(image.annorect)):
            joints = read_people(annotation)
            stats["pessoas_lidas"] += 1
            torso = {"left_shoulder", "right_shoulder", "left_hip", "right_hip"}
            if not joints or not torso <= joints.keys():
                stats["descartada_sem_tronco"] += 1
                continue
            label = geometric_label(joints)
            if label is None:
                stats["descartada_ambigua"] += 1
                continue
            if not activity_allows(label, activity_name):
                stats["descartada_atividade_incompativel"] += 1
                continue
            sample_id = f"mpii_{image_index:05d}_{person_index:02d}"
            by_label[label].append(to_record(joints, label, estimate_viewpoint(joints), sample_id, group_id, rng))
    records = []
    for label, items in sorted(by_label.items()):
        rng.shuffle(items)
        kept = items[:max_per_class] if max_per_class else items
        stats[f"rotulo_{label}"] = len(kept)
        records.extend(kept)
    return records, dict(stats)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mat", type=Path, help="mpii_human_pose_v1_u12_1.mat")
    parser.add_argument("--saida", type=Path, required=True, help="Arquivo JSONL de saída (fica fora do Git)")
    parser.add_argument("--max-por-classe", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    records, stats = convert(args.mat, args.seed, args.max_por_classe)
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    with args.saida.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"{len(records)} registros gravados em {args.saida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
