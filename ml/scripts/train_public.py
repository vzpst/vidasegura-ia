"""Treina o classificador de postura com dados PÚBLICOS e testa na câmera real.

Ideia (honesta) para quem tem poucos participantes:
  1. Validação cruzada por grupo (vídeo do MPII) só dentro dos dados públicos.
  2. TESTE EXTERNO: o modelo treinado só com dados públicos é avaliado nas
     amostras reais da câmera, que ele nunca viu. É o número que importa.
     Com um único participante o resultado é apenas indicativo.
  3. O modelo final (públicos + câmera) é exportado em JSON para rodar no
     navegador ao lado da regra geométrica.

Uso:
    python ml/scripts/train_public.py --publico ml/data/public/mpii_posturas.jsonl \
        --camera "C:/caminho/para/coleta.jsonl" --exportar-navegador models/posture-forest.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import FEATURE_TRANSFORM_VERSION, POSTURE_LABELS, feature_order, read_jsonl_files, training_sample_reasons
from import_mpii import PUBLIC_CAPTURE_METHOD, PUBLIC_ORIGIN
from occlusion import OCCLUSION_SCENARIOS, augment_records
from train_forest import ML_ROOT, build_models, geometric_baseline, macro_f1, rejection_curve, to_matrix

BROWSER_TREES = 60
BROWSER_MAX_DEPTH = 12


def public_gate(record: dict[str, Any]) -> list[str]:
    """Mesmas regras da coleta; só a origem pública declarada é aceita no lugar de 'camera'."""
    if record.get("data_origin") == PUBLIC_ORIGIN and record.get("capture_method") == PUBLIC_CAPTURE_METHOD:
        record = {**record, "data_origin": "camera", "capture_method": "movenet_singlepose_lightning"}
    return training_sample_reasons(record)


def load(patterns: list[str], gate) -> tuple[list[dict], int]:
    paths = sorted({path for pattern in patterns for path in (glob.glob(pattern) or [pattern])})
    records, problems = read_jsonl_files(paths)
    usable = [record for record in records if not gate(record)]
    return usable, len(records) - len(usable) + len(problems)


def browser_forest(records: list[dict], seed: int):
    from sklearn.ensemble import RandomForestClassifier

    features, labels = to_matrix(records)
    model = RandomForestClassifier(n_estimators=BROWSER_TREES, max_depth=BROWSER_MAX_DEPTH, min_samples_leaf=2,
                                   class_weight="balanced", random_state=seed, n_jobs=-1)
    return model.fit(features, labels)


def export_forest(model, summary: dict[str, Any]) -> dict[str, Any]:
    """Árvores em arrays compactos: folha quando left == -1; value = probabilidades por classe."""
    trees = []
    for estimator in model.estimators_:
        tree = estimator.tree_
        values = tree.value[:, 0, :]
        values = values / values.sum(axis=1, keepdims=True)
        trees.append({
            "feature": tree.feature.tolist(),
            "threshold": [float(value) for value in tree.threshold],
            "left": tree.children_left.tolist(),
            "right": tree.children_right.tolist(),
            "value": [[round(float(p), 4) for p in row] for row in values],
        })
    return {
        "format": "vidasegura-random-forest-v1",
        "feature_transform": FEATURE_TRANSFORM_VERSION,
        "feature_count": len(feature_order()),
        "labels": list(POSTURE_LABELS),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "warning": "Modelo experimental de pesquisa acadêmica; não é dispositivo médico nem confirma quedas.",
        "evaluation": summary,
        "trees": trees,
    }


def evaluate(model, records, seed) -> dict[str, Any]:
    import random

    import numpy as np
    from sklearn.metrics import confusion_matrix

    report: dict[str, Any] = {"por_cenario": {}}
    for name, scenario in OCCLUSION_SCENARIOS.items():
        rng = random.Random(seed)
        occluded = [scenario(record, rng) for record in records]
        occluded = [record for record in occluded if not public_gate(record)]
        features, labels = to_matrix(occluded)
        predictions = model.predict(features)
        report["por_cenario"][name] = {"n": len(occluded), "macro_f1": macro_f1(labels, predictions),
                                       "acuracia": float((predictions == labels).mean())}
        if name == "sem_oclusao":
            report["matriz_confusao"] = confusion_matrix(labels, predictions,
                                                         labels=list(range(len(POSTURE_LABELS)))).tolist()
            report["rejeicao"] = rejection_curve(model.predict_proba(features), labels)
            baseline = geometric_baseline(occluded)
            if baseline is not None:
                baseline = np.asarray(baseline)
                report["regra_geometrica"] = {"macro_f1": macro_f1(labels, baseline),
                                              "acuracia": float((baseline == labels).mean()),
                                              "inconclusivas": int((baseline == -1).sum())}
    return report


def run(args) -> dict[str, Any]:
    from sklearn.model_selection import StratifiedGroupKFold

    warnings.filterwarnings("ignore", message=".*Unknown solver options.*")
    public, public_rejected = load(args.publico, public_gate)
    camera, camera_rejected = load(args.camera, training_sample_reasons) if args.camera else ([], 0)
    if not public:
        raise SystemExit("Nenhum registro público utilizável.")

    groups = [record["subject_id"] for record in public]
    features_all, labels_all = to_matrix(public)
    folds = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    cv: dict[str, list[float]] = {name: [] for name in build_models(args.seed)}
    for fold, (train_index, test_index) in enumerate(folds.split(features_all, labels_all, groups)):
        assert not {groups[i] for i in train_index} & {groups[i] for i in test_index}
        train = augment_records([public[i] for i in train_index], args.copias_oclusao, args.seed + fold, gate=public_gate)
        x_train, y_train = to_matrix(train)
        for name, model in build_models(args.seed).items():
            model.fit(x_train, y_train)
            cv[name].append(macro_f1(labels_all[test_index], model.predict(features_all[test_index])))

    report: dict[str, Any] = {
        "aviso": "Rótulos públicos derivados de atividade + geometria do MPII; teste externo com poucos "
                 "participantes é apenas indicativo. Não é dispositivo médico.",
        "dados": {"publicos": len(public), "publicos_recusados": public_rejected,
                  "camera": len(camera), "camera_recusados": camera_rejected,
                  "camera_participantes": len({r["subject_id"] for r in camera}),
                  "classes_publicas": dict(Counter(r["posture_label"] for r in public)),
                  "classes_camera": dict(Counter(r["posture_label"] for r in camera))},
        "validacao_cruzada_publica_macro_f1": {name: {"media": float(sum(v) / len(v)), "folds": v}
                                               for name, v in cv.items()},
    }

    augmented_public = augment_records(public, args.copias_oclusao, args.seed, gate=public_gate)
    if camera:
        x_train, y_train = to_matrix(augmented_public)
        report["teste_externo_camera"] = {}
        for name, model in build_models(args.seed).items():
            model.fit(x_train, y_train)
            report["teste_externo_camera"][name] = evaluate(model, camera, args.seed)
        compact = browser_forest(augmented_public, args.seed)
        report["teste_externo_camera"]["floresta_navegador"] = evaluate(compact, camera, args.seed)

    final_records = augmented_public + augment_records(camera, args.copias_oclusao, args.seed, gate=public_gate)
    final = browser_forest(final_records, args.seed)
    external = report.get("teste_externo_camera", {}).get("floresta_navegador", {})
    summary = {
        "public_samples": len(public), "camera_samples": len(camera),
        "camera_participants": report["dados"]["camera_participantes"],
        "external_camera_macro_f1": external.get("por_cenario", {}).get("sem_oclusao", {}).get("macro_f1"),
        "note": "teste externo: modelo treinado só com dados públicos, avaliado nas amostras da câmera",
    }
    return {"report": report, "export": export_forest(final, summary)}


def markdown(report: dict[str, Any]) -> str:
    lines = ["# Classificador de postura: dados públicos + teste na câmera", "", f"> {report['aviso']}", "",
             "## Dados", "", "```", json.dumps(report["dados"], ensure_ascii=False, indent=2), "```", "",
             "## Validação cruzada por grupo (somente dados públicos)", ""]
    for name, stats in report["validacao_cruzada_publica_macro_f1"].items():
        lines.append(f"- {name}: macro F1 médio {stats['media']:.3f}")
    external = report.get("teste_externo_camera")
    if external:
        lines += ["", "## Teste externo nas amostras da câmera (nunca vistas no treino)", "",
                  "| Modelo | " + " | ".join(OCCLUSION_SCENARIOS) + " |",
                  "|---|" + "---|" * len(OCCLUSION_SCENARIOS)]
        for name, result in external.items():
            cells = [f"{result['por_cenario'][s]['macro_f1']:.3f}" for s in OCCLUSION_SCENARIOS]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        geometric = next(iter(external.values())).get("regra_geometrica")
        if geometric:
            lines += ["", f"Regra geométrica atual nas mesmas amostras (sem oclusão): macro F1 "
                          f"{geometric['macro_f1']:.3f}, acurácia {geometric['acuracia']:.3f}, "
                          f"{geometric['inconclusivas']} inconclusivas (contam como erro)."]
        lines += ["", f"Classes: {', '.join(POSTURE_LABELS)}. Matriz de confusão da floresta do navegador:",
                  "", "```", json.dumps(external["floresta_navegador"]["matriz_confusao"]), "```"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--publico", nargs="+", required=True, help="JSONL gerado por import_mpii.py")
    parser.add_argument("--camera", nargs="*", default=[], help="JSONL da ferramenta de coleta")
    parser.add_argument("--output-dir", type=Path, default=ML_ROOT / "artifacts")
    parser.add_argument("--exportar-navegador", type=Path, help="Grava o modelo JSON para js/posture-model.js")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copias-oclusao", type=int, default=2)
    args = parser.parse_args(argv)

    result = run(args)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / f"public_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "relatorio.json").write_text(json.dumps(result["report"], ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "relatorio.md").write_text(markdown(result["report"]), encoding="utf-8")
    if args.exportar_navegador:
        args.exportar_navegador.parent.mkdir(parents=True, exist_ok=True)
        args.exportar_navegador.write_text(json.dumps(result["export"], separators=(",", ":")), encoding="utf-8")
        print(f"Modelo do navegador: {args.exportar_navegador} "
              f"({args.exportar_navegador.stat().st_size / 1024:.0f} KB)")
    print(markdown(result["report"]))
    print(f"Relatório: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
