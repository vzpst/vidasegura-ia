"""Treina e avalia o classificador de postura Random Forest (com baselines).

Fluxo:
  1. Lê os JSONL da ferramenta de coleta e aplica as mesmas regras de validação
     de prepare_dataset.py (common.training_sample_reasons).
  2. Recusa o treino se a diversidade mínima não for atingida (seção 7.7).
  3. Validação cruzada com StratifiedGroupKFold: a mesma pessoa/sessão nunca
     aparece no treino e no teste do mesmo fold.
  4. Em cada fold, o TREINO recebe cópias com oclusão simulada (occlusion.py)
     para o modelo aprender a lidar com o corpo parcialmente visível.
  5. O TESTE é avaliado sem oclusão e em cada cenário de oclusão.
  6. Compara: regra geométrica atual (posture.js), regressão logística e
     Random Forest. Usa macro F1, nunca só acurácia.
  7. Rejeição: se a maior probabilidade ficar abaixo do limiar, a resposta é
     "inconclusive" em vez de um chute. O relatório mostra cobertura × acerto.

Exemplo (com o ambiente ml\\.venv ativo):
    python ml\\scripts\\train_forest.py ml\\data\\raw\\*.jsonl
Ensaio com dados sintéticos (NÃO são resultados):
    python ml\\scripts\\train_forest.py --demo-sintetico
"""

from __future__ import annotations

import argparse
import glob
import warnings
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import (
    FEATURE_TRANSFORM_VERSION,
    POSTURE_LABELS,
    VIEWPOINT_LABELS,
    connected_group_ids,
    dataset_fingerprint,
    diversity_reasons,
    extract_features,
    feature_order,
    read_jsonl_files,
    training_sample_reasons,
)
from occlusion import OCCLUSION_SCENARIOS, augment_records

ML_ROOT = Path(__file__).resolve().parents[1]
REJECTION_THRESHOLDS = (0.0, 0.4, 0.5, 0.6, 0.7, 0.8)
GEOMETRIC_TO_LABEL = {"standing": "standing", "sitting": "sitting", "lying": "lying"}


def build_models(seed: int) -> dict[str, Any]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "regressao_logistica": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
    }


def to_matrix(records: list[dict[str, Any]]):
    import numpy as np

    features = np.asarray([extract_features(record) for record in records], dtype=np.float32)
    labels = np.asarray([POSTURE_LABELS.index(record["posture_label"]) for record in records], dtype=np.int64)
    return features, labels


def macro_f1(y_true, y_pred) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, labels=list(range(len(POSTURE_LABELS))), average="macro", zero_division=0))


def rejection_curve(probabilities, y_true) -> list[dict[str, float]]:
    import numpy as np

    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    rows = []
    for threshold in REJECTION_THRESHOLDS:
        accepted = confidence >= threshold
        coverage = float(accepted.mean()) if len(accepted) else 0.0
        accuracy = float((predictions[accepted] == y_true[accepted]).mean()) if accepted.any() else None
        rows.append({"limiar": threshold, "cobertura": coverage, "acerto_nas_aceitas": accuracy})
    return rows


def geometric_baseline(records: list[dict[str, Any]]) -> list[int] | None:
    """Predição da regra geométrica salva pela coleta; inconclusiva conta como erro (-1)."""
    predictions = [record.get("geometric_prediction") for record in records]
    if all(prediction is None for prediction in predictions):
        return None
    return [POSTURE_LABELS.index(GEOMETRIC_TO_LABEL[p]) if p in GEOMETRIC_TO_LABEL else -1 for p in predictions]


def summarize(values: list[float]) -> dict[str, float | None]:
    import numpy as np

    clean = [value for value in values if value is not None]
    if not clean:
        return {"media": None, "desvio": None}
    return {"media": float(np.mean(clean)), "desvio": float(np.std(clean))}


def load_records(args) -> tuple[list[dict[str, Any]], bool, list[str]]:
    if args.demo_sintetico:
        from synthetic_demo import make_dataset

        return make_dataset(seed=args.seed), True, []
    paths = [path for pattern in args.inputs for path in (glob.glob(pattern) or [pattern])]
    records, read_issues = read_jsonl_files(paths)
    problems = [f"{issue['file']}:{issue['line']}: {issue['reason']}" for issue in read_issues]
    return records, False, problems


def make_gate(synthetic: bool):
    # O modo demo usa origem própria; só a regra de origem é relaxada.
    if not synthetic:
        return training_sample_reasons
    return lambda record: training_sample_reasons({**record, "data_origin": "camera"})


def usable_records(records, synthetic: bool):
    usable, rejected = [], []
    gate = make_gate(synthetic)
    for record in records:
        reasons = gate(record)
        (rejected if reasons else usable).append((record, reasons))
    return [record for record, _ in usable], rejected


def run(args) -> dict[str, Any]:
    import numpy as np
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import StratifiedGroupKFold

    records, synthetic, read_problems = load_records(args)
    usable, rejected = usable_records(records, synthetic)
    if read_problems or (rejected and not args.allow_invalid_records):
        print("Treino interrompido: há linhas inválidas ou rejeitadas.")
        for problem in read_problems[:20]:
            print(f"- {problem}")
        for record, reasons in rejected[:20]:
            print(f"- {record.get('sample_id', '?') if isinstance(record, dict) else '?'}: {'; '.join(reasons)}")
        print("Revise os arquivos ou use --allow-invalid-records para treinar só com as linhas válidas.")
        raise SystemExit(1)

    reasons = diversity_reasons(usable)
    if reasons:
        print("Treino recusado: diversidade insuficiente. Nenhum modelo foi criado.")
        for reason in reasons:
            print(f"- {reason}")
        raise SystemExit(2)

    groups = np.asarray(connected_group_ids(usable))
    _, labels = to_matrix(usable)
    n_splits = min(args.folds, len(set(groups)))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)

    model_names = list(build_models(args.seed))
    scores: dict[str, dict[str, list[float]]] = {name: defaultdict(list) for name in model_names}
    geometric_scores: list[float] = []
    rejection: dict[str, list[list[dict]]] = defaultdict(list)
    viewpoint_scores: dict[str, list[float]] = defaultdict(list)
    total_confusion = np.zeros((len(POSTURE_LABELS), len(POSTURE_LABELS)), dtype=int)
    folds_info = []

    for fold, (train_index, test_index) in enumerate(splitter.split(np.zeros(len(usable)), labels, groups), start=1):
        train_records = [usable[i] for i in train_index]
        test_records = [usable[i] for i in test_index]
        if set(groups[train_index]) & set(groups[test_index]):
            raise AssertionError("vazamento: grupo presente no treino e no teste")
        if not args.sem_oclusao_no_treino:
            train_records = augment_records(train_records, copies_per_record=args.copias_oclusao,
                                            seed=args.seed + fold, gate=make_gate(synthetic))
        x_train, y_train = to_matrix(train_records)
        folds_info.append({"fold": fold, "treino_com_aumento": len(train_records), "teste": len(test_records),
                           "grupos_teste": sorted(set(groups[test_index].tolist()))})

        rng = random.Random(args.seed * 100 + fold)
        test_sets = {name: [scenario(record, rng) for record in test_records]
                     for name, scenario in OCCLUSION_SCENARIOS.items()}
        test_matrices = {name: to_matrix(rows) for name, rows in test_sets.items()}

        geometric = geometric_baseline(test_records)
        if geometric is not None:
            geometric_scores.append(macro_f1(test_matrices["sem_oclusao"][1], geometric))

        for name, model in build_models(args.seed).items():
            model.fit(x_train, y_train)
            for scenario, (x_test, y_test) in test_matrices.items():
                scores[name][scenario].append(macro_f1(y_test, model.predict(x_test)))
            if name == "random_forest":
                x_clean, y_clean = test_matrices["sem_oclusao"]
                prediction = model.predict(x_clean)
                total_confusion += confusion_matrix(y_clean, prediction, labels=list(range(len(POSTURE_LABELS))))
                for scenario in ("sem_oclusao", "lado_direito_oculto", "aleatorio_50"):
                    x_test, y_test = test_matrices[scenario]
                    rejection[scenario].append(rejection_curve(model.predict_proba(x_test), y_test))
                for viewpoint in VIEWPOINT_LABELS:
                    mask = np.asarray([record["viewpoint_label"] == viewpoint for record in test_records])
                    if mask.any():
                        viewpoint_scores[viewpoint].append(macro_f1(y_clean[mask], prediction[mask]))
        print(f"Fold {fold}/{n_splits}: RF macro F1 = {scores['random_forest']['sem_oclusao'][-1]:.3f} "
              f"| com 50% oculto = {scores['random_forest']['aleatorio_50'][-1]:.3f}")

    # Modelo final com todos os dados (e oclusão simulada) para uso posterior.
    final_records = usable if args.sem_oclusao_no_treino else augment_records(usable, args.copias_oclusao, args.seed, gate=make_gate(synthetic))
    x_all, y_all = to_matrix(final_records)
    final_model = build_models(args.seed)["random_forest"].fit(x_all, y_all)
    names = feature_order()
    importances = sorted(zip(names, final_model.feature_importances_.tolist()), key=lambda item: -item[1])[:15]

    def average_curve(curves: list[list[dict]]) -> list[dict]:
        out = []
        for position, threshold in enumerate(REJECTION_THRESHOLDS):
            out.append({
                "limiar": threshold,
                "cobertura": summarize([curve[position]["cobertura"] for curve in curves])["media"],
                "acerto_nas_aceitas": summarize([curve[position]["acerto_nas_aceitas"] for curve in curves])["media"],
            })
        return out

    report = {
        "aviso": ("DADOS SINTÉTICOS: ensaio do pipeline. NÃO usar estas métricas na apresentação."
                  if synthetic else
                  "Resultado experimental com poucos participantes; não prova generalização nem validade clínica."),
        "dados_sinteticos": synthetic,
        "criado_em_utc": datetime.now(timezone.utc).isoformat(),
        "transformacao": FEATURE_TRANSFORM_VERSION,
        "amostras_validas": len(usable),
        "amostras_rejeitadas": len(rejected),
        "participantes": len({record["subject_id"] for record in usable}),
        "sessoes": len({record["session_id"] for record in usable}),
        "grupos_independentes": len(set(groups.tolist())),
        "contagem_por_classe": dict(Counter(record["posture_label"] for record in usable)),
        "contagem_por_ponto_de_vista": dict(Counter(record["viewpoint_label"] for record in usable)),
        "dataset_fingerprint": dataset_fingerprint(usable),
        "folds": folds_info,
        "oclusao_no_treino": not args.sem_oclusao_no_treino,
        "macro_f1": {name: {scenario: summarize(values) for scenario, values in by_scenario.items()}
                     for name, by_scenario in scores.items()},
        "macro_f1_regra_geometrica_sem_oclusao": summarize(geometric_scores) if geometric_scores else
            "indisponível: os registros não trazem geometric_prediction",
        "random_forest_macro_f1_por_ponto_de_vista": {k: summarize(v) for k, v in viewpoint_scores.items()},
        "random_forest_matriz_confusao_sem_oclusao": {
            "ordem": list(POSTURE_LABELS),
            "linhas_reais_colunas_preditas": total_confusion.tolist(),
        },
        "random_forest_rejeicao": {scenario: average_curve(curves) for scenario, curves in rejection.items()},
        "random_forest_importancia_top15": [{"caracteristica": n, "importancia": v} for n, v in importances],
    }
    return {"report": report, "model": final_model, "synthetic": synthetic}


def write_outputs(result: dict[str, Any], output_dir: Path) -> Path:
    import joblib

    report = result["report"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_dir / f"forest_{'demo_' if result['synthetic'] else ''}{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "relatorio.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (run_dir / "relatorio.md").write_text(markdown_report(report), encoding="utf-8")
    joblib.dump({"model": result["model"], "classes": list(POSTURE_LABELS),
                 "feature_transform": FEATURE_TRANSFORM_VERSION, "synthetic": result["synthetic"]},
                run_dir / "random_forest.joblib")
    return run_dir


def _fmt(stat) -> str:
    if not isinstance(stat, dict) or stat.get("media") is None:
        return "—"
    return f"{stat['media']:.3f} ± {stat['desvio']:.3f}"


def markdown_report(report: dict[str, Any]) -> str:
    lines = [f"# Relatório do classificador de postura\n", f"> {report['aviso']}\n",
             f"- Amostras válidas: {report['amostras_validas']} · participantes: {report['participantes']} · "
             f"grupos independentes: {report['grupos_independentes']}",
             f"- Oclusão simulada no treino: {'sim' if report['oclusao_no_treino'] else 'não'}\n",
             "## Macro F1 (média ± desvio entre folds)\n",
             "| Cenário de teste | Regressão logística | Random Forest |", "|---|---|---|"]
    for scenario in report["macro_f1"]["random_forest"]:
        lines.append(f"| {scenario} | {_fmt(report['macro_f1']['regressao_logistica'][scenario])} | "
                     f"{_fmt(report['macro_f1']['random_forest'][scenario])} |")
    geometric = report["macro_f1_regra_geometrica_sem_oclusao"]
    lines += ["", f"Regra geométrica atual (sem oclusão): {_fmt(geometric) if isinstance(geometric, dict) else geometric}", "",
              "## Random Forest por ponto de vista\n", "| Ponto de vista | Macro F1 |", "|---|---|"]
    lines += [f"| {k} | {_fmt(v)} |" for k, v in report["random_forest_macro_f1_por_ponto_de_vista"].items()]
    matrix = report["random_forest_matriz_confusao_sem_oclusao"]
    lines += ["", "## Matriz de confusão (Random Forest, sem oclusão, soma dos folds)\n",
              "| real \\ predito | " + " | ".join(matrix["ordem"]) + " |", "|---" * (len(matrix["ordem"]) + 1) + "|"]
    for label, row in zip(matrix["ordem"], matrix["linhas_reais_colunas_preditas"]):
        lines.append(f"| {label} | " + " | ".join(str(v) for v in row) + " |")
    lines += ["", "## Rejeição (responder \"inconclusivo\" abaixo do limiar)\n"]
    for scenario, curve in report["random_forest_rejeicao"].items():
        lines += [f"**{scenario}**\n", "| Limiar | Cobertura | Acerto nas aceitas |", "|---|---|---|"]
        for row in curve:
            accuracy = "—" if row["acerto_nas_aceitas"] is None else f"{row['acerto_nas_aceitas']:.3f}"
            lines.append(f"| {row['limiar']:.1f} | {row['cobertura']:.2f} | {accuracy} |")
        lines.append("")
    lines += ["## Características mais importantes\n"]
    lines += [f"{i}. `{item['caracteristica']}`: {item['importancia']:.4f}"
              for i, item in enumerate(report["random_forest_importancia_top15"], 1)]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="*", help="Arquivos JSONL da coleta (aceita curingas)")
    parser.add_argument("--demo-sintetico", action="store_true",
                        help="Usa bonecos sintéticos só para testar o pipeline (métricas sem valor real)")
    parser.add_argument("--output-dir", type=Path, default=ML_ROOT / "artifacts",
                        help="Diretório local ignorado pelo Git")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copias-oclusao", type=int, default=2,
                        help="Cópias com oclusão simulada por amostra de treino")
    parser.add_argument("--sem-oclusao-no-treino", action="store_true",
                        help="Desliga o aumento por oclusão (para comparar o efeito)")
    parser.add_argument("--allow-invalid-records", action="store_true")
    args = parser.parse_args(argv)
    if not args.inputs and not args.demo_sintetico:
        parser.error("informe arquivos JSONL ou use --demo-sintetico")

    try:
        import numpy  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError:
        print("Ative o ambiente ml\\.venv e instale ml\\requirements.txt.")
        return 2

    # Aviso conhecido e inofensivo da combinação SciPy/scikit-learn instalada.
    warnings.filterwarnings("ignore", message="Unknown solver options")
    result = run(args)
    run_dir = write_outputs(result, args.output_dir)
    rf = result["report"]["macro_f1"]["random_forest"]
    print(f"\n{result['report']['aviso']}")
    print(f"Random Forest macro F1 sem oclusão: {_fmt(rf['sem_oclusao'])} | 50% oculto: {_fmt(rf['aleatorio_50'])}")
    print(f"Relatório e modelo salvos em: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
