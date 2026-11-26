// Classificador de postura aprendido (Random Forest exportado por ml/scripts/train_public.py).
// Roda ao lado da regra geométrica de posture.js: quando o modelo está confiante, a resposta dele
// é usada; caso contrário vale a regra geométrica. Experimental, sem validação clínica.

export const POSTURE_MODEL_CONFIG = Object.freeze({
    url: "./models/posture-forest.json",
    minProbability: 0.6,
    visibilityThreshold: 0.25
});

// Mesma ordem e mesmas contas de ml/scripts/common.py (pose-features-v1).
const KEYPOINT_NAMES = [
    "left_shoulder", "right_shoulder", "left_hip", "right_hip",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
];
const GEOMETRY_PAIRS = [
    ["left_shoulder", "right_shoulder"], ["left_hip", "right_hip"],
    ["left_shoulder", "left_hip"], ["right_shoulder", "right_hip"],
    ["left_hip", "left_knee"], ["right_hip", "right_knee"],
    ["left_knee", "left_ankle"], ["right_knee", "right_ankle"],
    ["left_shoulder", "left_elbow"], ["right_shoulder", "right_elbow"],
    ["left_elbow", "left_wrist"], ["right_elbow", "right_wrist"],
    ["left_shoulder", "right_hip"], ["right_shoulder", "left_hip"]
];
export const FEATURE_COUNT = KEYPOINT_NAMES.length * 4 + GEOMETRY_PAIRS.length * 5;
const LABELS_PT = { standing: "EM PÉ", sitting: "SENTADA", lying: "DEITADA", other: "OUTRA" };

/**
 * @param {Array<{name:string,x:number,y:number,score:number}>} keypoints coordenadas em pixels (MoveNet)
 * @param {number} width largura do vídeo
 * @param {number} height altura do vídeo
 */
export function extractPoseFeatures(keypoints, width, height) {
    const observed = new Map();
    const scores = new Map();
    for (const name of KEYPOINT_NAMES) {
        const point = keypoints?.find(item => item.name === name);
        const score = Number.isFinite(point?.score) ? Math.min(1, Math.max(0, point.score)) : 0;
        scores.set(name, score);
        if (point && Number.isFinite(point.x) && Number.isFinite(point.y) &&
            score >= POSTURE_MODEL_CONFIG.visibilityThreshold) {
            observed.set(name, { x: point.x / width, y: point.y / height, score });
        }
    }

    let center = ["left_hip", "right_hip"].filter(name => observed.has(name));
    if (!center.length) center = ["left_shoulder", "right_shoulder"].filter(name => observed.has(name));
    if (!center.length) center = [...observed.keys()];
    let centerX = 0.5, centerY = 0.5, scale = 1;
    if (center.length) {
        centerX = center.reduce((sum, name) => sum + observed.get(name).x, 0) / center.length;
        centerY = center.reduce((sum, name) => sum + observed.get(name).y, 0) / center.length;
        const xs = [...observed.values()].map(point => point.x);
        const ys = [...observed.values()].map(point => point.y);
        scale = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys), 1e-6);
    }

    const vector = [];
    for (const name of KEYPOINT_NAMES) {
        const point = observed.get(name);
        if (!point) vector.push(0, 0, scores.get(name), 0);
        else vector.push((point.x - centerX) / scale, (point.y - centerY) / scale, point.score, 1);
    }
    for (const [firstName, secondName] of GEOMETRY_PAIRS) {
        const first = observed.get(firstName);
        const second = observed.get(secondName);
        if (!first || !second) {
            vector.push(0, 0, 0, 0, 0);
            continue;
        }
        const dx = second.x - first.x;
        const dy = second.y - first.y;
        const angle = Math.atan2(Math.abs(dx), Math.abs(dy)) / (Math.PI / 2);
        vector.push(dx / scale, dy / scale, Math.hypot(dx, dy) / scale, angle, 1);
    }
    return vector;
}

export function predictForest(model, features) {
    const totals = new Array(model.labels.length).fill(0);
    for (const tree of model.trees) {
        let node = 0;
        while (tree.left[node] !== -1) {
            // scikit-learn compara com float32: x <= limiar segue para a esquerda.
            node = Math.fround(features[tree.feature[node]]) <= tree.threshold[node]
                ? tree.left[node]
                : tree.right[node];
        }
        tree.value[node].forEach((probability, index) => { totals[index] += probability; });
    }
    const probabilities = totals.map(total => total / model.trees.length);
    let best = 0;
    probabilities.forEach((probability, index) => { if (probability > probabilities[best]) best = index; });
    return { label: model.labels[best], probability: probabilities[best], probabilities };
}

export function validateModel(model) {
    return Boolean(model && model.format === "vidasegura-random-forest-v1" &&
        model.feature_count === FEATURE_COUNT && Array.isArray(model.trees) && model.trees.length &&
        Array.isArray(model.labels));
}

/** Combina a regra geométrica com o modelo; o modelo só decide quando está confiante. */
export function fusePosture(geometric, prediction, minProbability = POSTURE_MODEL_CONFIG.minProbability) {
    if (prediction && prediction.probability >= minProbability) {
        const state = LABELS_PT[prediction.label] ?? null;
        if (state) return { state, method: geometric === state ? "modelo+regra" : "modelo" };
    }
    if (geometric) return { state: geometric, method: "regra" };
    return { state: null, method: prediction ? "modelo_incerto" : "regra" };
}

export function installPostureModel(app) {
    app.postureModel = null;
    app.postureModelStatus = "carregando";
    app.postureModelReady = fetch(POSTURE_MODEL_CONFIG.url, { cache: "no-cache" })
        .then(response => response.ok ? response.json() : null)
        .then(model => {
            if (!validateModel(model)) {
                app.postureModelStatus = "indisponível";
                return;
            }
            app.postureModel = model;
            app.postureModelStatus = "ativo";
        })
        .catch(() => { app.postureModelStatus = "indisponível"; });

    app.predictPostureModel = function(pose, width, height) {
        if (!app.postureModel || !pose?.keypoints?.length || !width || !height) return null;
        return predictForest(app.postureModel, extractPoseFeatures(pose.keypoints, width, height));
    };
}
