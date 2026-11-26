import test from "node:test";
import assert from "node:assert/strict";

const { extractPoseFeatures, predictForest, fusePosture, validateModel, FEATURE_COUNT } =
    await import(new URL("../js/posture-model.js", import.meta.url));

const standing = [
    ["left_shoulder", 340, 100], ["right_shoulder", 300, 100], ["left_hip", 335, 220], ["right_hip", 305, 220],
    ["left_elbow", 345, 160], ["right_elbow", 295, 160], ["left_wrist", 345, 210], ["right_wrist", 295, 210],
    ["left_knee", 335, 320], ["right_knee", 305, 320], ["left_ankle", 335, 420], ["right_ankle", 305, 420]
].map(([name, x, y]) => ({ name, x, y, score: 0.9 }));

// Árvore mínima: feature 0 <= 0 → "standing", senão "lying".
const tinyModel = {
    format: "vidasegura-random-forest-v1",
    feature_count: FEATURE_COUNT,
    labels: ["standing", "sitting", "lying", "other"],
    trees: [{
        feature: [0, -2, -2],
        threshold: [0, -2, -2],
        left: [1, -1, -1],
        right: [2, -1, -1],
        value: [[0.5, 0, 0.5, 0], [0.9, 0.1, 0, 0], [0, 0, 1, 0]]
    }]
};

test("vetor de features tem 118 valores e máscara zero para ponto ausente", () => {
    const features = extractPoseFeatures(standing.filter(point => point.name !== "left_knee"), 640, 480);
    assert.equal(features.length, FEATURE_COUNT);
    assert.equal(FEATURE_COUNT, 118);
    const knee = 8 * 4;
    assert.deepEqual(features.slice(knee, knee + 4), [0, 0, 0, 0]);
    assert.ok(features.every(Number.isFinite));
});

test("ponto com confiança baixa não entra como observado", () => {
    const lowered = standing.map(point => point.name === "right_ankle" ? { ...point, score: 0.1 } : point);
    const features = extractPoseFeatures(lowered, 640, 480);
    const ankle = 11 * 4;
    assert.deepEqual(features.slice(ankle, ankle + 4), [0, 0, 0.1, 0]);
});

test("floresta percorre a árvore como o scikit-learn (x <= limiar vai à esquerda)", () => {
    const left = predictForest(tinyModel, [-0.2, ...new Array(FEATURE_COUNT - 1).fill(0)]);
    assert.equal(left.label, "standing");
    assert.equal(left.probability, 0.9);
    const right = predictForest(tinyModel, [0.3, ...new Array(FEATURE_COUNT - 1).fill(0)]);
    assert.equal(right.label, "lying");
});

test("modelo inválido é recusado", () => {
    assert.equal(validateModel(tinyModel), true);
    assert.equal(validateModel({ ...tinyModel, feature_count: 10 }), false);
    assert.equal(validateModel(null), false);
});

test("fusão: modelo confiante decide; incerto devolve a regra geométrica", () => {
    assert.deepEqual(fusePosture(null, { label: "lying", probability: 0.8 }), { state: "DEITADA", method: "modelo" });
    assert.deepEqual(fusePosture("DEITADA", { label: "lying", probability: 0.8 }),
        { state: "DEITADA", method: "modelo+regra" });
    assert.deepEqual(fusePosture("SENTADA", { label: "lying", probability: 0.4 }), { state: "SENTADA", method: "regra" });
    assert.deepEqual(fusePosture(null, { label: "other", probability: 0.7 }), { state: "OUTRA", method: "modelo" });
    assert.deepEqual(fusePosture(null, null), { state: null, method: "regra" });
});
