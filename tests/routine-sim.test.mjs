import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

// Executa o dia simulado inteiro, sem navegador, pelo mesmo caminho da câmera:
// postura (regra + Random Forest) → movimento → queda → gravidade → atividade.
const load = path => import(new URL(path, import.meta.url));
const [{ installPosture }, { installVisibility }, { extractPoseFeatures, predictForest },
    { createMotionAnalyzer }, { createFallDetector }, { createFallSeverityTracker },
    { createActivityClassifier }, sim] = await Promise.all([
    load("../js/posture.js"), load("../js/visibility.js"), load("../js/posture-model.js"),
    load("../js/motion-analysis.js"), load("../js/fall-detection.js"), load("../js/fall-severity.js"),
    load("../js/activity.js"), load("../js/routine-sim.js")
]);
const model = JSON.parse(await readFile(new URL("../models/posture-forest.json", import.meta.url), "utf8"));

function seeded(seed) {
    return () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
}

export function runDay({ step = 33 } = {}) {
    const app = { camera: { videoWidth: 640, videoHeight: 480 } };
    installPosture(app);
    installVisibility(app);
    app.predictPostureModel = (pose, width, height) => predictForest(model, extractPoseFeatures(pose.keypoints, width, height));
    let now = 1_000_000;
    const motion = createMotionAnalyzer({ now: () => now });
    const detector = createFallDetector();
    const severity = createFallSeverityTracker();
    const activity = createActivityClassifier();
    const random = seeded(42);
    const segments = {};
    let currentIndex = -1;

    for (let elapsed = 0; elapsed < sim.TOTAL_MS; elapsed += step) {
        now += step;
        const { segment, index, local } = sim.segmentAt(elapsed);
        if (index !== currentIndex) {
            motion.resetContinuity();
            detector.reset();
            severity.reset();
            activity.reset();
            currentIndex = index;
            segments[segment.id] = { postures: {}, activities: [], maxLevel: 0, states: new Set(), lastActivity: null };
        }
        const info = segments[segment.id];
        const body = sim.buildBody(segment.pose(local), random);
        const pose = { keypoints: body.keypoints };
        const analysisResult = app.createAnalysisResult(pose, now);
        const sample = motion.recordInference({ pose, analysisResult, videoWidth: 640, videoHeight: 480 });
        const fall = detector.analyze(sample);
        const level = severity.update(fall, sample);
        const posture = analysisResult.posture.state === "INCONCLUSIVA" ? null : analysisResult.posture.state;
        const result = activity.update({ timestamp: now, posture, keypoints: body.keypoints, width: 640, height: 480 });
        info.postures[analysisResult.posture.state] = (info.postures[analysisResult.posture.state] ?? 0) + 1;
        if (result.changed) info.activities.push(result.id);
        info.lastActivity = result.id;
        info.maxLevel = Math.max(info.maxLevel, level.level);
        info.states.add(fall.state);
    }
    return segments;
}

const day = runDay();

test("o dia simulado nunca gera alerta de queda (deitar para dormir incluído)", () => {
    for (const [id, info] of Object.entries(day)) {
        assert.equal(info.maxLevel, 0, `${id}: nível ${info.maxLevel}`);
        assert.equal(info.states.has("DESCIDA_SUSPEITA"), false, `${id}: descida suspeita`);
    }
});

test("cada cena termina na atividade esperada", () => {
    assert.equal(day.cafe.lastActivity, "refeicao");
    assert.equal(day.tv.lastActivity, "tv");
    assert.equal(day.leitura.lastActivity, "leitura");
    assert.equal(day.descanso.lastActivity, "descanso");
    assert.equal(day.dormir.lastActivity, "dormindo");
});

test("a pessoa é vista caminhando antes de sentar e antes de deitar", () => {
    assert.ok(day.cafe.activities.includes("caminhando"), JSON.stringify(day.cafe.activities));
    assert.ok(day.dormir.activities.includes("caminhando"), JSON.stringify(day.dormir.activities));
});

test("posturas predominantes: sentada nas cenas da sala, deitada ao dormir", () => {
    const top = info => Object.entries(info.postures).sort((a, b) => b[1] - a[1])[0][0];
    assert.equal(top(day.tv), "SENTADA");
    assert.equal(top(day.leitura), "SENTADA");
    assert.equal(top(day.descanso), "SENTADA");
    assert.equal(top(day.dormir), "DEITADA");
});

test("relógio da simulação avança dentro de cada cena", () => {
    const first = sim.SEGMENTS[0];
    assert.equal(sim.clockAt(first, 0).getHours(), 7);
    assert.equal(sim.clockAt(sim.SEGMENTS.at(-1), sim.SEGMENTS.at(-1).ms).getHours(), 22);
});
