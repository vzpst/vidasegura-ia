import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const source = await readFile(new URL("../js/motion-analysis.js", import.meta.url), "utf8");
const { createMotionAnalyzer } = await import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);

const jointNames = [
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"
];
const base = {
    left_shoulder: [450, 250], right_shoulder: [550, 250],
    left_elbow: [400, 350], right_elbow: [600, 350],
    left_wrist: [380, 450], right_wrist: [620, 450],
    left_hip: [470, 500], right_hip: [530, 500],
    left_knee: [470, 700], right_knee: [530, 700],
    left_ankle: [470, 900], right_ankle: [530, 900]
};

function makePose({ dx = 0, dy = 0, shoulderDx = 0, omit = [], lowConfidence = [] } = {}) {
    return {
        keypoints: jointNames.filter(name => !omit.includes(name)).map(name => {
            const [x, y] = base[name];
            return {
                name,
                x: x + dx + (name.includes("shoulder") ? shoulderDx : 0),
                y: y + dy,
                score: lowConfidence.includes(name) ? 0.1 : 0.95
            };
        })
    };
}

function makeResult(pose, timestamp, posture = "EM PÉ") {
    const keypoints = jointNames.map(name => {
        const point = pose?.keypoints.find(item => item.name === name);
        return {
            name,
            x: point?.x ?? null,
            y: point?.y ?? null,
            score: point?.score ?? null,
            availability: !point ? "INDISPONÍVEL" : point.score >= 0.25 ? "OBSERVADO" : "BAIXA_CONFIANÇA"
        };
    });
    return {
        timestamp,
        personDetection: { detected: Boolean(pose), state: pose ? "PESSOA DETECTADA" : "PESSOA NÃO DETECTADA" },
        posture: { state: posture },
        visibility: {
            state: pose ? "BOA VISIBILIDADE" : "NÃO AVALIÁVEL",
            regionCounts: { head: 0, torso: pose ? 4 : 0, arms: pose ? 4 : 0, legs: pose ? 4 : 0 }
        },
        fallRisk: { state: "NÃO AVALIADO", evaluated: false },
        keypoints
    };
}

function record(analyzer, { timestamp, pose = makePose(), posture = "EM PÉ", width = 1000, height = 1000 } = {}) {
    return analyzer.recordInference({
        pose,
        analysisResult: makeResult(pose, timestamp, posture),
        videoWidth: width,
        videoHeight: height
    });
}

test("1. pessoa parada em pé produz deslocamento normalizado zero", () => {
    const analyzer = createMotionAnalyzer();
    record(analyzer, { timestamp: 1000 });
    const current = record(analyzer, { timestamp: 1500 });
    assert.equal(current.posture, "EM PÉ");
    assert.equal(current.displacement.deltaXNormalized, 0);
    assert.equal(current.displacement.deltaYNormalized, 0);
    assert.equal(current.displacement.intervalMs, 500);
});

test("2. movimento para baixo tem delta Y positivo", () => {
    const analyzer = createMotionAnalyzer();
    record(analyzer, { timestamp: 1000 });
    const current = record(analyzer, { timestamp: 1500, pose: makePose({ dy: 100 }) });
    assert.ok(current.displacement.deltaYNormalized > 0);
    assert.ok(current.displacement.normalizedYPerSecond > 0);
});

test("3. movimento para cima tem delta Y negativo", () => {
    const analyzer = createMotionAnalyzer();
    record(analyzer, { timestamp: 1000, pose: makePose({ dy: 100 }) });
    const current = record(analyzer, { timestamp: 1500 });
    assert.ok(current.displacement.deltaYNormalized < 0);
});

test("4. inclinação gradual atualiza ângulo do tronco em graus", () => {
    const analyzer = createMotionAnalyzer();
    const first = record(analyzer, { timestamp: 1000 });
    const second = record(analyzer, { timestamp: 1100, pose: makePose({ shoulderDx: 15 }) });
    const third = record(analyzer, { timestamp: 1200, pose: makePose({ shoulderDx: 30 }) });
    assert.equal(first.trunkAngleDegrees, 0);
    assert.ok(Math.abs(second.trunkAngleDegrees) > Math.abs(first.trunkAngleDegrees));
    assert.ok(Math.abs(third.trunkAngleDegrees) > Math.abs(second.trunkAngleDegrees));
});

test("5. registra contexto de transição em pé para sentada, sem classificar por conta própria", () => {
    const analyzer = createMotionAnalyzer();
    const standing = record(analyzer, { timestamp: 1000, posture: "EM PÉ" });
    const seated = record(analyzer, { timestamp: 2000, posture: "SENTADA", pose: makePose({ dy: 30 }) });
    assert.equal(standing.posture, "EM PÉ");
    assert.equal(seated.posture, "SENTADA");
});

test("6. deitada permanece apenas como postura, sem criar evento de queda", () => {
    const analyzer = createMotionAnalyzer();
    const lying = record(analyzer, { timestamp: 1000, posture: "DEITADA" });
    assert.equal(lying.posture, "DEITADA");
    assert.equal(lying.fallRisk, undefined);
    assert.equal(lying.fallEvent, undefined);
});

test("7. articulações necessárias ausentes deixam centros e ângulo indisponíveis", () => {
    const analyzer = createMotionAnalyzer();
    const sample = record(analyzer, {
        timestamp: 1000,
        pose: makePose({ omit: ["right_shoulder", "left_hip"] })
    });
    assert.equal(sample.centers.shoulders, null);
    assert.equal(sample.centers.hips, null);
    assert.equal(sample.centers.body, null);
    assert.equal(sample.trunkAngleDegrees, null);
    assert.equal(sample.joints.find(point => point.name === "right_shoulder").x, null);
    assert.equal(sample.boundingBox.representsWholeBody, false);
});

test("8. pontos abaixo do limiar mantêm coordenadas e score, mas não são observados", () => {
    const analyzer = createMotionAnalyzer();
    const pose = makePose({ lowConfidence: ["left_shoulder", "right_shoulder"] });
    const sample = record(analyzer, { timestamp: 1000, pose });
    const point = sample.joints.find(item => item.name === "left_shoulder");
    assert.equal(point.x, 450);
    assert.equal(point.normalizedX, 0.45);
    assert.equal(point.confidence, 0.1);
    assert.equal(point.observed, false);
    assert.equal(sample.centers.shoulders, null);
    assert.equal(sample.trunkAngleDegrees, null);
});

test("9. ausência de pessoa gera amostra sem geometria e quebra continuidade", () => {
    const analyzer = createMotionAnalyzer({ now: () => 10_000 });
    record(analyzer, { timestamp: 1000 });
    const absent = record(analyzer, { timestamp: 1100, pose: null, posture: "INCONCLUSIVA" });
    const resumed = record(analyzer, { timestamp: 1200, pose: makePose({ dx: 200 }) });
    assert.equal(absent.personDetection.detected, false);
    assert.equal(absent.joints.every(point => point.x === null && point.y === null), true);
    assert.equal(absent.centers.body, null);
    assert.equal(resumed.displacement.deltaXNormalized, null);
    assert.equal(analyzer.getHistory().length, 3);
});

test("10. falha de inferência interrompe a continuidade sem virar ausência nem amostra", () => {
    const analyzer = createMotionAnalyzer({ now: () => 10_000 });
    record(analyzer, { timestamp: 1000 });
    analyzer.interruptContinuity(); // representa exceção/resultado inválido do detector
    const afterFailure = record(analyzer, { timestamp: 1300, pose: makePose({ dy: 80 }) });
    assert.equal(analyzer.getHistory().length, 2);
    assert.equal(afterFailure.personDetection.detected, true);
    assert.equal(afterFailure.displacement.deltaYNormalized, null);
});

test("11. troca ou reinício de câmera zera referência, preservando as amostras", () => {
    const analyzer = createMotionAnalyzer({ now: () => 10_000 });
    record(analyzer, { timestamp: 1000 });
    analyzer.resetContinuity();
    const afterRestart = record(analyzer, { timestamp: 1100, pose: makePose({ dx: 200 }) });
    assert.equal(analyzer.getHistory().length, 2);
    assert.equal(afterRestart.displacement.deltaXNormalized, null);
});

test("12. intervalos irregulares usam tempo real e intervalo inválido não gera velocidade", () => {
    const analyzer = createMotionAnalyzer({ now: () => 10_000 });
    record(analyzer, { timestamp: 1000 });
    const irregular = record(analyzer, { timestamp: 1750, pose: makePose({ dx: 75 }) });
    assert.equal(irregular.displacement.intervalMs, 750);
    assert.ok(Math.abs(irregular.displacement.normalizedXPerSecond - 0.1) < 1e-12);
    const invalidInterval = record(analyzer, { timestamp: 1750, pose: makePose({ dx: 100 }) });
    assert.equal(invalidInterval.displacement.intervalMs, 0);
    assert.equal(invalidInterval.displacement.normalizedXPerSecond, null);
});

test("13. o buffer nunca ultrapassa 300 amostras", () => {
    const analyzer = createMotionAnalyzer({ now: () => 100_000 });
    for (let i = 0; i < 305; i++) record(analyzer, { timestamp: 90_000 + i });
    assert.equal(analyzer.getHistory().length, 300);
});

test("14. retenção remove amostras anteriores a 60 segundos", () => {
    let clock = 10_000;
    const analyzer = createMotionAnalyzer({ now: () => clock });
    record(analyzer, { timestamp: 10_000 });
    clock = 70_002;
    record(analyzer, { timestamp: 70_002 });
    assert.deepEqual(analyzer.getHistory().map(sample => sample.timestamp), [70_002]);
});

test("dimensões usam largura/altura reais e amostras sem dimensão preservam normalizados null", () => {
    const analyzer = createMotionAnalyzer();
    const sample = record(analyzer, { timestamp: 1000, width: 2000, height: 1000 });
    assert.equal(sample.joints.find(point => point.name === "left_shoulder").normalizedX, 0.225);
    const missingDimensions = record(analyzer, { timestamp: 1100, width: 0, height: 0 });
    assert.equal(missingDimensions.centers.body.normalizedX, null);
    assert.equal(missingDimensions.boundingBox.widthNormalized, null);
});

test("a caixa parcial declara explicitamente que não representa o corpo inteiro", () => {
    const analyzer = createMotionAnalyzer();
    const sample = record(analyzer, {
        timestamp: 1000,
        pose: makePose({ omit: ["left_ankle", "right_ankle"] })
    });
    assert.equal(sample.boundingBox.pointsUsed, 10);
    assert.equal(sample.boundingBox.isPartial, true);
    assert.equal(sample.boundingBox.representsWholeBody, false);
});

test("perfil: ombro e quadril do mesmo lado geram centro corporal e ângulo do tronco", () => {
    const analyzer = createMotionAnalyzer();
    const sample = record(analyzer, {
        timestamp: 1000,
        pose: makePose({ omit: ["right_shoulder", "right_hip", "right_elbow", "right_wrist", "right_knee", "right_ankle"] })
    });
    assert.equal(sample.centers.shoulders, null);
    assert.equal(sample.centers.hips, null);
    assert.equal(sample.centers.body.source, "left");
    assert.equal(sample.centers.body.normalizedY, (0.25 + 0.5) / 2);
    assert.ok(Number.isFinite(sample.trunkAngleDegrees));
});

test("troca da origem do centro (bilateral para um lado) não gera deslocamento artificial", () => {
    const analyzer = createMotionAnalyzer();
    record(analyzer, { timestamp: 1000 });
    const profile = record(analyzer, {
        timestamp: 1100,
        pose: makePose({ omit: ["right_shoulder", "right_hip"] })
    });
    assert.equal(profile.centers.body.source, "left");
    assert.equal(profile.displacement.deltaYNormalized, null);
    const next = record(analyzer, {
        timestamp: 1200,
        pose: makePose({ dy: 50, omit: ["right_shoulder", "right_hip"] })
    });
    assert.ok(next.displacement.deltaYNormalized > 0);
});
