import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const source = await readFile(new URL("../js/posture.js", import.meta.url), "utf8");
const { installPosture, POSTURE_CLASSIFICATION_CONFIG } = await import(
    `data:text/javascript;base64,${Buffer.from(source).toString("base64")}`
);

function classifier() {
    const app = { postureHistory: [] };
    installPosture(app);
    return app;
}

function makePose(points, { lowConfidence = [], confidence = 0.95 } = {}) {
    return {
        keypoints: Object.entries(points).map(([name, [x, y]]) => ({
            name,
            x,
            y,
            score: lowConfidence.includes(name) ? 0.1 : confidence
        }))
    };
}

function standingFront() {
    return makePose({
        left_shoulder: [420, 180], right_shoulder: [580, 180],
        left_hip: [440, 440], right_hip: [560, 440],
        left_knee: [445, 650], right_knee: [555, 650],
        left_ankle: [450, 860], right_ankle: [550, 860]
    });
}

function standingProfile(side = "left") {
    const visible = side === "left" ? "left" : "right";
    const hidden = visible === "left" ? "right" : "left";
    const points = {
        [`${visible}_shoulder`]: [500, 180], [`${visible}_hip`]: [500, 440],
        [`${visible}_knee`]: [520, 650], [`${visible}_ankle`]: [510, 860],
        [`${hidden}_shoulder`]: [504, 182], [`${hidden}_hip`]: [503, 442],
        [`${hidden}_knee`]: [521, 651], [`${hidden}_ankle`]: [511, 861]
    };
    return makePose(points, {
        lowConfidence: Object.keys(points).filter(name => name.startsWith(hidden))
    });
}

function seatedFront() {
    return makePose({
        left_shoulder: [420, 180], right_shoulder: [580, 180],
        left_hip: [440, 440], right_hip: [560, 440],
        left_knee: [340, 500], right_knee: [660, 500],
        left_ankle: [370, 700], right_ankle: [630, 700]
    });
}

function seatedProfile(side = "left") {
    const visible = side === "left" ? "left" : "right";
    const hidden = visible === "left" ? "right" : "left";
    const points = {
        [`${visible}_shoulder`]: [500, 180], [`${visible}_hip`]: [500, 440],
        [`${visible}_knee`]: [680, 470], [`${visible}_ankle`]: [700, 700],
        [`${hidden}_shoulder`]: [504, 182], [`${hidden}_hip`]: [503, 442],
        [`${hidden}_knee`]: [682, 472], [`${hidden}_ankle`]: [702, 702]
    };
    return makePose(points, {
        lowConfidence: Object.keys(points).filter(name => name.startsWith(hidden))
    });
}

function lyingFrontFacingCamera() {
    return makePose({
        left_shoulder: [300, 450], right_shoulder: [300, 550],
        left_hip: [550, 450], right_hip: [550, 550],
        left_knee: [750, 430], right_knee: [750, 570],
        left_ankle: [950, 420], right_ankle: [950, 580]
    });
}

function lyingProfile() {
    const points = {
        left_shoulder: [250, 500], left_hip: [500, 520],
        left_knee: [700, 520], left_ankle: [900, 525],
        right_shoulder: [254, 502], right_hip: [504, 522],
        right_knee: [704, 522], right_ankle: [904, 527]
    };
    return makePose(points, {
        lowConfidence: Object.keys(points).filter(name => name.startsWith("right"))
    });
}

test("limiares geométricos estão centralizados e marcados como provisórios", () => {
    assert.equal(POSTURE_CLASSIFICATION_CONFIG.provisional, true);
    assert.equal(POSTURE_CLASSIFICATION_CONFIG.minKeypointConfidence, 0.35);
});

test("pessoa em pé de frente", () => {
    assert.equal(classifier().classifyPose(standingFront()), "EM PÉ");
});

test("pessoa em pé de perfil esquerdo usa a cadeia visível", () => {
    assert.equal(classifier().classifyPose(standingProfile("left")), "EM PÉ");
});

test("pessoa em pé de perfil direito usa a cadeia visível", () => {
    assert.equal(classifier().classifyPose(standingProfile("right")), "EM PÉ");
});

test("pessoa sentada de frente combina joelho flexionado e posição relativa", () => {
    assert.equal(classifier().classifyPose(seatedFront()), "SENTADA");
});

test("pessoa sentada de perfil esquerdo usa a cadeia visível", () => {
    assert.equal(classifier().classifyPose(seatedProfile("left")), "SENTADA");
});

test("pessoa sentada de perfil direito usa a cadeia visível", () => {
    assert.equal(classifier().classifyPose(seatedProfile("right")), "SENTADA");
});

test("pessoa deitada de frente para a câmera tem tronco e pernas projetados horizontalmente", () => {
    assert.equal(classifier().classifyPose(lyingFrontFacingCamera()), "DEITADA");
});

test("pessoa deitada de lado usa os pontos confiáveis do perfil", () => {
    assert.equal(classifier().classifyPose(lyingProfile()), "DEITADA");
});

test("geometria oblíqua com pernas quase estendidas continua compatível com posição em pé", () => {
    const oblique = makePose({
        left_shoulder: [400, 180], right_shoulder: [570, 195],
        left_hip: [450, 440], right_hip: [600, 455],
        left_knee: [485, 650], right_knee: [630, 665],
        left_ankle: [535, 860], right_ankle: [680, 875]
    });
    assert.equal(classifier().classifyPose(oblique), "EM PÉ");
});

test("um lado do corpo oculto não impede classificação quando o outro é suficiente", () => {
    const pose = standingFront();
    pose.keypoints = pose.keypoints.filter(point => point.name.startsWith("left"));
    assert.equal(classifier().classifyPose(pose), "EM PÉ");
});

test("ombros e quadris parcialmente sobrepostos não são promediados para inventar tronco vertical", () => {
    const pose = makePose({
        left_shoulder: [540, 180], right_shoulder: [459, 180],
        left_hip: [540, 440], right_hip: [621, 440],
        left_knee: [540, 650], right_knee: [621, 650],
        left_ankle: [540, 860], right_ankle: [621, 860]
    });
    assert.equal(classifier().classifyPose(pose), null);
});

test("baixa confiança nos pontos necessários torna a classificação inconclusiva", () => {
    const pose = standingFront();
    pose.keypoints.forEach(point => { point.score = 0.1; });
    assert.equal(classifier().classifyPose(pose), null);
});

test("inclinar o tronco para alcançar um objeto não classifica como deitada", () => {
    const pose = standingFront();
    for (const point of pose.keypoints.filter(item => item.name.includes("shoulder"))) point.x -= 100;
    assert.notEqual(classifier().classifyPose(pose), "DEITADA");
});

test("agachamento com joelhos baixos não é confundido com postura sentada", () => {
    const squat = makePose({
        left_shoulder: [420, 180], right_shoulder: [580, 180],
        left_hip: [440, 440], right_hip: [560, 440],
        left_knee: [240, 690], right_knee: [760, 690],
        left_ankle: [240, 890], right_ankle: [760, 890]
    });
    assert.equal(classifier().classifyPose(squat), null);
});

test("ajoelhamento sem progressão vertical compatível não recebe rótulo de sentada", () => {
    const kneeling = makePose({
        left_shoulder: [420, 180], right_shoulder: [580, 180],
        left_hip: [440, 440], right_hip: [560, 440],
        left_knee: [700, 800], right_knee: [300, 800],
        left_ankle: [560, 780], right_ankle: [440, 780]
    });
    assert.equal(classifier().classifyPose(kneeling), null);
});

test("ausência de pontos suficientes retorna null para o fluxo marcar inconclusiva", () => {
    const pose = makePose({ left_shoulder: [400, 180], left_hip: [420, 440] });
    assert.equal(classifier().classifyPose(pose), null);
});

test("a classificação geométrica precede e preserva a estabilização temporal existente", () => {
    const app = classifier();
    const sequence = [
        [standingFront(), "EM PÉ"], [standingFront(), "EM PÉ"], [standingFront(), "EM PÉ"],
        [standingFront(), "EM PÉ"], [standingFront(), "EM PÉ"],
        [seatedProfile(), "SENTADA"], [seatedProfile(), "SENTADA"], [seatedProfile(), "SENTADA"],
        [seatedProfile(), "SENTADA"], [seatedProfile(), "SENTADA"],
        [lyingProfile(), "DEITADA"], [lyingProfile(), "DEITADA"], [lyingProfile(), "DEITADA"],
        [lyingProfile(), "DEITADA"], [lyingProfile(), "DEITADA"]
    ];
    const raw = sequence.map(([pose]) => app.classifyPose(pose));
    assert.deepEqual(raw, sequence.map(([, posture]) => posture));

    const stable = raw.map(posture => app.updatePostureHistory(posture));
    assert.equal(stable[3], null);
    assert.equal(stable[4], "EM PÉ");
    assert.equal(stable[9], "SENTADA");
    assert.equal(stable.at(-1), "DEITADA");
});
