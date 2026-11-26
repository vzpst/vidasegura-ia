import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

async function importLocalModule(relativePath) {
    const source = await readFile(new URL(relativePath, import.meta.url), "utf8");
    return import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);
}

const [{ createMotionAnalyzer }, { createFallDetector, FALL_DETECTION_CONFIG, FALL_DETECTION_STATES, installFallDetection }] = await Promise.all([
    importLocalModule("../js/motion-analysis.js"),
    importLocalModule("../js/fall-detection.js")
]);

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

function makePose({ dy = 0, shoulderDx = 0, lowConfidence = false } = {}) {
    return {
        keypoints: jointNames.map(name => {
            const [x, y] = base[name];
            return {
                name,
                x: x + (name.includes("shoulder") ? shoulderDx : 0),
                y: y + dy,
                score: lowConfidence ? 0.1 : 0.95
            };
        })
    };
}

function makeAnalysisResult(pose, timestamp, posture) {
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
    const allBodyPointsObserved = Boolean(pose) && pose.keypoints.every(point => point.score >= 0.25);
    return {
        timestamp,
        personDetection: { detected: Boolean(pose) },
        posture: { state: posture },
        visibility: {
            state: !pose ? "NÃO AVALIÁVEL" : allBodyPointsObserved ? "BOA VISIBILIDADE" : "VISIBILIDADE INSUFICIENTE",
            regionCounts: { head: 0, torso: allBodyPointsObserved ? 4 : 0,
                arms: allBodyPointsObserved ? 4 : 0, legs: allBodyPointsObserved ? 4 : 0 }
        },
        keypoints
    };
}

function createFixture() {
    const motion = createMotionAnalyzer({ now: () => 100_000 });
    const fall = createFallDetector();
    return {
        motion,
        fall,
        frame({ timestamp, dy = 0, shoulderDx = 0, posture = "EM PÉ", absent = false, lowConfidence = false }) {
            const pose = absent ? null : makePose({ dy, shoulderDx, lowConfidence });
            const sample = motion.recordInference({
                pose,
                analysisResult: makeAnalysisResult(pose, timestamp, absent ? "INCONCLUSIVA" : posture),
                videoWidth: 1000,
                videoHeight: 1000
            });
            return { sample, result: fall.analyze(sample) };
        },
        fail(timestamp) {
            motion.interruptContinuity();
            return fall.interrupt("FALHA_DE_INFERÊNCIA", timestamp);
        },
        switchCamera() {
            motion.resetContinuity();
            return fall.reset();
        }
    };
}

const S = FALL_DETECTION_STATES;
const noPossibleFall = results => assert.equal(
    results.some(result => result.state === S.POSSIVEL_QUEDA_NAO_CONFIRMADA),
    false
);

test("1. pessoa permanecendo em pé fica em monitoramento", () => {
    const fixture = createFixture();
    const results = [1000, 1250, 1500, 1750, 2000].map(timestamp =>
        fixture.frame({ timestamp }).result);
    assert.ok(results.slice(1).every(result => result.state === S.MONITORANDO));
});

test("2. sentar normalmente não evolui para possível queda", () => {
    const fixture = createFixture();
    const results = [
        fixture.frame({ timestamp: 1000 }).result,
        fixture.frame({ timestamp: 2000, dy: 40, posture: "SENTADA" }).result,
        fixture.frame({ timestamp: 2300, dy: 40, posture: "SENTADA" }).result,
        fixture.frame({ timestamp: 2600, dy: 40, posture: "SENTADA" }).result
    ];
    assert.equal(results[1].state, S.MUDANCA_DE_POSTURA);
    noPossibleFall(results);
});

test("3. levantar sem uma sequência suspeita permanece uma mudança normal", () => {
    const fixture = createFixture();
    const results = [
        fixture.frame({ timestamp: 1000, dy: 60, posture: "SENTADA" }).result,
        fixture.frame({ timestamp: 1500, dy: 40, posture: "SENTADA" }).result,
        fixture.frame({ timestamp: 2000, dy: 20, posture: "EM PÉ" }).result,
        fixture.frame({ timestamp: 2500, posture: "EM PÉ" }).result
    ];
    assert.equal(results[2].state, S.MUDANCA_DE_POSTURA);
    noPossibleFall(results);
});

test("4. deitar lentamente não é tratado como descida suspeita", () => {
    const fixture = createFixture();
    const results = [fixture.frame({ timestamp: 1000 }).result];
    for (let index = 1; index <= 8; index++) {
        results.push(fixture.frame({
            timestamp: 1000 + index * 400,
            dy: index * 15,
            shoulderDx: index * 20,
            posture: index === 8 ? "DEITADA" : "EM PÉ"
        }).result);
    }
    noPossibleFall(results);
    assert.ok(results.every(result => result.state !== S.DESCIDA_SUSPEITA));
});

test("5. inclinar-se sem deslocamento vertical rápido não inicia suspeita", () => {
    const fixture = createFixture();
    const results = [fixture.frame({ timestamp: 1000 }).result];
    for (let index = 1; index <= 4; index++) {
        results.push(fixture.frame({
            timestamp: 1000 + index * 250,
            dy: index * 5,
            shoulderDx: index * 40,
            posture: "INCONCLUSIVA"
        }).result);
    }
    noPossibleFall(results);
    assert.ok(results.every(result => result.state !== S.DESCIDA_SUSPEITA));
});

test("6. agachar gradualmente não combina os sinais exigidos", () => {
    const fixture = createFixture();
    const results = [fixture.frame({ timestamp: 1000 }).result];
    for (let index = 1; index <= 5; index++) {
        results.push(fixture.frame({ timestamp: 1000 + index * 400, dy: index * 18,
            shoulderDx: index * 15, posture: "INCONCLUSIVA" }).result);
    }
    noPossibleFall(results);
    assert.ok(results.every(result => result.state !== S.DESCIDA_SUSPEITA));
});

test("7. ajoelhar gradualmente não é concluído como queda", () => {
    const fixture = createFixture();
    const results = [fixture.frame({ timestamp: 1000 }).result];
    for (let index = 1; index <= 6; index++) {
        results.push(fixture.frame({ timestamp: 1000 + index * 400, dy: index * 20,
            shoulderDx: index * 12, posture: "INCONCLUSIVA" }).result);
    }
    noPossibleFall(results);
    assert.ok(results.every(result => result.state !== S.POSSIVEL_QUEDA_NAO_CONFIRMADA));
});

test("8. descida rápida junto com mudança angular inicia suspeita", () => {
    const fixture = createFixture();
    fixture.frame({ timestamp: 1000 });
    const { result } = fixture.frame({ timestamp: 1250, dy: 140, shoulderDx: 160, posture: "EM PÉ" });
    assert.equal(result.state, S.DESCIDA_SUSPEITA);
    assert.equal(result.indicators.rapidDownwardMovement, true);
    assert.equal(result.indicators.trunkOrientationChange, true);
});

test("9. transição para deitada após a descida suspeita é observada como mudança de postura", () => {
    const fixture = createFixture();
    fixture.frame({ timestamp: 1000 });
    fixture.frame({ timestamp: 1250, dy: 140, shoulderDx: 160, posture: "EM PÉ" });
    const { result } = fixture.frame({ timestamp: 1500, dy: 140, shoulderDx: 160, posture: "DEITADA" });
    assert.equal(result.state, S.MUDANCA_DE_POSTURA);
    assert.equal(result.indicators.lowPosition, true);
});

test("10. posição baixa persistente após múltiplos sinais entra em estado possível não confirmado", () => {
    const fixture = createFixture();
    fixture.frame({ timestamp: 1000 });
    fixture.frame({ timestamp: 1250, dy: 140, shoulderDx: 160, posture: "EM PÉ" });
    const results = [fixture.frame({ timestamp: 1500, dy: 140, shoulderDx: 160,
        posture: "DEITADA" }).result];
    for (const timestamp of [1800, 2100, 2400, 2700]) {
        results.push(fixture.frame({ timestamp, dy: 140, shoulderDx: 160, posture: "DEITADA" }).result);
    }
    assert.equal(results.at(-1).state, S.POSSIVEL_QUEDA_NAO_CONFIRMADA);
    assert.equal(results.at(-1).indicators.persistentLowPosition, true);
});

test("11. tentativa de recuperação encerra a sequência sem estado de queda confirmado", () => {
    const fixture = createFixture();
    fixture.frame({ timestamp: 1000 });
    fixture.frame({ timestamp: 1250, dy: 140, shoulderDx: 160, posture: "EM PÉ" });
    fixture.frame({ timestamp: 1500, dy: 140, shoulderDx: 160, posture: "DEITADA" });
    fixture.frame({ timestamp: 1800, dy: 140, shoulderDx: 160, posture: "DEITADA" });
    const { result } = fixture.frame({ timestamp: 2100, dy: 20, shoulderDx: 160, posture: "EM PÉ" });
    assert.equal(result.state, S.RECUPERACAO);
    assert.equal(result.indicators.recoveryMovement, true);
    assert.equal("fallEvent" in result, false);
});

test("12. ausência de pessoa torna a análise inconclusiva e quebra a continuidade", () => {
    const fixture = createFixture();
    fixture.frame({ timestamp: 1000 });
    const absent = fixture.frame({ timestamp: 1250, absent: true });
    assert.equal(absent.sample.centers.body, null);
    assert.equal(absent.result.state, S.ANALISE_INCONCLUSIVA);
    const resumed = fixture.frame({ timestamp: 1500, dy: 140, shoulderDx: 160 });
    assert.equal(resumed.sample.displacement.deltaYNormalized, null);
    assert.equal(resumed.result.state, S.MONITORANDO);
});

test("13. baixa visibilidade gera análise inconclusiva, nunca imobilidade", () => {
    const fixture = createFixture();
    fixture.frame({ timestamp: 1000 });
    const { sample, result } = fixture.frame({ timestamp: 1250, lowConfidence: true });
    assert.equal(sample.visibility, "VISIBILIDADE INSUFICIENTE");
    assert.equal(result.state, S.ANALISE_INCONCLUSIVA);
    assert.equal(result.indicators.lowPosition, false);
});

test("14. falha de inferência interrompe análise sem registrar ausência ou movimento", () => {
    const fixture = createFixture();
    fixture.frame({ timestamp: 1000 });
    const failure = fixture.fail(1100);
    assert.equal(failure.state, S.ANALISE_INCONCLUSIVA);
    assert.equal(failure.reason, "FALHA_DE_INFERÊNCIA");
    const resumed = fixture.frame({ timestamp: 1300, dy: 140, shoulderDx: 160 });
    assert.equal(resumed.sample.displacement.deltaYNormalized, null);
    assert.equal(resumed.result.state, S.MONITORANDO);
});

test("15. intervalo irregular interrompe a sequência usando timestamps reais", () => {
    const fixture = createFixture();
    fixture.frame({ timestamp: 1000 });
    const irregular = fixture.frame({ timestamp: 2600, dy: 140, shoulderDx: 160 });
    assert.equal(irregular.sample.displacement.intervalMs, 1600);
    assert.equal(irregular.result.state, S.ANALISE_INCONCLUSIVA);
    assert.equal(irregular.result.reason, "INTERVALO_DE_INFERÊNCIA_IRREGULAR");
});

test("16. troca de câmera inicia uma nova sequência independente", () => {
    const fixture = createFixture();
    fixture.frame({ timestamp: 1000 });
    fixture.switchCamera();
    const resumed = fixture.frame({ timestamp: 1250, dy: 140, shoulderDx: 160 });
    assert.equal(resumed.sample.displacement.deltaYNormalized, null);
    assert.equal(resumed.result.state, S.MONITORANDO);
});

test("17. sequência suspeita expira e volta a estado inconclusivo", () => {
    const fixture = createFixture();
    fixture.frame({ timestamp: 1000 });
    fixture.frame({ timestamp: 1250, dy: 140, shoulderDx: 160, posture: "EM PÉ" });
    let result;
    for (const timestamp of [2050, 2850, 3650, 4450, 5250, 6050, 6850, 7650, 8450, 9250, 10050]) {
        result = fixture.frame({ timestamp, dy: 140, shoulderDx: 160, posture: "EM PÉ" }).result;
    }
    assert.equal(result.state, S.ANALISE_INCONCLUSIVA);
    assert.equal(result.reason, "SEQUÊNCIA_SUSPEITA_EXPIRADA");
});

test("expiração automática encerra estado suspeito mesmo sem novas inferências", async () => {
    const expired = [];
    const detector = createFallDetector({
        config: { ...FALL_DETECTION_CONFIG, maxSequenceDurationMs: 25 },
        onExpire: result => expired.push(result)
    });
    const motion = createMotionAnalyzer();
    const sample = (timestamp, dy, shoulderDx) => {
        const pose = makePose({ dy, shoulderDx });
        const result = motion.recordInference({
            pose,
            analysisResult: makeAnalysisResult(pose, timestamp, "EM PÉ"),
            videoWidth: 1000,
            videoHeight: 1000
        });
        return detector.analyze(result);
    };
    sample(1000, 0, 0);
    assert.equal(sample(1250, 140, 160).state, S.DESCIDA_SUSPEITA);
    await new Promise(resolve => setTimeout(resolve, 45));
    assert.equal(detector.getState(), S.ANALISE_INCONCLUSIVA);
    assert.equal(expired.at(-1).reason, "SEQUÊNCIA_SUSPEITA_EXPIRADA");
});

test("18. estados internos não criam nem anexam registros de queda", () => {
    let records = 0;
    const app = { attachFallBodyRecord: () => { records++; } };
    installFallDetection(app);
    const fixture = createFixture();
    const baseline = fixture.frame({ timestamp: 1000 }).sample;
    app.evaluateFallDetection(baseline);
    const sample = fixture.frame({ timestamp: 1250, dy: 140, shoulderDx: 160, posture: "EM PÉ" }).sample;
    const result = app.evaluateFallDetection(sample);
    assert.equal(result.state, S.DESCIDA_SUSPEITA);
    assert.equal(records, 0);
    assert.equal("fallEvent" in result, false);
    assert.equal("fallRisk" in result, false);
});

test("diagnóstico fica desligado por padrão e registra transições, interrupções e amostras resumidas", () => {
    const originalWindow = globalThis.window;
    const originalInfo = console.info;
    const logs = [];
    globalThis.window = {};
    console.info = (...args) => logs.push(args);

    try {
        const app = { latestAnalysisResult: { fallRisk: { state: "NÃO AVALIADO" } } };
        installFallDetection(app);
        const diagnostics = globalThis.window.vidaSeguraDiagnostics;
        assert.equal(diagnostics.isEnabled(), false);
        assert.equal(diagnostics.getLatestSample(), null);

        const motion = createMotionAnalyzer();
        const recordSample = ({ timestamp, dy = 0, shoulderDx = 0, lowConfidence = false }) => {
            const pose = makePose({ dy, shoulderDx, lowConfidence });
            return motion.recordInference({
                pose,
                analysisResult: makeAnalysisResult(pose, timestamp, "EM PÉ"),
                videoWidth: 1000,
                videoHeight: 1000
            });
        };

        app.evaluateFallDetection(recordSample({ timestamp: 1000 }));
        assert.equal(logs.length, 0);
        diagnostics.enable();
        app.evaluateFallDetection(recordSample({ timestamp: 1100 }));
        app.evaluateFallDetection(recordSample({ timestamp: 1500 }));
        app.evaluateFallDetection(recordSample({ timestamp: 2100 }));
        app.evaluateFallDetection(recordSample({ timestamp: 2200, dy: 140, shoulderDx: 160 }));
        motion.interruptContinuity();
        app.interruptFallDetection("FALHA_DE_INFERÊNCIA", 2300);
        app.interruptFallDetection("FALHA_DE_INFERÊNCIA", 2350);
        app.evaluateFallDetection(recordSample({ timestamp: 2400, lowConfidence: true }));

        const transitions = logs.filter(([label]) => label.includes("Transição de estado"));
        const interruptions = logs.filter(([label]) => label.includes("Interrupção"));
        const samples = logs.filter(([label]) => label.endsWith("Amostra"));
        assert.ok(transitions.some(([, value]) => value.state === S.DESCIDA_SUSPEITA));
        assert.ok(transitions.some(([, value]) => value.reason === "FALHA_DE_INFERÊNCIA" &&
            value.continuity === "INTERROMPIDA"));
        assert.equal(interruptions.filter(([, value]) => value.reason === "FALHA_DE_INFERÊNCIA").length, 0);
        assert.ok(interruptions.some(([, value]) =>
            value.reason === "VISIBILIDADE_OU_GEOMETRIA_INSUFICIENTE" &&
            value.state === S.ANALISE_INCONCLUSIVA));
        assert.equal(samples.length, 2);

        const latest = diagnostics.getLatestSample();
        assert.equal(latest.state, S.ANALISE_INCONCLUSIVA);
        assert.equal(latest.reason, "VISIBILIDADE_OU_GEOMETRIA_INSUFICIENTE");
        assert.equal("joints" in latest, false);
        assert.equal("centers" in latest, false);
        assert.equal("x" in latest, false);
        assert.equal(app.latestAnalysisResult.fallRisk.state, "NÃO AVALIADO");

        diagnostics.disable();
        assert.equal(diagnostics.isEnabled(), false);
    } finally {
        console.info = originalInfo;
        if (originalWindow === undefined) delete globalThis.window;
        else globalThis.window = originalWindow;
    }
});

test("sentar ou deitar normalmente nunca gera automaticamente possível queda não confirmada", () => {
    const sit = createFixture();
    const sitting = [
        sit.frame({ timestamp: 1000 }).result,
        sit.frame({ timestamp: 2000, dy: 40, posture: "SENTADA" }).result,
        sit.frame({ timestamp: 3000, dy: 40, posture: "SENTADA" }).result,
        sit.frame({ timestamp: 4000, dy: 40, posture: "SENTADA" }).result
    ];
    const lie = createFixture();
    const lying = [lie.frame({ timestamp: 1000 }).result];
    for (let index = 1; index <= 8; index++) {
        lying.push(lie.frame({ timestamp: 1000 + index * 400, dy: index * 15,
            shoulderDx: index * 20, posture: index === 8 ? "DEITADA" : "EM PÉ" }).result);
    }
    noPossibleFall([...sitting, ...lying]);
});

test("queda a 25 fps (passos pequenos por frame) é detectada pela janela de descida", () => {
    const fixture = createFixture();
    const results = [fixture.frame({ timestamp: 1000 }).result];
    for (let index = 1; index <= 10; index++) {
        results.push(fixture.frame({ timestamp: 1000 + index * 40, dy: index * 30,
            shoulderDx: index * 18, posture: "EM PÉ" }).result);
    }
    assert.ok(results.some(result => result.state === S.DESCIDA_SUSPEITA));
    // Antes da janela, nenhum passo isolado de 0,03 chegava ao limiar de 0,08.
    assert.ok(30 / 1000 < FALL_DETECTION_CONFIG.rapidDownwardDisplacement);
});

test("sentar devagar a 25 fps continua sem descida suspeita", () => {
    const fixture = createFixture();
    const results = [fixture.frame({ timestamp: 1000 }).result];
    for (let index = 1; index <= 50; index++) {
        results.push(fixture.frame({ timestamp: 1000 + index * 40, dy: index * 4,
            shoulderDx: index * 3, posture: index > 40 ? "SENTADA" : "EM PÉ" }).result);
    }
    assert.ok(results.every(result => result.state !== S.DESCIDA_SUSPEITA));
    noPossibleFall(results);
});

test("perder a pessoa durante a descida suspeita é sinalizado como rastreio perdido", () => {
    const fixture = createFixture();
    fixture.frame({ timestamp: 1000 });
    fixture.frame({ timestamp: 1250, dy: 140, shoulderDx: 160, posture: "EM PÉ" });
    const { result } = fixture.frame({ timestamp: 1500, absent: true });
    assert.equal(result.state, S.ANALISE_INCONCLUSIVA);
    assert.equal(result.reason, "RASTREIO_PERDIDO_APÓS_DESCIDA");
    assert.equal(result.indicators.trackingLostAfterDescent, true);
    assert.equal(result.sequenceAgeMs, 250);
    const again = fixture.frame({ timestamp: 1750, absent: true });
    assert.equal(again.result.reason, "PESSOA_AUSENTE");
});

test("desabar sem girar o tronco (descida muito forte) também inicia suspeita", () => {
    const fixture = createFixture();
    fixture.frame({ timestamp: 1000 });
    const results = [];
    for (let index = 1; index <= 8; index++) {
        results.push(fixture.frame({ timestamp: 1000 + index * 40, dy: index * 30, posture: "EM PÉ" }).result);
    }
    assert.ok(results.some(result => result.state === S.DESCIDA_SUSPEITA));
});

test("sentar rápido numa cadeira (termina SENTADA, sem girar o tronco) não usa a regra da descida forte", () => {
    const fixture = createFixture();
    fixture.frame({ timestamp: 1000 });
    const results = [];
    for (let index = 1; index <= 8; index++) {
        results.push(fixture.frame({ timestamp: 1000 + index * 40, dy: index * 25,
            posture: index >= 2 ? "SENTADA" : "EM PÉ" }).result);
    }
    assert.ok(results.every(result => result.state !== S.DESCIDA_SUSPEITA));
});

test("pessoa longe da câmera (pequena no quadro) usa limiares proporcionais", () => {
    const config = { ...FALL_DETECTION_CONFIG, strongDropDisplacement: null };
    const detector = createFallDetector({ config });
    const motion = createMotionAnalyzer({ now: () => 100_000 });
    const small = (dy, shoulderDx, posture) => {
        const pose = { keypoints: jointNames.map(name => {
            const [x, y] = base[name];
            return { name, x: 500 + (x - 500) * 0.4 + (name.includes("shoulder") ? shoulderDx : 0),
                y: 300 + (y - 250) * 0.4 + dy, score: 0.95 };
        }) };
        return { pose, posture };
    };
    const run = (timestamp, { pose, posture }) => detector.analyze(motion.recordInference({
        pose, analysisResult: makeAnalysisResult(pose, timestamp, posture), videoWidth: 1000, videoHeight: 1000 }));
    for (let index = 0; index < 5; index++) run(1000 + index * 40, small(0, 0, "EM PÉ"));
    const results = [];
    for (let index = 1; index <= 6; index++) {
        results.push(run(1200 + index * 40, small(index * 12, index * 12, "EM PÉ")));
    }
    // 6 × 12 px = 0,072 do quadro: abaixo de 0,08, mas grande para uma pessoa com 26% da altura.
    assert.ok(results.some(result => result.state === S.DESCIDA_SUSPEITA));
});

test("tremor dos pontos (pessoa parada ou sentando devagar) não dispara descida", () => {
    const fixture = createFixture();
    const results = [];
    let seed = 7;
    const noise = () => { seed = (seed * 16807) % 2147483647; return (seed / 2147483647 - 0.5) * 12; };
    for (let index = 0; index < 150; index++) {
        const slowSit = index > 75 ? (index - 75) * 2 : 0; // ~0,05 do quadro por segundo
        results.push(fixture.frame({ timestamp: 1000 + index * 33, dy: slowSit + noise(),
            shoulderDx: noise(), posture: index > 120 ? "SENTADA" : "EM PÉ" }).result);
    }
    assert.ok(results.every(result => result.state !== S.DESCIDA_SUSPEITA));
    noPossibleFall(results);
});
