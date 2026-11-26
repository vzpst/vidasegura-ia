import test from "node:test";
import assert from "node:assert/strict";

const { createFallSeverityTracker, FALL_SEVERITY_CONFIG, SEVERITY_LEVELS } =
    await import(new URL("../js/fall-severity.js", import.meta.url));
const { FALL_DETECTION_STATES: S } = await import(new URL("../js/fall-detection.js", import.meta.url));

const config = FALL_SEVERITY_CONFIG;

function sample(timestamp, { y = 0.4, posture = "EM PÉ", speed = 0, detected = true, visibility = "BOA VISIBILIDADE" } = {}) {
    return {
        timestamp,
        personDetection: { detected },
        visibility,
        posture,
        // speed > 0 simula movimento real do corpo (o centro se desloca no quadro).
        centers: { body: detected ? { normalizedX: 0.5 + (speed > 0 ? 0.08 * Math.sin(timestamp / 150) : 0),
            normalizedY: y, source: "bilateral" } : null },
        displacement: { normalizedXPerSecond: speed, normalizedYPerSecond: 0 }
    };
}

function fall(state, timestamp, sequenceAgeMs = null) {
    return { state, timestamp, sequenceAgeMs };
}

// Pessoa em pé, descida rápida e posição baixa persistente.
function startPossibleFall(tracker, closed = []) {
    tracker.update(fall(S.MONITORANDO, 0), sample(0));
    tracker.update(fall(S.DESCIDA_SUSPEITA, 500), sample(500, { y: 0.55, posture: "INCONCLUSIVA", speed: 0.6 }));
    return tracker.update(
        fall(S.POSSIVEL_QUEDA_NAO_CONFIRMADA, 2000, 1500),
        sample(2000, { y: 0.7, posture: "DEITADA", speed: 0.2 })
    );
}

test("recuperação durante a descida suspeita vira nível 1, sem alarme", () => {
    const closed = [];
    const tracker = createFallSeverityTracker({ onEventClosed: event => closed.push(event) });
    tracker.update(fall(S.MONITORANDO, 0), sample(0));
    tracker.update(fall(S.DESCIDA_SUSPEITA, 500), sample(500, { y: 0.55, speed: 0.6 }));
    const result = tracker.update(fall(S.RECUPERACAO, 1500), sample(1500));
    assert.equal(result.level, SEVERITY_LEVELS.RECOVERED);
    assert.equal(result.alarm, "nenhum");
    assert.equal(closed.length, 1);
    assert.equal(closed[0].outcome, "RECUPERADA_LOGO_APÓS_IMPACTO");
});

test("possível queda começa no nível 1 e não alarma durante a carência", () => {
    const tracker = createFallSeverityTracker();
    const result = startPossibleFall(tracker);
    assert.equal(result.level, SEVERITY_LEVELS.RECOVERED);
    assert.equal(result.alarm, "nenhum");
    assert.equal(result.phase, "EM_ACOMPANHAMENTO");
    assert.equal(result.elapsedSinceImpactMs, 1500);
});

test("levantar logo após o impacto encerra o evento sem nenhum alarme", () => {
    const closed = [];
    const tracker = createFallSeverityTracker({ onEventClosed: event => closed.push(event) });
    startPossibleFall(tracker);
    let result;
    for (let t = 3000; t < 3000 + config.recoverySamplesRequired * 100; t += 100) {
        result = tracker.update(fall(S.OBSERVACAO_POS_MOVIMENTO, t), sample(t, { posture: "EM PÉ" }));
    }
    assert.equal(result.alarm, "nenhum");
    assert.equal(result.phase, "ENCERRADO");
    assert.equal(closed[0].outcome, "RECUPERADA_LOGO_APÓS_IMPACTO");
    assert.equal(closed[0].maxLevel, SEVERITY_LEVELS.RECOVERED);
});

test("chão além da carência com movimento sobe para nível 2 (aviso)", () => {
    const tracker = createFallSeverityTracker();
    startPossibleFall(tracker);
    let result;
    for (let t = 2500; t <= 500 + config.gracePeriodMs + 1000; t += 250) {
        result = tracker.update(fall(S.ANALISE_INCONCLUSIVA, t), sample(t, { y: 0.7, posture: "DEITADA", speed: 0.2 }));
    }
    assert.equal(result.level, SEVERITY_LEVELS.ATTENTION);
    assert.equal(result.alarm, "aviso");
});

test("imobilidade prolongada após a carência sobe para nível 3 (alerta)", () => {
    const tracker = createFallSeverityTracker();
    startPossibleFall(tracker);
    let result;
    const end = 500 + config.gracePeriodMs + config.immobileForCriticalMs + 500;
    for (let t = 2500; t <= end; t += 250) {
        result = tracker.update(fall(S.ANALISE_INCONCLUSIVA, t), sample(t, { y: 0.7, posture: "DEITADA", speed: 0 }));
    }
    assert.ok(end - 500 < config.criticalAfterImpactMs, "o teste precisa isolar a regra de imobilidade");
    assert.equal(result.level, SEVERITY_LEVELS.PRIORITY);
    assert.equal(result.alarm, "alerta");
});

test("tempo no chão além do limite crítico sobe para nível 3 mesmo com movimento", () => {
    const tracker = createFallSeverityTracker();
    startPossibleFall(tracker);
    let result;
    for (let t = 2500; t <= 500 + config.criticalAfterImpactMs + 500; t += 500) {
        result = tracker.update(fall(S.ANALISE_INCONCLUSIVA, t), sample(t, { y: 0.7, posture: "DEITADA", speed: 0.2 }));
    }
    assert.equal(result.level, SEVERITY_LEVELS.PRIORITY);
});

test("'Estou bem' encerra o evento e desliga o aviso", () => {
    const closed = [];
    const tracker = createFallSeverityTracker({ onEventClosed: event => closed.push(event) });
    startPossibleFall(tracker);
    for (let t = 2500; t <= 500 + config.gracePeriodMs + 1000; t += 250) {
        tracker.update(fall(S.ANALISE_INCONCLUSIVA, t), sample(t, { y: 0.7, posture: "DEITADA", speed: 0.2 }));
    }
    const result = tracker.acknowledge(20_000);
    assert.equal(result.alarm, "nenhum");
    assert.equal(closed[0].outcome, "PESSOA_INFORMOU_ESTAR_BEM");
    assert.equal(closed[0].maxLevel, SEVERITY_LEVELS.ATTENTION);
});

test("sem evidência visual o nível fica congelado e não escala", () => {
    const tracker = createFallSeverityTracker();
    startPossibleFall(tracker);
    let result;
    for (let t = 2500; t <= 500 + config.criticalAfterImpactMs + 5000; t += 500) {
        result = tracker.update(fall(S.ANALISE_INCONCLUSIVA, t), sample(t, { detected: false }));
    }
    assert.equal(result.level, SEVERITY_LEVELS.RECOVERED);
    assert.equal(result.evidence, "insuficiente");
    assert.equal(result.alarm, "nenhum");
});

test("movimento normal sem descida suspeita permanece no nível 0", () => {
    const tracker = createFallSeverityTracker();
    let result;
    for (let t = 0; t <= 5000; t += 250) {
        result = tracker.update(fall(S.MONITORANDO, t), sample(t, { speed: 0.1 }));
    }
    assert.equal(result.level, SEVERITY_LEVELS.NONE);
    assert.equal(result.alarm, "nenhum");
});

test("reset (troca ou desligamento da câmera) limpa o evento sem gerar alarme", () => {
    const closed = [];
    const tracker = createFallSeverityTracker({ onEventClosed: event => closed.push(event) });
    startPossibleFall(tracker);
    const result = tracker.reset();
    assert.equal(result.level, SEVERITY_LEVELS.NONE);
    assert.equal(closed.length, 0);
});

test("pessoa some logo após descida rápida: evento sem alarme na carência, depois só aviso", () => {
    const tracker = createFallSeverityTracker();
    tracker.update(fall(S.MONITORANDO, 0), sample(0));
    tracker.update(fall(S.DESCIDA_SUSPEITA, 500), sample(500, { y: 0.55, speed: 0.6 }));
    const lost = { state: S.ANALISE_INCONCLUSIVA, timestamp: 800, sequenceAgeMs: 300,
        indicators: { trackingLostAfterDescent: true } };
    let result = tracker.update(lost, sample(800, { detected: false }));
    assert.equal(result.phase, "EM_ACOMPANHAMENTO");
    assert.equal(result.trackingLost, true);
    assert.equal(result.alarm, "nenhum");
    for (let t = 1000; t <= 500 + config.criticalAfterImpactMs + 5000; t += 500) {
        result = tracker.update(fall(S.ANALISE_INCONCLUSIVA, t), sample(t, { detected: false }));
    }
    assert.equal(result.level, SEVERITY_LEVELS.ATTENTION);
    assert.equal(result.alarm, "aviso");
});

test("pessoa some após descida e volta em pé: encerra como recuperação", () => {
    const closed = [];
    const tracker = createFallSeverityTracker({ onEventClosed: event => closed.push(event) });
    tracker.update(fall(S.MONITORANDO, 0), sample(0));
    tracker.update(fall(S.DESCIDA_SUSPEITA, 500), sample(500, { y: 0.55, speed: 0.6 }));
    tracker.update({ state: S.ANALISE_INCONCLUSIVA, timestamp: 800, sequenceAgeMs: 300,
        indicators: { trackingLostAfterDescent: true } }, sample(800, { detected: false }));
    let result;
    for (let t = 1500; t < 1500 + config.recoverySamplesRequired * 100; t += 100) {
        result = tracker.update(fall(S.MONITORANDO, t), sample(t));
    }
    assert.equal(result.phase, "ENCERRADO");
    assert.equal(closed[0].outcome, "RECUPERADA_LOGO_APÓS_IMPACTO");
});

test("tremor normal dos pontos ainda conta como imobilidade (nível 3 rápido)", () => {
    const tracker = createFallSeverityTracker();
    startPossibleFall(tracker);
    let result;
    let t = 2000;
    for (; t <= 500 + config.gracePeriodMs + config.immobileForCriticalMs + 500; t += 40) {
        const jitter = (Math.sin(t) * 0.006);
        const s = sample(t, { y: 0.7 + jitter, posture: "DEITADA" });
        s.centers.body.normalizedX = 0.5 - jitter;
        result = tracker.update(fall(S.ANALISE_INCONCLUSIVA, t), s);
    }
    assert.equal(result.level, SEVERITY_LEVELS.PRIORITY);
    assert.ok(t - 500 < config.criticalAfterImpactMs);
});

test("nível 1 aparece já no início do acompanhamento, sem som", () => {
    const tracker = createFallSeverityTracker();
    const result = startPossibleFall(tracker);
    assert.equal(result.label, "Nível 1 · Possível queda detectada");
    assert.equal(result.alarm, "nenhum");
});
