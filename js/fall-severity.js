// Níveis de gravidade para POSSÍVEIS quedas (experimental, sem validação clínica).
//
// Este módulo não detecta quedas sozinho: ele consome a saída da máquina de
// estados de fall-detection.js e as amostras de motion-analysis.js e decide
// quanto o evento merece atenção, com base no que acontece DEPOIS do impacto.
// Objetivo: não disparar alarme para quem se recuperou logo após o impacto.
//
//   Nível 0  Sem evento.
//   Nível 1  Desequilíbrio / possível queda com recuperação. Só histórico.
//   Nível 2  Posição baixa além da carência. Aviso local com "Estou bem".
//   Nível 3  Posição baixa prolongada ou imóvel. Alerta destacado na página.
//
// A ausência de evidência (pessoa fora do quadro, visibilidade ruim) congela o
// nível atual e marca a análise como inconclusiva; ela nunca aumenta o nível.

import { FALL_DETECTION_STATES } from "./fall-detection.js";

// PROVISÓRIOS: definidos para demonstração, não são limiares clínicos.
export const FALL_SEVERITY_CONFIG = Object.freeze({
    status: "PROVISÓRIO — ajustar após testes com câmera real e dados representativos",
    gracePeriodMs: 10_000,            // carência após o impacto antes de qualquer aviso
    criticalAfterImpactMs: 30_000,    // tempo no chão que leva ao nível 3
    immobileForCriticalMs: 15_000,    // imobilidade contínua (após a carência) que leva ao nível 3
    immobileSpeedPerSecond: 0.03,     // velocidade normalizada abaixo da qual a pessoa é considerada imóvel
    movementSmoothing: 0.3,           // suavização exponencial da velocidade observada
    lowOffsetFromReference: 0.07,     // quanto o centro corporal precisa estar abaixo da referência
    recoveryMarginFromReference: 0.05,// centro de volta perto da referência conta como recuperação
    recoverySamplesRequired: 3,       // amostras seguidas compatíveis com recuperação
    candidateTimeoutMs: 8_000         // descida suspeita sem desfecho é descartada
});

export const SEVERITY_LEVELS = Object.freeze({
    NONE: 0,
    RECOVERED: 1,
    ATTENTION: 2,
    PRIORITY: 3
});

const LEVEL_LABELS = Object.freeze({
    0: "Sem evento",
    1: "Nível 1 · Desequilíbrio ou possível queda com recuperação",
    2: "Nível 2 · Possível queda: atenção",
    3: "Nível 3 · Possível queda: prioritária"
});

const LEVEL_ALARMS = Object.freeze({ 0: "nenhum", 1: "nenhum", 2: "aviso", 3: "alerta" });

const finite = Number.isFinite;

function sampleHasEvidence(sample) {
    return sample?.personDetection?.detected === true &&
        finite(sample?.centers?.body?.normalizedY) &&
        (sample.visibility === "BOA VISIBILIDADE" || sample.visibility === "VISIBILIDADE PARCIAL");
}

function sampleSpeed(sample) {
    const { normalizedXPerSecond: vx, normalizedYPerSecond: vy } = sample?.displacement ?? {};
    if (!finite(vx) || !finite(vy)) return null;
    return Math.hypot(vx, vy);
}

export function createFallSeverityTracker({ config = FALL_SEVERITY_CONFIG, onEventClosed = () => {} } = {}) {
    let lastStableSample = null;   // última amostra com evidência fora de qualquer evento
    let candidate = null;          // descida suspeita ainda sem desfecho
    let event = null;              // possível queda em acompanhamento
    let lastResult = idleResult(null);

    function idleResult(timestamp, extra = {}) {
        return {
            level: SEVERITY_LEVELS.NONE,
            label: LEVEL_LABELS[0],
            alarm: "nenhum",
            phase: "SEM_EVENTO",
            evidence: "adequada",
            elapsedSinceImpactMs: null,
            immobileMs: null,
            outcome: null,
            timestamp,
            ...extra
        };
    }

    function eventResult(timestamp) {
        const level = event.level;
        return {
            level,
            label: LEVEL_LABELS[level],
            alarm: event.evidenceMissing ? (level >= 2 ? LEVEL_ALARMS[level] : "nenhum") : LEVEL_ALARMS[level],
            phase: "EM_ACOMPANHAMENTO",
            evidence: event.evidenceMissing ? "insuficiente" : "adequada",
            trackingLost: Boolean(event.trackingLost && event.evidenceMissing),
            elapsedSinceImpactMs: Math.max(0, timestamp - event.impactAt),
            immobileMs: event.immobileSince !== null ? Math.max(0, timestamp - event.immobileSince) : 0,
            outcome: null,
            timestamp
        };
    }

    function closeEvent(timestamp, outcome, levelOverride = null) {
        const summary = {
            startedAt: event?.impactAt ?? candidate?.startedAt ?? timestamp,
            endedAt: timestamp,
            maxLevel: levelOverride ?? event?.level ?? SEVERITY_LEVELS.RECOVERED,
            outcome
        };
        event = null;
        candidate = null;
        onEventClosed(summary);
        lastResult = idleResult(timestamp, {
            phase: "ENCERRADO",
            outcome,
            level: summary.maxLevel === SEVERITY_LEVELS.RECOVERED ? SEVERITY_LEVELS.RECOVERED : SEVERITY_LEVELS.NONE,
            label: summary.maxLevel === SEVERITY_LEVELS.RECOVERED ? LEVEL_LABELS[1] : LEVEL_LABELS[0]
        });
        return lastResult;
    }

    function startEvent(timestamp, fallResult) {
        const impactAt = finite(fallResult?.sequenceAgeMs) ? timestamp - fallResult.sequenceAgeMs
            : candidate?.startedAt ?? timestamp;
        event = {
            impactAt,
            referenceY: candidate?.referenceY ?? lastStableSample?.centers?.body?.normalizedY ?? null,
            level: SEVERITY_LEVELS.RECOVERED,
            smoothedSpeed: null,
            immobileSince: null,
            recoveryStreak: 0,
            evidenceMissing: false,
            trackingLost: false
        };
        candidate = null;
    }

    function isRecoverySample(sample) {
        if (sample.posture === "EM PÉ") return true;
        const y = sample.centers.body.normalizedY;
        return finite(event.referenceY) && sample.posture !== "DEITADA" &&
            y <= event.referenceY + config.recoveryMarginFromReference;
    }

    function updateMovement(sample, timestamp) {
        const speed = sampleSpeed(sample);
        if (speed === null) return;
        event.smoothedSpeed = event.smoothedSpeed === null ? speed
            : event.smoothedSpeed + config.movementSmoothing * (speed - event.smoothedSpeed);
        if (event.smoothedSpeed < config.immobileSpeedPerSecond) event.immobileSince ??= timestamp;
        else event.immobileSince = null;
    }

    function escalate(timestamp) {
        const elapsed = timestamp - event.impactAt;
        if (elapsed < config.gracePeriodMs) return;
        let next = SEVERITY_LEVELS.ATTENTION;
        const immobileFor = event.immobileSince !== null
            ? timestamp - Math.max(event.immobileSince, event.impactAt + config.gracePeriodMs) : 0;
        if (elapsed >= config.criticalAfterImpactMs || immobileFor >= config.immobileForCriticalMs) {
            next = SEVERITY_LEVELS.PRIORITY;
        }
        // O nível só sobe durante o evento; recuperação encerra o evento.
        event.level = Math.max(event.level, next);
    }

    function update(fallResult, sample) {
        const timestamp = finite(sample?.timestamp) ? sample.timestamp
            : finite(fallResult?.timestamp) ? fallResult.timestamp : null;
        if (timestamp === null) return lastResult;
        const evidence = sampleHasEvidence(sample);
        const state = fallResult?.state;

        if (event) {
            if (!evidence) {
                // Sem evidência: congela o nível, não escala e não encerra.
                event.evidenceMissing = true;
                event.recoveryStreak = 0;
                // Saiu de vista logo após a descida: sem como ver recuperação, sobe só até "atenção".
                if (event.trackingLost && timestamp - event.impactAt >= config.gracePeriodMs) {
                    event.level = Math.max(event.level, SEVERITY_LEVELS.ATTENTION);
                }
                lastResult = eventResult(timestamp);
                return lastResult;
            }
            event.evidenceMissing = false;
            if (isRecoverySample(sample)) {
                event.recoveryStreak++;
                if (event.recoveryStreak >= config.recoverySamplesRequired) {
                    const quick = timestamp - event.impactAt < config.gracePeriodMs;
                    return closeEvent(timestamp, quick ? "RECUPERADA_LOGO_APÓS_IMPACTO" : "RECUPERADA");
                }
            } else {
                event.recoveryStreak = 0;
            }
            updateMovement(sample, timestamp);
            escalate(timestamp);
            lastResult = eventResult(timestamp);
            return lastResult;
        }

        if (state === FALL_DETECTION_STATES.POSSIVEL_QUEDA_NAO_CONFIRMADA && evidence) {
            startEvent(timestamp, fallResult);
            updateMovement(sample, timestamp);
            escalate(timestamp);
            lastResult = eventResult(timestamp);
            return lastResult;
        }

        if (fallResult?.indicators?.trackingLostAfterDescent) {
            startEvent(timestamp, fallResult);
            event.trackingLost = true;
            event.evidenceMissing = true;
            lastResult = eventResult(timestamp);
            return lastResult;
        }

        if (state === FALL_DETECTION_STATES.DESCIDA_SUSPEITA && !candidate) {
            candidate = {
                startedAt: timestamp,
                referenceY: lastStableSample?.centers?.body?.normalizedY ?? null
            };
        }

        if (candidate) {
            if (state === FALL_DETECTION_STATES.RECUPERACAO) {
                return closeEvent(timestamp, "RECUPERADA_LOGO_APÓS_IMPACTO", SEVERITY_LEVELS.RECOVERED);
            }
            if (timestamp - candidate.startedAt > config.candidateTimeoutMs) {
                candidate = null;
            }
            lastResult = idleResult(timestamp, { phase: candidate ? "OBSERVANDO_DESCIDA" : "SEM_EVENTO" });
            return lastResult;
        }

        if (evidence && (state === FALL_DETECTION_STATES.MONITORANDO || state === FALL_DETECTION_STATES.MUDANCA_DE_POSTURA)) {
            lastStableSample = sample;
        }
        // Mantém o desfecho anterior visível até a próxima mudança relevante.
        if (lastResult.phase !== "ENCERRADO") lastResult = idleResult(timestamp);
        return lastResult;
    }

    // Botão "Estou bem": encerra o evento sem alarme. Não conta como validação.
    function acknowledge(timestamp = Date.now()) {
        if (!event) return lastResult;
        return closeEvent(timestamp, "PESSOA_INFORMOU_ESTAR_BEM");
    }

    function reset(timestamp = null) {
        event = null;
        candidate = null;
        lastStableSample = null;
        lastResult = idleResult(timestamp);
        return lastResult;
    }

    return { update, acknowledge, reset, getResult: () => ({ ...lastResult }) };
}

export function installFallSeverity(app) {
    const tracker = createFallSeverityTracker({
        onEventClosed: summary => app.recordFallSeverityEvent?.(summary)
    });
    app.latestFallSeverity = tracker.getResult();
    // Troca, desligamento ou reinício da câmera também zeram a gravidade.
    const resetFallDetection = app.resetFallDetection;
    if (typeof resetFallDetection === "function") {
        app.resetFallDetection = (...args) => {
            const result = resetFallDetection(...args);
            app.resetFallSeverity();
            return result;
        };
    }
    app.evaluateFallSeverity = (fallResult, sample) => {
        app.latestFallSeverity = tracker.update(fallResult, sample);
        app.renderFallSeverity?.(app.latestFallSeverity);
        return app.latestFallSeverity;
    };
    app.acknowledgeFallSeverity = () => {
        app.latestFallSeverity = tracker.acknowledge(Date.now());
        app.renderFallSeverity?.(app.latestFallSeverity);
        return app.latestFallSeverity;
    };
    app.resetFallSeverity = () => {
        app.latestFallSeverity = tracker.reset();
        app.renderFallSeverity?.(app.latestFallSeverity);
        return app.latestFallSeverity;
    };
}
