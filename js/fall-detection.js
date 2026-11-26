export const FALL_DETECTION_STATES = Object.freeze({
    MONITORANDO: "MONITORANDO",
    DESCIDA_SUSPEITA: "DESCIDA_SUSPEITA",
    MUDANCA_DE_POSTURA: "MUDANCA_DE_POSTURA",
    OBSERVACAO_POS_MOVIMENTO: "OBSERVACAO_POS_MOVIMENTO",
    RECUPERACAO: "RECUPERACAO",
    POSSIVEL_QUEDA_NAO_CONFIRMADA: "POSSIVEL_QUEDA_NAO_CONFIRMADA",
    ANALISE_INCONCLUSIVA: "ANALISE_INCONCLUSIVA"
});

// Todos os parâmetros são PROVISÓRIOS. Devem ser revistos com gravações reais
// representativas; não são limiares clínicos ou cientificamente validados.
export const FALL_DETECTION_CONFIG = Object.freeze({
    status: "PROVISÓRIO — ajustar após testes com câmera real e dados representativos",
    maxInferenceIntervalMs: 1000,
    minimumObservedJoints: 6,
    minimumObservedTorsoJoints: 3,
    // Perfil: ombro + quadril do mesmo lado bastam quando o centro corporal
    // foi calculado a partir de um único lado (motion-analysis.js).
    minimumObservedTorsoJointsSingleSide: 2,
    rapidDownwardDisplacement: 0.08,
    rapidDownwardSpeedPerSecond: 0.45,
    significantTrunkAngleChangeDegrees: 28,
    lowPositionDisplacement: 0.14,
    minimumLowPersistenceMs: 1200,
    minimumLowObservationSamples: 3,
    recoveryUpwardDisplacement: 0.05,
    recoveryUpwardSpeedPerSecond: 0.20,
    maxSequenceDurationMs: 8000
});

const KNOWN_POSTURES = new Set(["EM PÉ", "SENTADA", "DEITADA"]);
const LOWER_POSTURES = new Set(["SENTADA", "DEITADA"]);
const finite = Number.isFinite;

function angleDifferenceDegrees(a, b) {
    const difference = ((a - b + 540) % 360) - 180;
    return Math.abs(difference);
}

function usableSample(sample, config) {
    const centerY = sample?.centers?.body?.normalizedY;
    const torsoCount = sample?.observedJointsByRegion?.torso;
    const observedCount = sample?.boundingBoxPointsUsed;
    const singleSide = sample?.centers?.body?.source === "left" || sample?.centers?.body?.source === "right";
    const minimumTorso = singleSide
        ? config.minimumObservedTorsoJointsSingleSide ?? config.minimumObservedTorsoJoints
        : config.minimumObservedTorsoJoints;
    return sample?.personDetection?.detected === true &&
        (sample.visibility === "BOA VISIBILIDADE" || sample.visibility === "VISIBILIDADE PARCIAL") &&
        finite(centerY) && finite(sample.trunkAngleDegrees) &&
        finite(torsoCount) && torsoCount >= minimumTorso &&
        finite(observedCount) && observedCount >= config.minimumObservedJoints;
}

export function createFallDetector({ config = FALL_DETECTION_CONFIG, onExpire = () => {} } = {}) {
    let state = FALL_DETECTION_STATES.ANALISE_INCONCLUSIVA;
    let previousSample = null;
    let continuityId = null;
    let reference = null;
    let sequence = null;
    let sequenceExpiryTimer = null;
    let sequenceToken = 0;

    function snapshot(timestamp, reason, indicators = {}) {
        return {
            state,
            timestamp: finite(timestamp) ? timestamp : null,
            reason,
            indicators: {
                rapidDownwardMovement: false,
                trunkOrientationChange: false,
                postureTransition: false,
                lowPosition: false,
                persistentLowPosition: false,
                recoveryMovement: false,
                adequateVisibility: false,
                continuousInference: false,
                referenceEstablished: Boolean(reference?.bodyCenterNormalizedY !== undefined),
                ...indicators
            },
            sequenceAgeMs: sequence && finite(timestamp) ? Math.max(0, timestamp - sequence.startedAt) : null
        };
    }

    function clearSequence() {
        sequence = null;
        sequenceToken++;
        if (sequenceExpiryTimer !== null) clearTimeout(sequenceExpiryTimer);
        sequenceExpiryTimer = null;
    }

    function scheduleSequenceExpiration() {
        const expiringSequence = sequence;
        const token = ++sequenceToken;
        sequenceExpiryTimer = setTimeout(() => {
            if (sequence !== expiringSequence || sequenceToken !== token) return;
            sequenceExpiryTimer = null;
            breakContinuity();
            state = FALL_DETECTION_STATES.ANALISE_INCONCLUSIVA;
            onExpire(snapshot(Date.now(), "SEQUÊNCIA_SUSPEITA_EXPIRADA"));
        }, config.maxSequenceDurationMs);
        sequenceExpiryTimer?.unref?.();
    }

    function setReference(sample) {
        reference = {
            posture: KNOWN_POSTURES.has(sample.posture) ? sample.posture : null,
            bodyCenterNormalizedY: sample.centers.body.normalizedY,
            trunkAngleDegrees: sample.trunkAngleDegrees
        };
    }

    function breakContinuity(sample = null) {
        previousSample = null;
        continuityId = null;
        reference = null;
        clearSequence();
        if (sample && usableSample(sample, config)) {
            previousSample = sample;
            continuityId = sample.continuityId ?? null;
            setReference(sample);
        }
    }

    function analyze(sample) {
        const timestamp = sample?.timestamp;
        if (!finite(timestamp)) {
            breakContinuity();
            state = FALL_DETECTION_STATES.ANALISE_INCONCLUSIVA;
            return snapshot(null, "TIMESTAMP_INVALIDO");
        }

        if (sample?.personDetection?.detected !== true) {
            breakContinuity();
            state = FALL_DETECTION_STATES.ANALISE_INCONCLUSIVA;
            return snapshot(timestamp, "PESSOA_AUSENTE");
        }

        if (!usableSample(sample, config)) {
            breakContinuity();
            state = FALL_DETECTION_STATES.ANALISE_INCONCLUSIVA;
            return snapshot(timestamp, "VISIBILIDADE_OU_GEOMETRIA_INSUFICIENTE");
        }

        const intervalMs = sample.displacement?.intervalMs;
        if (finite(intervalMs) && (intervalMs <= 0 || intervalMs > config.maxInferenceIntervalMs)) {
            breakContinuity(sample);
            state = FALL_DETECTION_STATES.ANALISE_INCONCLUSIVA;
            return snapshot(timestamp, "INTERVALO_DE_INFERÊNCIA_IRREGULAR", {
                adequateVisibility: true
            });
        }

        const sampleContinuityId = sample.continuityId ?? null;
        if (previousSample && (sampleContinuityId !== continuityId || timestamp <= previousSample.timestamp)) {
            breakContinuity(sample);
            state = FALL_DETECTION_STATES.ANALISE_INCONCLUSIVA;
            return snapshot(timestamp, "CONTINUIDADE_INTERROMPIDA", {
                adequateVisibility: true
            });
        }

        const visible = { adequateVisibility: true };
        if (!previousSample) {
            previousSample = sample;
            continuityId = sampleContinuityId;
            setReference(sample);
            state = FALL_DETECTION_STATES.MONITORANDO;
            return snapshot(timestamp, "REFERÊNCIA_CORPORAL_ESTABELECIDA", visible);
        }

        const deltaY = sample.displacement?.deltaYNormalized;
        const verticalSpeed = sample.displacement?.normalizedYPerSecond;
        const hasVerticalMovement = finite(deltaY) && finite(verticalSpeed);
        const posture = KNOWN_POSTURES.has(sample.posture) ? sample.posture : null;
        const previousPosture = KNOWN_POSTURES.has(previousSample.posture) ? previousSample.posture : null;
        const postureTransition = Boolean(posture && previousPosture && posture !== previousPosture);
        const angleChanged = angleDifferenceDegrees(sample.trunkAngleDegrees,
            previousSample.trunkAngleDegrees) >= config.significantTrunkAngleChangeDegrees ||
            angleDifferenceDegrees(sample.trunkAngleDegrees,
                reference?.trunkAngleDegrees ?? previousSample.trunkAngleDegrees) >= config.significantTrunkAngleChangeDegrees;
        const rapidDownward = hasVerticalMovement &&
            deltaY >= config.rapidDownwardDisplacement &&
            verticalSpeed >= config.rapidDownwardSpeedPerSecond;
        const lowerPostureTransition = postureTransition && LOWER_POSTURES.has(posture) &&
            (previousPosture === "EM PÉ" || previousPosture === "SENTADA");
        const indicators = {
            ...visible,
            continuousInference: finite(intervalMs) && intervalMs > 0,
            rapidDownwardMovement: rapidDownward,
            trunkOrientationChange: angleChanged,
            postureTransition
        };

        if (!sequence && rapidDownward && (angleChanged || lowerPostureTransition)) {
            sequence = {
                startedAt: timestamp,
                basePosture: reference?.posture ?? previousPosture,
                downwardDisplacement: Math.max(0, deltaY),
                observationStartedAt: null,
                lowStartedAt: null,
                lowSamples: 0,
                everLow: false
            };
            scheduleSequenceExpiration();
            state = FALL_DETECTION_STATES.DESCIDA_SUSPEITA;
            previousSample = sample;
            setReference(sample);
            return snapshot(timestamp, "DESCIDA_RÁPIDA_COM_SINAL_CORROBORADOR", indicators);
        }

        if (!sequence) {
            state = postureTransition
                ? FALL_DETECTION_STATES.MUDANCA_DE_POSTURA
                : FALL_DETECTION_STATES.MONITORANDO;
            if (postureTransition) setReference(sample);
            previousSample = sample;
            return snapshot(timestamp, postureTransition ? "TRANSIÇÃO_DE_POSTURA" : "ACOMPANHAMENTO_NORMAL", indicators);
        }

        if (hasVerticalMovement) sequence.downwardDisplacement += deltaY;
        const lowPosition = sequence.downwardDisplacement >= config.lowPositionDisplacement &&
            (LOWER_POSTURES.has(posture) || posture === null);
        if (lowPosition) {
            sequence.everLow = true;
            sequence.lowStartedAt ??= timestamp;
            sequence.lowSamples++;
        } else {
            sequence.lowStartedAt = null;
            sequence.lowSamples = 0;
        }

        const recoveryMovement = sequence.everLow && hasVerticalMovement &&
            deltaY <= -config.recoveryUpwardDisplacement &&
            verticalSpeed <= -config.recoveryUpwardSpeedPerSecond &&
            (sequence.downwardDisplacement <= config.lowPositionDisplacement / 2 || posture === "EM PÉ");
        const posturalRecovery = sequence.everLow && posture === "EM PÉ" &&
            previousPosture !== "EM PÉ";
        if (recoveryMovement || posturalRecovery) {
            indicators.recoveryMovement = true;
            state = FALL_DETECTION_STATES.RECUPERACAO;
            clearSequence();
            setReference(sample);
            previousSample = sample;
            return snapshot(timestamp, "MOVIMENTO_COMPATÍVEL_COM_RECUPERAÇÃO", indicators);
        }

        if (timestamp - sequence.startedAt > config.maxSequenceDurationMs) {
            clearSequence();
            state = FALL_DETECTION_STATES.ANALISE_INCONCLUSIVA;
            setReference(sample);
            previousSample = sample;
            return snapshot(timestamp, "SEQUÊNCIA_SUSPEITA_EXPIRADA", indicators);
        }

        indicators.lowPosition = lowPosition;
        indicators.persistentLowPosition = Boolean(sequence.lowStartedAt &&
            timestamp - sequence.lowStartedAt >= config.minimumLowPersistenceMs &&
            sequence.lowSamples >= config.minimumLowObservationSamples);

        if (indicators.persistentLowPosition) {
            state = FALL_DETECTION_STATES.POSSIVEL_QUEDA_NAO_CONFIRMADA;
            previousSample = sample;
            return snapshot(timestamp, "MÚLTIPLOS_SINAIS_SUSPEITOS_PERSISTENTES", indicators);
        }

        if (postureTransition) {
            state = FALL_DETECTION_STATES.MUDANCA_DE_POSTURA;
        } else if (state === FALL_DETECTION_STATES.DESCIDA_SUSPEITA) {
            state = FALL_DETECTION_STATES.OBSERVACAO_POS_MOVIMENTO;
            sequence.observationStartedAt = timestamp;
        } else if (state === FALL_DETECTION_STATES.MUDANCA_DE_POSTURA) {
            state = FALL_DETECTION_STATES.OBSERVACAO_POS_MOVIMENTO;
            sequence.observationStartedAt ??= timestamp;
        } else {
            state = FALL_DETECTION_STATES.OBSERVACAO_POS_MOVIMENTO;
        }

        previousSample = sample;
        return snapshot(timestamp, lowPosition ? "OBSERVAÇÃO_DE_POSIÇÃO_BAIXA" : "OBSERVAÇÃO_APÓS_MOVIMENTO", indicators);
    }

    function interrupt(reason = "CONTINUIDADE_INTERROMPIDA", timestamp = null) {
        breakContinuity();
        state = FALL_DETECTION_STATES.ANALISE_INCONCLUSIVA;
        return snapshot(timestamp, reason);
    }

    function reset() {
        breakContinuity();
        state = FALL_DETECTION_STATES.ANALISE_INCONCLUSIVA;
        return snapshot(null, "SESSÃO_REINICIADA");
    }

    return { analyze, interrupt, reset, getState: () => state };
}

export function installFallDetection(app) {
    const diagnosticSampleIntervalMs = 1000;
    let diagnosticsEnabled = false;
    let latestDiagnosticSample = null;
    let lastLoggedState = FALL_DETECTION_STATES.ANALISE_INCONCLUSIVA;
    let lastSampleLogAt = Number.NEGATIVE_INFINITY;
    let lastInterruptionKey = null;
    let lastInterruptionLogAt = Number.NEGATIVE_INFINITY;

    const interruptionReasons = new Set([
        "TIMESTAMP_INVALIDO",
        "PESSOA_AUSENTE",
        "VISIBILIDADE_OU_GEOMETRIA_INSUFICIENTE",
        "INTERVALO_DE_INFERÊNCIA_IRREGULAR",
        "CONTINUIDADE_INTERROMPIDA",
        "FALHA_DE_INFERÊNCIA",
        "SESSÃO_REINICIADA",
        "SEQUÊNCIA_SUSPEITA_EXPIRADA"
    ]);

    const isFinite = Number.isFinite;
    const formatTimestamp = timestamp => new Date(isFinite(timestamp) ? timestamp : Date.now()).toISOString();

    function summarizeDiagnostic(sample, result) {
        const reason = result?.reason ?? null;
        const interrupted = !sample || interruptionReasons.has(reason);
        let continuity = "SEM_REFERÊNCIA";
        if (result?.indicators?.continuousInference) continuity = "CONTÍNUA";
        else if (reason === "REFERÊNCIA_CORPORAL_ESTABELECIDA") continuity = "REFERÊNCIA_ESTABELECIDA";
        else if (interrupted) continuity = "INTERROMPIDA";

        return {
            timestamp: formatTimestamp(sample?.timestamp ?? result?.timestamp),
            state: result?.state ?? null,
            posture: sample?.posture ?? null,
            verticalSpeedNormalizedPerSecond: isFinite(sample?.displacement?.normalizedYPerSecond)
                ? sample.displacement.normalizedYPerSecond : null,
            verticalDisplacementNormalized: isFinite(sample?.displacement?.deltaYNormalized)
                ? sample.displacement.deltaYNormalized : null,
            trunkAngleDegrees: isFinite(sample?.trunkAngleDegrees) ? sample.trunkAngleDegrees : null,
            observedJointCount: isFinite(sample?.boundingBoxPointsUsed) ? sample.boundingBoxPointsUsed : null,
            continuity,
            reason
        };
    }

    function reportDiagnostics(result, sample = null) {
        const currentState = result?.state ?? FALL_DETECTION_STATES.ANALISE_INCONCLUSIVA;
        const previousState = lastLoggedState;
        const stateChanged = currentState !== previousState;
        const diagnostic = summarizeDiagnostic(sample, result);
        const timestampMs = isFinite(sample?.timestamp) ? sample.timestamp
            : isFinite(result?.timestamp) ? result.timestamp : Date.now();
        lastLoggedState = currentState;
        if (sample) latestDiagnosticSample = diagnostic;
        if (!diagnosticsEnabled) return;

        if (stateChanged) {
            console.info("[VidaSegura IA][Diagnóstico] Transição de estado", {
                previousState,
                ...diagnostic
            });
            lastSampleLogAt = timestampMs;
            lastInterruptionKey = interruptionReasons.has(result?.reason)
                ? `${currentState}:${result.reason}` : null;
            lastInterruptionLogAt = lastSampleLogAt;
        } else if (interruptionReasons.has(result?.reason)) {
            const interruptionKey = `${currentState}:${result.reason}`;
            if (interruptionKey !== lastInterruptionKey ||
                timestampMs - lastInterruptionLogAt >= diagnosticSampleIntervalMs) {
                console.info("[VidaSegura IA][Diagnóstico] Interrupção", {
                    previousState,
                    ...diagnostic
                });
                lastInterruptionKey = interruptionKey;
                lastInterruptionLogAt = timestampMs;
                lastSampleLogAt = timestampMs;
            }
        } else if (sample && timestampMs - lastSampleLogAt >= diagnosticSampleIntervalMs) {
            console.info("[VidaSegura IA][Diagnóstico] Amostra", diagnostic);
            lastSampleLogAt = timestampMs;
        }
    }

    const detector = createFallDetector({
        onExpire: result => {
            app.latestFallDetectionAnalysis = result;
            reportDiagnostics(result);
        }
    });

    const diagnosticsApi = Object.freeze({
        enable() {
            diagnosticsEnabled = true;
            lastLoggedState = detector.getState();
            lastSampleLogAt = Number.NEGATIVE_INFINITY;
            lastInterruptionKey = null;
        },
        disable() {
            diagnosticsEnabled = false;
        },
        isEnabled() {
            return diagnosticsEnabled;
        },
        getLatestSample() {
            return latestDiagnosticSample ? { ...latestDiagnosticSample } : null;
        }
    });

    app.latestFallDetectionAnalysis = null;
    app.evaluateFallDetection = sample => {
        const result = detector.analyze(sample);
        app.latestFallDetectionAnalysis = result;
        reportDiagnostics(result, sample);
        return result;
    };
    app.interruptFallDetection = (reason, timestamp) => {
        const result = detector.interrupt(reason, timestamp);
        app.latestFallDetectionAnalysis = result;
        reportDiagnostics(result);
        return result;
    };
    app.resetFallDetection = () => {
        const result = detector.reset();
        app.latestFallDetectionAnalysis = result;
        reportDiagnostics(result);
        return result;
    };

    if (typeof window !== "undefined") window.vidaSeguraDiagnostics = diagnosticsApi;
}
