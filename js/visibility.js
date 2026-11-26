import { fusePosture } from "./posture-model.js";

// Funções extraídas do fluxo original; o estado e as dependências são injetados por main.js.
export function installVisibility(app) {
    app.evaluateBodyVisibility = function(pose) {
    const keypoints = pose?.keypoints ?? [];
    const facialNames = ["nose", "left_eye", "right_eye", "left_ear", "right_ear"];
    const bodyRegions = {
        torso: ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
        arms: ["left_elbow", "right_elbow", "left_wrist", "right_wrist"],
        legs: ["left_knee", "right_knee", "left_ankle", "right_ankle"]
    };
    const bodyNames = [...bodyRegions.torso, ...bodyRegions.arms, ...bodyRegions.legs];
    const names = [...facialNames, ...bodyNames];
    const threshold = 0.25;
    const pointObservations = names.map(name => {
        // O nome é a única associação segura; o índice varia conforme o modelo
        // e não pode ser usado para inventar uma articulação ausente.
        const point = keypoints.find(item => item.name === name) ?? null;
        const hasCoordinates = Boolean(point && Number.isFinite(point.x) && Number.isFinite(point.y));
        const score = point?.score;
        const observed = hasCoordinates && Number.isFinite(score) && score >= threshold;
        return {
            name,
            x: hasCoordinates ? point.x : null,
            y: hasCoordinates ? point.y : null,
            score: Number.isFinite(score) ? score : null,
            availability: !hasCoordinates ? "INDISPONÍVEL" : observed ? "OBSERVADO" : "BAIXA_CONFIANÇA"
        };
    });
    const countObserved = group => group.filter(name =>
        pointObservations.find(point => point.name === name)?.availability === "OBSERVADO").length;
    const regionCounts = {
        head: countObserved(facialNames),
        torso: countObserved(bodyRegions.torso),
        arms: countObserved(bodyRegions.arms),
        legs: countObserved(bodyRegions.legs)
    };
    const observedCount = countObserved(bodyNames);
    const lowConfidencePoints = pointObservations.filter(point => point.availability === "BAIXA_CONFIANÇA").map(point => point.name);
    const unavailablePoints = pointObservations.filter(point => point.availability === "INDISPONÍVEL").map(point => point.name);

    let state = !pose ? "NÃO AVALIÁVEL" : "VISIBILIDADE INSUFICIENTE";
    if (pose && observedCount >= 9 && regionCounts.torso >= 3 && regionCounts.arms >= 2 && regionCounts.legs >= 2) {
        state = "BOA VISIBILIDADE";
    } else if (pose && observedCount >= 5 && regionCounts.torso >= 2) {
        state = "VISIBILIDADE PARCIAL";
    }

    return {
        state,
        observedCount,
        totalPoints: bodyNames.length,
        facialObservedCount: regionCounts.head,
        facialTotalPoints: facialNames.length,
        regionCounts,
        lowConfidencePoints,
        unavailablePoints,
        pointObservations
    };

    };
    app.createAnalysisResult = function(pose, timestamp = Date.now()) {
    const visibility = app.evaluateBodyVisibility(pose);
    const geometricPosture = app.classifyPose(pose);
    const prediction = app.predictPostureModel?.(pose, app.camera?.videoWidth, app.camera?.videoHeight) ?? null;
    const fused = fusePosture(geometricPosture, prediction);
    const posture = fused.state;
    const posturePoints = posture === "DEITADA"
        ? ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
        : posture === "EM PÉ" || posture === "SENTADA"
            ? ["left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
            : [];
    return {
        timestamp,
        personDetection: {
            state: pose ? "PESSOA DETECTADA" : "PESSOA NÃO DETECTADA",
            detected: Boolean(pose)
        },
        posture: {
            state: posture ?? "INCONCLUSIVA",
            method: fused.method,
            geometric: geometricPosture ?? "INCONCLUSIVA",
            model: prediction
                ? { label: prediction.label, probability: prediction.probability }
                : null,
            supportingPoints: visibility.pointObservations
                .filter(point => posturePoints.includes(point.name) && point.availability === "OBSERVADO" && point.score >= 0.35)
                .map(point => point.name)
        },
        visibility,
        fallRisk: {
            state: "NÃO AVALIADO",
            evaluated: false,
            reason: "A análise temporal de possíveis quedas será implementada em outra etapa."
        },
        keypoints: visibility.pointObservations
    };

    };
}
