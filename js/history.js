// Funções extraídas do fluxo original; o estado e as dependências são injetados por main.js.
export function installHistory(app) {
    const allowedPostures = new Set(["EM PÉ", "SENTADA", "DEITADA"]);
    const positionLimit = 300;

    // A linha do tempo normal contém somente a mudança estável e seu timestamp.
    app.recordPositionChange = function(posture, timestamp = Date.now()) {
        if (!allowedPostures.has(posture) || !Number.isFinite(timestamp)) return false;

        const latestPosition = app.positionHistory[app.positionHistory.length - 1];
        if (latestPosition?.posture === posture) return false;

        const position = { posture, timestamp };
        const insertAt = app.positionHistory.findIndex(item => item.timestamp > timestamp);
        if (insertAt === -1) {
            if (latestPosition?.posture === posture) return false;
            app.positionHistory.push(position);
        } else {
            const previous = app.positionHistory[insertAt - 1];
            const next = app.positionHistory[insertAt];
            if (previous?.posture === posture || next?.posture === posture) return false;
            app.positionHistory.splice(insertAt, 0, position);
        }
        app.positionHistory = app.positionHistory.slice(-positionLimit);
        app.renderPositionHistory(app.positionHistory);
        return true;
    };

    // Integração futura: anexa a captura exclusivamente a uma queda já registrada.
    // Não cria eventos e não é chamada enquanto não existir um detector real.
    app.attachFallBodyRecord = function(fallEvent, analysisResult = app.latestAnalysisResult) {
        if (!fallEvent || fallEvent.type !== "fall" || fallEvent.status !== "registered" ||
            fallEvent.id == null || !analysisResult) return null;

        const observations = analysisResult.keypoints ?? [];
        const joints = observations
            .filter(point => point && Number.isFinite(point.x) && Number.isFinite(point.y))
            .map(point => ({
                name: point.name,
                x: point.x,
                y: point.y,
                confidence: point.score
            }));
        const visibility = analysisResult.visibility;

        fallEvent.bodyRecord = {
            capturedAt: analysisResult.timestamp,
            joints,
            visibility: visibility ? {
                state: visibility.state,
                observedCount: visibility.observedCount,
                totalPoints: visibility.totalPoints,
                facialObservedCount: visibility.facialObservedCount,
                facialTotalPoints: visibility.facialTotalPoints,
                regionCounts: { ...visibility.regionCounts },
                lowConfidencePoints: [...visibility.lowConfidencePoints],
                unavailablePoints: [...visibility.unavailablePoints],
                pointObservations: visibility.pointObservations.map(point => ({ ...point }))
            } : null
        };
        return fallEvent.bodyRecord;
    };

    app.recordAnalysisResult = function(result) {
    app.analysisHistory.push(result);
    const cutoff = Date.now() - 60000;
    // Mantém no máximo 300 inferências e uma janela temporal de 60 s.
    // Como a inferência acompanha requestAnimationFrame, a janela efetiva
    // depende da frequência real observada; o timestamp preserva essa idade.
    app.analysisHistory = app.analysisHistory.filter(item => item.timestamp >= cutoff).slice(-300);
    app.latestAnalysisResult = result;
    app.renderAnalysisDimensions(result);

    };
    app.resetAnalysisHistory = function() {
    app.analysisHistory = [];
    app.latestAnalysisResult = null;
    const personState = document.getElementById("personState");
    if (personState) {
        personState.textContent = "Aguardando câmera";
        personState.dataset.state = "waiting";
    }

    };
}
