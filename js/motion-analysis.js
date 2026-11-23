const BODY_JOINTS = [
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
];
const OBSERVED_THRESHOLD = 0.25;
const MAX_SAMPLES = 300;
const RETENTION_MS = 60_000;

const finiteCoordinate = value => Number.isFinite(value) ? value : null;
const validDimension = value => Number.isFinite(value) && value > 0;

function makeCenter(points, width, height) {
    if (points.length !== 2 || points.some(point => !point.observed)) return null;
    const x = (points[0].x + points[1].x) / 2;
    const y = (points[0].y + points[1].y) / 2;
    return {
        x,
        y,
        normalizedX: validDimension(width) ? x / width : null,
        normalizedY: validDimension(height) ? y / height : null
    };
}

function emptyJoints() {
    return BODY_JOINTS.map(name => ({
        name, x: null, y: null, normalizedX: null, normalizedY: null,
        confidence: null, observed: false, availability: "INDISPONÍVEL"
    }));
}

export function createMotionAnalyzer({ now = () => Date.now() } = {}) {
    const samples = [];
    let previousMovementReference = null;
    let previousSampleTimestamp = null;
    let continuityId = 0;

    function interruptContinuity() {
        previousMovementReference = null;
        previousSampleTimestamp = null;
        continuityId++;
    }

    function resetContinuity() {
        interruptContinuity();
    }

    function recordInference({ pose = null, analysisResult, videoWidth, videoHeight, timestamp } = {}) {
        const sampledAt = Number.isFinite(timestamp) ? timestamp : analysisResult?.timestamp;
        if (!Number.isFinite(sampledAt)) {
            interruptContinuity();
            return null;
        }

        const personDetected = Boolean(pose);
        const rawKeypoints = analysisResult?.keypoints ?? analysisResult?.visibility?.pointObservations ?? [];
        const observationsByName = new Map(rawKeypoints
            .filter(point => point && typeof point.name === "string")
            .map(point => [point.name, point]));
        const joints = personDetected
            ? BODY_JOINTS.map(name => {
                const point = observationsByName.get(name);
                const x = finiteCoordinate(point?.x);
                const y = finiteCoordinate(point?.y);
                const confidence = finiteCoordinate(point?.score);
                const hasCoordinates = x !== null && y !== null;
                const observed = hasCoordinates && confidence !== null && confidence >= OBSERVED_THRESHOLD &&
                    point?.availability !== "BAIXA_CONFIANÇA" && point?.availability !== "INDISPONÍVEL";
                return {
                    name,
                    x,
                    y,
                    normalizedX: hasCoordinates && validDimension(videoWidth) ? x / videoWidth : null,
                    normalizedY: hasCoordinates && validDimension(videoHeight) ? y / videoHeight : null,
                    confidence,
                    observed,
                    availability: observed ? "OBSERVADO" : hasCoordinates ? "BAIXA_CONFIANÇA" : "INDISPONÍVEL"
                };
            })
            : emptyJoints();
        const observedJoints = joints.filter(point => point.observed);
        const shoulders = makeCenter([
            joints.find(point => point.name === "left_shoulder"),
            joints.find(point => point.name === "right_shoulder")
        ], videoWidth, videoHeight);
        const hips = makeCenter([
            joints.find(point => point.name === "left_hip"),
            joints.find(point => point.name === "right_hip")
        ], videoWidth, videoHeight);
        const bodyCenter = shoulders && hips ? {
            x: (shoulders.x + hips.x) / 2,
            y: (shoulders.y + hips.y) / 2,
            normalizedX: validDimension(videoWidth) ? (shoulders.x + hips.x) / (2 * videoWidth) : null,
            normalizedY: validDimension(videoHeight) ? (shoulders.y + hips.y) / (2 * videoHeight) : null
        } : null;

        // In image coordinates Y increases downwards. atan2(dx, dy) therefore
        // gives 0° for a vertical torso and a signed lean relative to image vertical.
        const trunkAngleDegrees = shoulders && hips
            ? Math.atan2(hips.x - shoulders.x, hips.y - shoulders.y) * 180 / Math.PI
            : null;

        let boundingBox = null;
        if (observedJoints.length) {
            const xs = observedJoints.map(point => point.x);
            const ys = observedJoints.map(point => point.y);
            const minX = Math.min(...xs);
            const maxX = Math.max(...xs);
            const minY = Math.min(...ys);
            const maxY = Math.max(...ys);
            boundingBox = {
                minX, minY, maxX, maxY,
                normalizedMinX: validDimension(videoWidth) ? minX / videoWidth : null,
                normalizedMinY: validDimension(videoHeight) ? minY / videoHeight : null,
                normalizedMaxX: validDimension(videoWidth) ? maxX / videoWidth : null,
                normalizedMaxY: validDimension(videoHeight) ? maxY / videoHeight : null,
                widthNormalized: validDimension(videoWidth) ? (maxX - minX) / videoWidth : null,
                heightNormalized: validDimension(videoHeight) ? (maxY - minY) / videoHeight : null,
                pointsUsed: observedJoints.length,
                isPartial: observedJoints.length < BODY_JOINTS.length,
                representsWholeBody: observedJoints.length === BODY_JOINTS.length
            };
        }

        const intervalMs = previousSampleTimestamp !== null ? sampledAt - previousSampleTimestamp : null;
        const intervalValid = Number.isFinite(intervalMs) && intervalMs > 0;
        const canCompare = personDetected && bodyCenter && previousMovementReference && intervalValid;
        const deltaX = canCompare && bodyCenter.normalizedX !== null &&
            previousMovementReference.normalizedX !== null
            ? bodyCenter.normalizedX - previousMovementReference.normalizedX : null;
        const deltaY = canCompare && bodyCenter.normalizedY !== null &&
            previousMovementReference.normalizedY !== null
            ? bodyCenter.normalizedY - previousMovementReference.normalizedY : null;
        const displacement = {
            intervalMs: Number.isFinite(intervalMs) ? intervalMs : null,
            deltaXNormalized: deltaX,
            deltaYNormalized: deltaY,
            normalizedXPerSecond: deltaX !== null ? deltaX / (intervalMs / 1000) : null,
            normalizedYPerSecond: deltaY !== null ? deltaY / (intervalMs / 1000) : null
        };

        const regionCounts = analysisResult?.visibility?.regionCounts ?? {};
        const sample = {
            timestamp: sampledAt,
            continuityId,
            personDetection: {
                detected: personDetected,
                state: analysisResult?.personDetection?.state ??
                    (personDetected ? "PESSOA DETECTADA" : "PESSOA NÃO DETECTADA")
            },
            posture: analysisResult?.posture?.state ?? "INCONCLUSIVA",
            visibility: analysisResult?.visibility?.state ?? "NÃO AVALIÁVEL",
            observedJointsByRegion: {
                head: Number.isFinite(regionCounts.head) ? regionCounts.head : 0,
                torso: Number.isFinite(regionCounts.torso) ? regionCounts.torso : 0,
                arms: Number.isFinite(regionCounts.arms) ? regionCounts.arms : 0,
                legs: Number.isFinite(regionCounts.legs) ? regionCounts.legs : 0
            },
            joints,
            centers: { shoulders, hips, body: bodyCenter },
            trunkAngleDegrees,
            boundingBox,
            boundingBoxPointsUsed: observedJoints.length,
            displacement
        };

        samples.push(sample);
        const cutoff = now() - RETENTION_MS;
        for (let index = samples.length - 1; index >= 0; index--) {
            if (samples[index].timestamp < cutoff) samples.splice(index, 1);
        }
        if (samples.length > MAX_SAMPLES) samples.splice(0, samples.length - MAX_SAMPLES);

        if (personDetected && bodyCenter) {
            previousMovementReference = bodyCenter;
        } else {
            previousMovementReference = null;
            continuityId++;
        }
        previousSampleTimestamp = sampledAt;
        return sample;
    }

    return {
        recordInference,
        interruptContinuity,
        resetContinuity,
        getHistory: () => samples.map(sample => structuredClone(sample))
    };
}

export function installMotionAnalysis(app) {
    const analyzer = createMotionAnalyzer();
    app.recordMotionInference = input => analyzer.recordInference(input);
    app.interruptMotionContinuity = () => analyzer.interruptContinuity();
    app.resetMotionContinuity = () => analyzer.resetContinuity();
    app.getMotionAnalysisHistory = () => analyzer.getHistory();
}
