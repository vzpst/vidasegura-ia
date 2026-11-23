// Limiares geométricos experimentais. Ajustar somente após testes com câmeras e
// dados representativos; estes valores não são validação científica.
export const POSTURE_CLASSIFICATION_CONFIG = Object.freeze({
    provisional: true,
    minKeypointConfidence: 0.35,
    minTorsoLengthPx: 25,
    minLegSegmentLengthPx: 18,
    minVerticalProgressPx: 12,
    maxCompatibleTorsoAngleDifferenceDegrees: 28,
    minSideConfidenceRatioForSelection: 0.7,
    minSideQualityRatioToResolveConflict: 1.4,
    overlapSeparationRatio: 0.3,
    maxStandingTorsoAngleDegrees: 30,
    maxStandingLegSegmentAngleDegrees: 35,
    minStandingKneeExtensionDegrees: 145,
    maxSittingTorsoAngleDegrees: 45,
    minSittingKneeFlexionDegrees: 65,
    maxSittingKneeFlexionDegrees: 135,
    minSittingThighAngleDegrees: 35,
    maxSittingShinAngleDegrees: 55,
    maxSittingKneeBelowHipRatio: 0.45,
    minLyingTorsoAngleDegrees: 65,
    minLyingLegSegmentAngleDegrees: 50,
    minLyingKneeExtensionDegrees: 120
});

const KEYPOINT_INDEX = Object.freeze({
    left_shoulder: 5,
    right_shoulder: 6,
    left_hip: 11,
    right_hip: 12,
    left_knee: 13,
    right_knee: 14,
    left_ankle: 15,
    right_ankle: 16
});

function getKeypoint(keypoints, name) {
    const named = keypoints.find(point => point.name === name);
    if (named) return named;
    const positional = keypoints[KEYPOINT_INDEX[name]];
    return positional && positional.name == null ? positional : null;
}

function isReliable(point) {
    return Boolean(point && Number.isFinite(point.x) && Number.isFinite(point.y) &&
        Number.isFinite(point.score) && point.score >= POSTURE_CLASSIFICATION_CONFIG.minKeypointConfidence);
}

function distance(a, b) {
    return Math.hypot(b.x - a.x, b.y - a.y);
}

// Convenção: 0° é vertical na imagem e 90° é horizontal. O ângulo é sem
// direção (0–90°), então subir/descer ou inverter a ordem dos pontos não muda
// a orientação estimada.
function angleFromVertical(a, b) {
    return Math.atan2(Math.abs(b.x - a.x), Math.abs(b.y - a.y)) * 180 / Math.PI;
}

function angleDifference(a, b) {
    return Math.abs(a - b);
}

function pointConfidence(...points) {
    return Math.min(...points.map(point => point.score));
}

function createTorsoCandidate(side, shoulder, hip) {
    if (!isReliable(shoulder) || !isReliable(hip)) return null;
    const length = distance(shoulder, hip);
    if (length < POSTURE_CLASSIFICATION_CONFIG.minTorsoLengthPx) return null;
    const confidence = pointConfidence(shoulder, hip);
    return {
        side,
        shoulder,
        hip,
        length,
        confidence,
        quality: confidence * Math.sqrt(length),
        angle: angleFromVertical(shoulder, hip)
    };
}

function selectTorsoOrientation(left, right) {
    if (!left) return right ? { angle: right.angle, hip: right.hip, sides: [right.side] } : null;
    if (!right) return { angle: left.angle, hip: left.hip, sides: [left.side] };

    const stronger = left.quality >= right.quality ? left : right;
    const weaker = stronger === left ? right : left;
    const confidenceRatio = weaker.confidence / stronger.confidence;
    const qualityRatio = stronger.quality / Math.max(weaker.quality, Number.EPSILON);
    const averageTorsoLength = (left.length + right.length) / 2;
    const overlapping = distance(left.shoulder, right.shoulder) <
            averageTorsoLength * POSTURE_CLASSIFICATION_CONFIG.overlapSeparationRatio ||
        distance(left.hip, right.hip) < averageTorsoLength * POSTURE_CLASSIFICATION_CONFIG.overlapSeparationRatio;

    // Perfil e sobreposição: escolha a observação individual mais forte; não
    // misture coordenadas de lados que podem representar o mesmo ponto visível.
    if (overlapping || confidenceRatio < POSTURE_CLASSIFICATION_CONFIG.minSideConfidenceRatioForSelection) {
        return { angle: stronger.angle, hip: stronger.hip, sides: [stronger.side] };
    }

    const difference = angleDifference(left.angle, right.angle);
    if (difference <= POSTURE_CLASSIFICATION_CONFIG.maxCompatibleTorsoAngleDifferenceDegrees) {
        const totalWeight = left.quality + right.quality;
        return {
            angle: (left.angle * left.quality + right.angle * right.quality) / totalWeight,
            hip: stronger.hip,
            sides: [left.side, right.side]
        };
    }

    if (qualityRatio >= POSTURE_CLASSIFICATION_CONFIG.minSideQualityRatioToResolveConflict) {
        return { angle: stronger.angle, hip: stronger.hip, sides: [stronger.side] };
    }
    return null;
}

function kneeFlexionDegrees(hip, knee, ankle) {
    const first = { x: hip.x - knee.x, y: hip.y - knee.y };
    const second = { x: ankle.x - knee.x, y: ankle.y - knee.y };
    const firstLength = Math.hypot(first.x, first.y);
    const secondLength = Math.hypot(second.x, second.y);
    if (!firstLength || !secondLength) return null;
    const cosine = Math.max(-1, Math.min(1,
        (first.x * second.x + first.y * second.y) / (firstLength * secondLength)));
    return Math.acos(cosine) * 180 / Math.PI;
}

function createLegChain(side, hip, knee, ankle) {
    if (![hip, knee, ankle].every(isReliable)) return null;
    const thighLength = distance(hip, knee);
    const shinLength = distance(knee, ankle);
    if (thighLength < POSTURE_CLASSIFICATION_CONFIG.minLegSegmentLengthPx ||
        shinLength < POSTURE_CLASSIFICATION_CONFIG.minLegSegmentLengthPx) return null;
    const confidence = pointConfidence(hip, knee, ankle);
    return {
        side,
        hip,
        knee,
        ankle,
        thighLength,
        shinLength,
        legLength: thighLength + shinLength,
        thighAngle: angleFromVertical(hip, knee),
        shinAngle: angleFromVertical(knee, ankle),
        kneeFlexion: kneeFlexionDegrees(hip, knee, ankle),
        confidence,
        quality: confidence * Math.sqrt(thighLength + shinLength)
    };
}

function classifyLegChain(chain, torsoAngle) {
    const config = POSTURE_CLASSIFICATION_CONFIG;
    const ankleProgress = chain.ankle.y - chain.knee.y;

    if (torsoAngle >= config.minLyingTorsoAngleDegrees &&
        chain.kneeFlexion >= config.minLyingKneeExtensionDegrees &&
        (chain.thighAngle >= config.minLyingLegSegmentAngleDegrees ||
            chain.shinAngle >= config.minLyingLegSegmentAngleDegrees)) {
        return "DEITADA";
    }

    if (torsoAngle <= config.maxStandingTorsoAngleDegrees &&
        chain.kneeFlexion >= config.minStandingKneeExtensionDegrees &&
        chain.thighAngle <= config.maxStandingLegSegmentAngleDegrees &&
        chain.shinAngle <= config.maxStandingLegSegmentAngleDegrees &&
        chain.knee.y > chain.hip.y &&
        ankleProgress >= config.minVerticalProgressPx) {
        return "EM PÉ";
    }

    const kneeBelowHip = chain.knee.y - chain.hip.y;
    if (torsoAngle <= config.maxSittingTorsoAngleDegrees &&
        chain.kneeFlexion >= config.minSittingKneeFlexionDegrees &&
        chain.kneeFlexion <= config.maxSittingKneeFlexionDegrees &&
        chain.thighAngle >= config.minSittingThighAngleDegrees &&
        chain.shinAngle <= config.maxSittingShinAngleDegrees &&
        ankleProgress >= config.minVerticalProgressPx &&
        kneeBelowHip <= chain.legLength * config.maxSittingKneeBelowHipRatio) {
        return "SENTADA";
    }

    return null;
}

function selectLegClassification(chains, torsoAngle) {
    if (!chains.length) return null;
    const evaluated = chains.map(chain => ({ chain, posture: classifyLegChain(chain, torsoAngle) }));
    if (evaluated.length === 1) return evaluated[0].posture;
    if (evaluated[0].posture === evaluated[1].posture) return evaluated[0].posture;

    const [first, second] = evaluated;
    const stronger = first.chain.quality >= second.chain.quality ? first : second;
    const weaker = stronger === first ? second : first;
    if (stronger.chain.quality / Math.max(weaker.chain.quality, Number.EPSILON) >=
        POSTURE_CLASSIFICATION_CONFIG.minSideQualityRatioToResolveConflict) {
        return stronger.posture;
    }
    return null;
}

export function installPosture(app) {
    app.classifyPose = function(pose) {
        const keypoints = pose?.keypoints;
        if (!Array.isArray(keypoints) || !keypoints.length) return null;

        const point = Object.fromEntries(Object.keys(KEYPOINT_INDEX).map(name => [name, getKeypoint(keypoints, name)]));
        const leftTorso = createTorsoCandidate("left", point.left_shoulder, point.left_hip);
        const rightTorso = createTorsoCandidate("right", point.right_shoulder, point.right_hip);
        const torso = selectTorsoOrientation(leftTorso, rightTorso);
        if (!torso) return null;

        const chains = [
            createLegChain("left", point.left_hip, point.left_knee, point.left_ankle),
            createLegChain("right", point.right_hip, point.right_knee, point.right_ankle)
        ].filter(Boolean);
        if (!chains.length) return null;
        return selectLegClassification(chains, torso.angle);
    };

    app.updatePostureHistory = function(posture) {
        const windowSize = 7;
        const votesRequired = 5;
        app.postureHistory.push(posture);
        if (app.postureHistory.length > windowSize) app.postureHistory.shift();

        const votes = new Map();
        app.postureHistory.filter(Boolean).forEach(value => votes.set(value, (votes.get(value) ?? 0) + 1));
        const winner = [...votes.entries()].sort((a, b) => b[1] - a[1])[0];
        return winner && winner[1] >= votesRequired ? winner[0] : null;
    };
}
