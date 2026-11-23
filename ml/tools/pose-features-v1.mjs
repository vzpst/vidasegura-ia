// Versão compartilhável com o navegador da transformação implementada em
// ml/scripts/common.py. A entrada deve ter passado pela validação do esquema v1.
export const FEATURE_TRANSFORM_VERSION = "pose-features-v1";
export const VISIBILITY_CONFIDENCE_THRESHOLD = 0.25;

export const KEYPOINT_NAMES = Object.freeze([
    "left_shoulder", "right_shoulder", "left_hip", "right_hip",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]);

export const GEOMETRY_PAIRS = Object.freeze([
    ["left_shoulder", "right_shoulder"],
    ["left_hip", "right_hip"],
    ["left_shoulder", "left_hip"],
    ["right_shoulder", "right_hip"],
    ["left_hip", "left_knee"],
    ["right_hip", "right_knee"],
    ["left_knee", "left_ankle"],
    ["right_knee", "right_ankle"],
    ["left_shoulder", "left_elbow"],
    ["right_shoulder", "right_elbow"],
    ["left_elbow", "left_wrist"],
    ["right_elbow", "right_wrist"],
    ["left_shoulder", "right_hip"],
    ["right_shoulder", "left_hip"]
].map(pair => Object.freeze(pair)));

function finiteNumber(value) {
    return typeof value === "number" && Number.isFinite(value);
}

function pointIsObserved(point) {
    return Boolean(point && point.validity === "observed" &&
        finiteNumber(point.x) && finiteNumber(point.y) && finiteNumber(point.score) &&
        point.score >= VISIBILITY_CONFIDENCE_THRESHOLD);
}

function safeConfidence(point) {
    return point && finiteNumber(point.score) && point.score >= 0 && point.score <= 1
        ? point.score
        : 0;
}

export function featureOrder() {
    const names = [];
    for (const name of KEYPOINT_NAMES) {
        names.push(
            `${name}.x_centered_scaled`, `${name}.y_centered_scaled`,
            `${name}.confidence`, `${name}.mask`
        );
    }
    for (const [first, second] of GEOMETRY_PAIRS) {
        const pair = `${first}__${second}`;
        names.push(
            `${pair}.dx_scaled`, `${pair}.dy_scaled`, `${pair}.distance_scaled`,
            `${pair}.angle_from_vertical_norm`, `${pair}.mask`
        );
    }
    return names;
}

export function extractPoseFeatures(sample) {
    const keypoints = sample?.keypoints && typeof sample.keypoints === "object"
        ? sample.keypoints
        : {};
    const observed = new Map(KEYPOINT_NAMES
        .filter(name => pointIsObserved(keypoints[name]))
        .map(name => [name, keypoints[name]]));

    let centerPoints = ["left_hip", "right_hip"]
        .filter(name => observed.has(name)).map(name => observed.get(name));
    if (centerPoints.length === 0) {
        centerPoints = ["left_shoulder", "right_shoulder"]
            .filter(name => observed.has(name)).map(name => observed.get(name));
    }
    if (centerPoints.length === 0) centerPoints = [...observed.values()];

    let centerX = 0.5;
    let centerY = 0.5;
    let scale = 1.0;
    if (centerPoints.length > 0) {
        centerX = centerPoints.reduce((sum, point) => sum + point.x, 0) / centerPoints.length;
        centerY = centerPoints.reduce((sum, point) => sum + point.y, 0) / centerPoints.length;
        const points = [...observed.values()];
        const xs = points.map(point => point.x);
        const ys = points.map(point => point.y);
        scale = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys), 1e-6);
    }

    const vector = [];
    for (const name of KEYPOINT_NAMES) {
        const point = observed.get(name);
        if (!point) {
            vector.push(0, 0, safeConfidence(keypoints[name]), 0);
        } else {
            vector.push((point.x - centerX) / scale, (point.y - centerY) / scale, point.score, 1);
        }
    }

    for (const [firstName, secondName] of GEOMETRY_PAIRS) {
        const first = observed.get(firstName);
        const second = observed.get(secondName);
        if (!first || !second) {
            vector.push(0, 0, 0, 0, 0);
            continue;
        }
        const dx = second.x - first.x;
        const dy = second.y - first.y;
        const length = Math.hypot(dx, dy);
        // Convenção igual ao Python: 0 significa vertical e 1 horizontal.
        const angle = Math.atan2(Math.abs(dx), Math.abs(dy)) / (Math.PI / 2);
        vector.push(dx / scale, dy / scale, length / scale, angle, 1);
    }

    if (vector.length !== featureOrder().length || vector.some(value => !finiteNumber(value))) {
        throw new Error("Transformação pose-features-v1 gerou um vetor incompatível ou não finito.");
    }
    return vector;
}

export function featureMetadata() {
    return {
        version: FEATURE_TRANSFORM_VERSION,
        schema_version: 1,
        keypoint_order: [...KEYPOINT_NAMES],
        geometry_pair_order: GEOMETRY_PAIRS.map(pair => [...pair]),
        features_per_keypoint: ["x_centered_scaled", "y_centered_scaled", "confidence", "mask"],
        features_per_pair: ["dx_scaled", "dy_scaled", "distance_scaled", "angle_from_vertical_norm", "mask"],
        angle_convention: "0 vertical, 90 degrees horizontal; stored normalized to [0, 1]",
        confidence_threshold: VISIBILITY_CONFIDENCE_THRESHOLD,
        feature_order: featureOrder(),
        feature_count: featureOrder().length,
        missing_coordinate_policy: "zero neutral values with mask=0; no coordinate is estimated",
        center_policy: "mean of observed hips, otherwise observed shoulders, otherwise observed points",
        scale_policy: "maximum observed x/y extent, lower bounded by 1e-6"
    };
}
