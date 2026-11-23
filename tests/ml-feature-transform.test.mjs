import test from "node:test";
import assert from "node:assert/strict";
import {
    FEATURE_TRANSFORM_VERSION,
    KEYPOINT_NAMES,
    extractPoseFeatures,
    featureMetadata,
    featureOrder
} from "../ml/tools/pose-features-v1.mjs";

function cameraSample() {
    const points = {
        left_shoulder: [0.42, 0.20], right_shoulder: [0.58, 0.20],
        left_hip: [0.45, 0.50], right_hip: [0.55, 0.50],
        left_elbow: [0.38, 0.32], right_elbow: [0.62, 0.32],
        left_wrist: [0.36, 0.45], right_wrist: [0.64, 0.45],
        left_knee: [0.45, 0.70], right_knee: [0.55, 0.70],
        left_ankle: [0.45, 0.90], right_ankle: [0.55, 0.90]
    };
    return {
        keypoints: Object.fromEntries(Object.entries(points).map(([name, [x, y]]) => [
            name, { x, y, score: 0.9, validity: "observed" }
        ]))
    };
}

test("pose-features-v1 matches the Python vector order and geometric values", () => {
    const vector = extractPoseFeatures(cameraSample());
    assert.equal(FEATURE_TRANSFORM_VERSION, "pose-features-v1");
    assert.equal(vector.length, 12 * 4 + 14 * 5);
    assert.equal(featureOrder().length, vector.length);
    assert.deepEqual(featureOrder().slice(0, 4), [
        "left_shoulder.x_centered_scaled", "left_shoulder.y_centered_scaled",
        "left_shoulder.confidence", "left_shoulder.mask"
    ]);
    assert.equal(featureOrder()[48], "left_shoulder__right_shoulder.dx_scaled");
    assert.ok(Math.abs(vector[0] - (-0.08 / 0.7)) < 1e-12);
    assert.ok(Math.abs(vector[1] - (-0.30 / 0.7)) < 1e-12);
    assert.ok(Math.abs(vector[48] - (0.16 / 0.7)) < 1e-12);
    assert.equal(vector[51], 1); // Ângulo de ombros horizontal: normalizado a 1.
    assert.equal(vector[52], 1);
});

test("pontos ausentes preservam somente a confiança e recebem máscara zero", () => {
    const sample = cameraSample();
    sample.keypoints.left_wrist = { x: null, y: null, score: 0.7, validity: "missing" };
    const vector = extractPoseFeatures(sample);
    const start = KEYPOINT_NAMES.indexOf("left_wrist") * 4;
    assert.deepEqual(vector.slice(start, start + 4), [0, 0, 0.7, 0]);
    assert.deepEqual(vector.slice(48 + 10 * 5, 48 + 11 * 5), [0, 0, 0, 0, 0]);
});

test("pontos de baixa confiança não contribuem coordenadas nem geometria", () => {
    const sample = cameraSample();
    sample.keypoints.left_knee = { x: 0.1, y: 0.1, score: 0.1, validity: "low_confidence" };
    const vector = extractPoseFeatures(sample);
    const start = KEYPOINT_NAMES.indexOf("left_knee") * 4;
    assert.deepEqual(vector.slice(start, start + 4), [0, 0, 0.1, 0]);
    assert.deepEqual(vector.slice(48 + 6 * 5, 48 + 7 * 5), [0, 0, 0, 0, 0]);
});

test("metadados versionam 118 características e políticas de máscara", () => {
    const metadata = featureMetadata();
    assert.equal(metadata.version, FEATURE_TRANSFORM_VERSION);
    assert.equal(metadata.feature_count, 118);
    assert.equal(metadata.confidence_threshold, 0.25);
    assert.match(metadata.missing_coordinate_policy, /mask=0/);
    assert.match(metadata.angle_convention, /0 vertical/);
});
