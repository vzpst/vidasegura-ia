import test from "node:test";
import assert from "node:assert/strict";

const { demoPose, DEMO_SCENARIOS } = await import(new URL("../js/fall-demo.js", import.meta.url));

test("pose simulada tem os 17 pontos do MoveNet com nomes", () => {
    const pose = demoPose(0, () => 0.5);
    assert.equal(pose.keypoints.length, 17);
    assert.equal(pose.keypoints[5].name, "left_shoulder");
    assert.ok(pose.keypoints.every(point => Number.isFinite(point.x) && point.score >= 0.8));
});

test("queda simulada leva o quadril para baixo e deitado", () => {
    const hip = k => demoPose(k, () => 0.5).keypoints.find(point => point.name === "left_hip");
    assert.ok(hip(1).y - hip(0).y > 150);
});

test("cenários: imóvel fica no chão, levanta volta a 0, some devolve null", () => {
    assert.equal(DEMO_SCENARIOS.imovel.position(0), 0);
    assert.equal(DEMO_SCENARIOS.imovel.position(15_000), 1);
    assert.equal(DEMO_SCENARIOS.levanta.position(4_000), 1);
    assert.equal(DEMO_SCENARIOS.levanta.position(9_000), 0);
    assert.equal(DEMO_SCENARIOS.some.position(6_000), null);
    assert.equal(demoPose(null), null);
});
