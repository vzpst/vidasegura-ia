import test from "node:test";
import assert from "node:assert/strict";

const { createActivityClassifier, createRoutineLog, activityLabel, mealLabel, formatMinutes } =
    await import(new URL("../js/activity.js", import.meta.url));
const { buildBody, POSES } = await import(new URL("../js/routine-sim.js", import.meta.url));

const at = (h, m = 0) => { const d = new Date(2026, 10, 26, h, m); return d; };
const keypoints = params => buildBody(params, () => 0.5).keypoints;

function feed(classifier, frames, posture, poseFor, start = 0, step = 33) {
    let result;
    for (let i = 0; i < frames; i++) {
        const t = start + i * step;
        result = classifier.update({ timestamp: t, posture, keypoints: keypoints(poseFor(t)), width: 640, height: 480 });
    }
    return result;
}

test("rótulos dependem do horário: café, almoço, jantar; dormindo à noite, cochilando de dia", () => {
    assert.equal(mealLabel(at(7, 30)), "Café da manhã");
    assert.equal(mealLabel(at(12)), "Almoço");
    assert.equal(mealLabel(at(19)), "Jantar");
    assert.equal(activityLabel("dormindo", at(23)), "Dormindo");
    assert.equal(activityLabel("dormindo", at(14)), "Cochilando");
});

test("deitado e imóvel vira 'dormindo'; deitado se mexendo fica 'deitado'", () => {
    const still = createActivityClassifier();
    const asleep = feed(still, 300, "DEITADA", () => POSES.lying(300));
    assert.equal(asleep.id, "dormindo");
    const moving = createActivityClassifier();
    const awake = feed(moving, 300, "DEITADA", t => POSES.lying(300 + Math.sin(t / 300) * 25));
    assert.equal(awake.id, "deitado");
});

test("sentado ereto, mãos no colo e parado vira 'assistindo TV'", () => {
    const classifier = createActivityClassifier();
    assert.equal(feed(classifier, 250, "SENTADA", () => POSES.seated(230)).id, "tv");
});

test("mãos erguidas à frente do peito viram 'lendo'; tronco reclinado parado vira 'descanso'", () => {
    assert.equal(feed(createActivityClassifier(), 150, "SENTADA", () => POSES.reading(230)).id, "leitura");
    assert.equal(feed(createActivityClassifier(), 200, "SENTADA", () => POSES.reclined(230)).id, "descanso");
});

test("em pé deslocando-se vira 'caminhando'; parado, 'em pé'", () => {
    assert.equal(feed(createActivityClassifier(), 120, "EM PÉ", t => POSES.walking(80 + t / 15, t / 260)).id, "caminhando");
    assert.equal(feed(createActivityClassifier(), 120, "EM PÉ", () => POSES.standing(300)).id, "em_pe");
});

test("atividade nova só é registrada depois de durar 1,5 s (sem piscar)", () => {
    const classifier = createActivityClassifier();
    feed(classifier, 120, "EM PÉ", () => POSES.standing(300));
    const brief = feed(classifier, 20, "OUTRA", () => POSES.standing(300), 120 * 33);
    assert.equal(brief.id, "em_pe");
});

test("registro da rotina junta blocos e soma o tempo por atividade", () => {
    const log = createRoutineLog();
    log.record("refeicao", at(7, 30));
    log.record("tv", at(8, 0));
    log.record("tv", at(8, 10));
    log.record("dormindo", at(22, 0));
    const entries = log.getEntries();
    assert.equal(entries.length, 3);
    assert.equal(entries[0].label, "Café da manhã");
    const totals = log.totals(at(23, 0));
    assert.equal(Math.round(totals.find(item => item.id === "tv").minutes), 840);
    assert.equal(formatMinutes(75), "1 h 15 min");
});
