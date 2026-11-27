// Classificação de ATIVIDADES do dia a dia (experimental) e registro da rotina.
//
// A câmera só enxerga os pontos do corpo, não objetos (xícara, TV, livro). Por isso
// a atividade é ESTIMADA pela combinação de postura, gestos e movimento ao longo de
// alguns segundos, mais o horário do dia:
//   • mão indo à boca repetidas vezes, sentado  → refeição (café, almoço, jantar)
//   • mãos erguidas à frente do peito, paradas  → leitura
//   • tronco reclinado para trás e pouco movimento → descanso na poltrona
//   • sentado ereto, mãos no colo, parado          → assistindo TV
//   • deitado e imóvel por alguns segundos          → dormindo (à noite) / cochilando
//   • em pé deslocando-se                           → caminhando
// Os limiares são PROVISÓRIOS e não foram validados com idosos.

export const ACTIVITY_CONFIG = Object.freeze({
    minimumScore: 0.3,
    historyMs: 12_000,
    stillWindowMs: 3_000,
    stillRange: 0.02,              // variação máxima do centro do corpo (fração do quadro)
    sleepAfterMs: 5_000,           // deitado e imóvel por este tempo → dormindo/cochilando
    tvAfterMs: 3_000,              // sentado ereto e parado por este tempo → TV
    eatingWindowMs: 10_000,
    eatingMinimumBites: 2,
    mouthNear: 0.45,               // distância punho–nariz (em comprimentos de tronco)
    mouthFar: 0.7,
    readingWindowMs: 2_000,
    readingMinimumFraction: 0.7,
    reclineMinimumDegrees: 18,
    walkingWindowMs: 1_500,
    walkingSpeed: 0.06,            // fração do quadro por segundo na horizontal
    switchAfterMs: 1_500           // nova atividade precisa durar isso antes de ser registrada
});

export const ACTIVITY_TYPES = Object.freeze({
    refeicao: { label: "Refeição", symbol: "icon-cup", color: "#e0913a" },
    tv: { label: "Assistindo TV", symbol: "icon-tv", color: "#786dc0" },
    leitura: { label: "Lendo", symbol: "icon-book", color: "#3f9bd0" },
    descanso: { label: "Descansando na poltrona", symbol: "icon-armchair", color: "#c78b5d" },
    dormindo: { label: "Dormindo", symbol: "icon-moon", color: "#3c4f8f" },
    deitado: { label: "Deitado, acordado", symbol: "icon-bed", color: "#6c7fb8" },
    sentado: { label: "Sentado", symbol: "icon-armchair", color: "#8f9aa6" },
    caminhando: { label: "Caminhando", symbol: "icon-walk", color: "#22a06b" },
    em_pe: { label: "Em pé", symbol: "icon-person", color: "#4bb58a" },
    outra: { label: "Agachado ou inclinado", symbol: "icon-person", color: "#b98a3e" },
    ausente: { label: "Fora de vista", symbol: "icon-camera", color: "#b5b9c0" }
});

export function mealLabel(date) {
    const hour = date.getHours() + date.getMinutes() / 60;
    if (hour >= 5 && hour < 10.5) return "Café da manhã";
    if (hour >= 11 && hour < 14.5) return "Almoço";
    if (hour >= 14.5 && hour < 17.5) return "Lanche da tarde";
    if (hour >= 17.5 && hour < 21.5) return "Jantar";
    return "Refeição";
}

export function isNight(date) {
    const hour = date.getHours();
    return hour >= 20 || hour < 7;
}

export function activityLabel(id, date) {
    if (id === "refeicao") return mealLabel(date);
    if (id === "dormindo") return isNight(date) ? "Dormindo" : "Cochilando";
    return ACTIVITY_TYPES[id]?.label ?? id;
}

const finite = Number.isFinite;
const mid = (a, b) => a && b ? { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 } : a ?? b ?? null;
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

/** Medidas de um quadro: centro, inclinação do tronco, mãos em relação à boca e ao peito. */
export function poseFeatures(keypoints, width, height, config = ACTIVITY_CONFIG) {
    if (!Array.isArray(keypoints) || !width || !height) return null;
    const get = name => {
        const point = keypoints.find(item => item.name === name);
        return point && finite(point.x) && finite(point.y) && (point.score ?? 0) >= config.minimumScore ? point : null;
    };
    const shoulder = mid(get("left_shoulder"), get("right_shoulder"));
    const hip = mid(get("left_hip"), get("right_hip"));
    if (!shoulder || !hip) return null;
    const torso = dist(shoulder, hip);
    if (torso < 5) return null;

    const nose = get("nose");
    const wrists = [get("left_wrist"), get("right_wrist")].filter(Boolean);
    const trunkDegrees = Math.atan2(Math.abs(shoulder.x - hip.x), Math.abs(hip.y - shoulder.y)) * 180 / Math.PI;
    // Reclinado = ombros atrás do quadril em relação ao lado para onde o rosto aponta.
    // A direção do rosto vem do nariz em relação às orelhas (o ombro engana quando o tronco reclina).
    const ear = mid(get("left_ear"), get("right_ear"));
    const facing = nose ? Math.sign(nose.x - (ear ?? shoulder).x) || 1 : 1;
    const leansBack = (shoulder.x - hip.x) * facing < 0;

    const mouthDistance = nose && wrists.length
        ? Math.min(...wrists.map(wrist => dist(wrist, { x: nose.x, y: nose.y + torso * 0.1 }))) / torso
        : null;
    const raised = wrists.map(wrist => (hip.y - wrist.y) / torso);
    const inFront = wrists.every(wrist => Math.abs(wrist.x - shoulder.x) <= torso * 0.9);
    const handsTogether = wrists.length < 2 || dist(wrists[0], wrists[1]) <= torso * 0.7;
    const readingPose = wrists.length > 0 && inFront && handsTogether &&
        raised.every(value => value >= 0.35 && value <= 1.25) &&
        (mouthDistance === null || mouthDistance > 0.5);
    const handsLow = wrists.length > 0 && raised.every(value => value < 0.35);

    return {
        centerX: (shoulder.x + hip.x) / 2 / width,
        centerY: (shoulder.y + hip.y) / 2 / height,
        trunkDegrees,
        reclined: leansBack && trunkDegrees >= config.reclineMinimumDegrees && trunkDegrees < 65,
        mouthDistance,
        readingPose,
        handsLow
    };
}

export function createActivityClassifier({ config = ACTIVITY_CONFIG } = {}) {
    let history = [];
    let current = null;          // atividade registrada (estável)
    let candidate = null;        // atividade nova aguardando confirmação
    let lyingStillSince = null;
    let sittingStillSince = null;

    function within(ms, now) {
        return history.filter(item => now - item.timestamp <= ms);
    }

    function range(items, key) {
        const values = items.map(item => item[key]).filter(finite);
        return values.length ? Math.max(...values) - Math.min(...values) : 0;
    }

    function countBites(now) {
        let bites = 0;
        let armed = false;
        for (const item of within(config.eatingWindowMs, now)) {
            if (!finite(item.mouthDistance)) continue;
            if (item.mouthDistance >= config.mouthFar) armed = true;
            else if (armed && item.mouthDistance <= config.mouthNear) {
                bites++;
                armed = false;
            }
        }
        return bites;
    }

    function decide(posture, features, now) {
        if (!features) return posture ? { id: current?.id ?? "outra", evidence: "pose incompleta" } : { id: "ausente", evidence: "nenhuma pessoa na imagem" };
        const recent = within(config.stillWindowMs, now);
        const spanMs = recent.length ? now - recent[0].timestamp : 0;
        const still = spanMs >= config.stillWindowMs * 0.8 &&
            Math.max(range(recent, "centerX"), range(recent, "centerY")) < config.stillRange;

        if (posture === "DEITADA") {
            sittingStillSince = null;
            if (still) lyingStillSince ??= recent[0].timestamp;
            else lyingStillSince = null;
            if (lyingStillSince !== null && now - lyingStillSince >= config.sleepAfterMs) {
                return { id: "dormindo", evidence: "deitado e imóvel há alguns segundos" };
            }
            return { id: "deitado", evidence: "deitado, ainda se mexendo" };
        }
        lyingStillSince = null;

        if (posture === "SENTADA") {
            const bites = countBites(now);
            if (bites >= config.eatingMinimumBites) {
                sittingStillSince = null;
                return { id: "refeicao", evidence: `mão levada à boca ${bites} vezes em ${config.eatingWindowMs / 1000} s` };
            }
            const reading = within(config.readingWindowMs, now);
            const readingFraction = reading.filter(item => item.readingPose).length / Math.max(1, reading.length);
            if (readingFraction >= config.readingMinimumFraction) {
                return { id: "leitura", evidence: "mãos erguidas à frente do peito, paradas" };
            }
            if (features.reclined && still) {
                return { id: "descanso", evidence: `tronco reclinado ${Math.round(features.trunkDegrees)}° e pouco movimento` };
            }
            if (still && features.handsLow && !features.reclined) {
                sittingStillSince ??= recent[0].timestamp;
                if (now - sittingStillSince >= config.tvAfterMs) {
                    return { id: "tv", evidence: "sentado ereto, mãos no colo, parado olhando à frente" };
                }
            } else {
                sittingStillSince = null;
            }
            return { id: "sentado", evidence: "sentado" };
        }
        sittingStillSince = null;

        if (posture === "EM PÉ") {
            const walk = within(config.walkingWindowMs, now);
            const seconds = walk.length > 1 ? (now - walk[0].timestamp) / 1000 : 0;
            const speed = seconds > 0.5 ? Math.abs(features.centerX - walk[0].centerX) / seconds : 0;
            return speed >= config.walkingSpeed
                ? { id: "caminhando", evidence: "em pé, deslocando-se" }
                : { id: "em_pe", evidence: "em pé, parado" };
        }
        if (posture === "OUTRA") return { id: "outra", evidence: "postura baixa sem estar sentado ou deitado" };
        // Postura indefinida neste quadro: mantém a atividade atual.
        return { id: current?.id ?? "sentado", evidence: current?.evidence ?? "postura indefinida" };
    }

    /**
     * @param {{timestamp:number, posture:string|null, keypoints:Array|null, width:number, height:number}} input
     * @returns {{id:string, evidence:string, since:number, changed:boolean, previous:object|null}}
     */
    function update({ timestamp, posture, keypoints, width, height }) {
        const features = keypoints ? poseFeatures(keypoints, width, height, config) : null;
        if (features) history.push({ timestamp, ...features });
        history = history.filter(item => timestamp - item.timestamp <= config.historyMs);
        const decision = decide(keypoints ? posture : null, features, timestamp);

        if (!current) {
            current = { ...decision, since: timestamp };
            return { ...current, changed: true, previous: null };
        }
        if (decision.id === current.id) {
            current.evidence = decision.evidence;
            candidate = null;
            return { ...current, changed: false, previous: null };
        }
        if (!candidate || candidate.id !== decision.id) candidate = { ...decision, since: timestamp };
        if (timestamp - candidate.since < config.switchAfterMs) return { ...current, changed: false, previous: null };
        const previous = current;
        current = { ...decision, since: candidate.since };
        candidate = null;
        return { ...current, changed: true, previous };
    }

    function reset() {
        history = [];
        current = null;
        candidate = null;
        lyingStillSince = null;
        sittingStillSince = null;
    }

    return { update, reset, getCurrent: () => current };
}

/** Registro da rotina: blocos contínuos de atividade com início e fim no relógio do dia. */
export function createRoutineLog({ limit = 200 } = {}) {
    let entries = [];

    function record(id, at, { evidence = "", source = "câmera" } = {}) {
        const last = entries.at(-1);
        if (last && !last.end) last.end = at;
        if (last && last.id === id && last.source === source) {
            last.end = null;
            return last;
        }
        const entry = { id, label: activityLabel(id, at), start: at, end: null, evidence, source };
        entries.push(entry);
        entries = entries.slice(-limit);
        return entry;
    }

    function close(at) {
        const last = entries.at(-1);
        if (last && !last.end) last.end = at;
    }

    function totals(now) {
        const sums = new Map();
        for (const entry of entries) {
            if (entry.id === "ausente") continue;
            const end = entry.end ?? now;
            const minutes = Math.max(0, (end - entry.start) / 60_000);
            const key = entry.label;
            const item = sums.get(key) ?? { label: key, id: entry.id, minutes: 0 };
            item.minutes += minutes;
            sums.set(key, item);
        }
        return [...sums.values()].sort((a, b) => b.minutes - a.minutes);
    }

    return {
        record, close, totals,
        clear: () => { entries = []; },
        getEntries: () => entries.slice()
    };
}

function formatTime(date) {
    return date.toLocaleTimeString("pt-BR", { hour: "2-digit", minute: "2-digit" });
}

export function formatMinutes(minutes) {
    if (minutes < 1) return "< 1 min";
    const hours = Math.floor(minutes / 60);
    const rest = Math.round(minutes % 60);
    return hours ? `${hours} h ${String(rest).padStart(2, "0")} min` : `${rest} min`;
}

export function installActivity(app) {
    const classifier = createActivityClassifier();
    const log = createRoutineLog();
    app.routineLog = log;
    app.routineSource = "câmera";
    // A simulação substitui o relógio para contar um dia inteiro em poucos minutos.
    app.routineClock = () => new Date();
    app.latestActivity = null;

    const element = id => document.getElementById(id);

    app.updateActivity = function(analysisResult, pose, width, height) {
        const timestamp = analysisResult?.timestamp ?? Date.now();
        const posture = analysisResult?.posture?.state === "INCONCLUSIVA" ? null : analysisResult?.posture?.state ?? null;
        const result = classifier.update({ timestamp, posture, keypoints: pose?.keypoints ?? null, width, height });
        const clock = app.routineClock();
        const label = activityLabel(result.id, clock);
        app.latestActivity = { ...result, label, clock };
        if (result.changed) {
            log.record(result.id, clock, { evidence: result.evidence, source: app.routineSource });
            if (result.previous && result.id !== "ausente") {
                app.registerEvent?.(`Rotina: ${label} (${result.evidence}).`);
            }
        }
        app.renderRoutine();
        return app.latestActivity;
    };

    app.resetActivity = function({ clearLog = false } = {}) {
        classifier.reset();
        log.close(app.routineClock());
        if (clearLog) log.clear();
        app.latestActivity = null;
        app.renderRoutine(true);
    };

    let lastRender = 0;
    app.renderRoutine = function(force = false) {
        const now = Date.now();
        if (!force && now - lastRender < 250) return;
        lastRender = now;
        const clock = app.routineClock();
        const current = app.latestActivity;

        const card = element("routineCurrent");
        if (card) {
            const type = ACTIVITY_TYPES[current?.id] ?? null;
            const activityId = current?.id ?? "nenhuma";
            if (card.dataset.activity !== activityId) {
                card.dataset.activity = activityId;
                card.style.setProperty("--activity-color", type?.color ?? "#8f9aa6");
            }
            element("routineCurrentIcon")?.setAttribute("href", `#${type?.symbol ?? "icon-activity"}`);
            element("routineCurrentLabel").textContent = current ? current.label : "Aguardando câmera ou simulação";
            const entries = log.getEntries();
            const open = entries.at(-1);
            element("routineCurrentSince").textContent = current && open
                ? `desde ${formatTime(open.start)} · ${formatMinutes((clock - open.start) / 60_000)}`
                : "A atividade aparece aqui quando há uma pessoa na imagem.";
            element("routineCurrentEvidence").textContent = current ? `Baseado em: ${current.evidence}` : "";
        }

        const entries = log.getEntries();
        const list = element("routineList");
        if (list) {
            list.replaceChildren();
            if (!entries.length) {
                const empty = document.createElement("li");
                empty.className = "empty-state";
                empty.textContent = "Nenhuma atividade registrada ainda.";
                list.append(empty);
            }
            for (const entry of entries.slice().reverse()) {
                const type = ACTIVITY_TYPES[entry.id] ?? ACTIVITY_TYPES.sentado;
                const item = document.createElement("li");
                item.className = "routine-entry";
                item.style.setProperty("--activity-color", type.color);
                const end = entry.end ?? clock;
                item.innerHTML = `<span class="routine-entry-icon"><svg><use href="#${type.symbol}"/></svg></span>
                    <div class="routine-entry-body"><strong></strong><small></small></div>
                    <div class="routine-entry-time"><time></time><span></span></div>`;
                item.querySelector("strong").textContent = entry.label;
                item.querySelector("small").textContent = entry.evidence;
                item.querySelector("time").textContent = `${formatTime(entry.start)} – ${entry.end ? formatTime(entry.end) : "agora"}`;
                item.querySelector(".routine-entry-time span").textContent =
                    `${formatMinutes((end - entry.start) / 60_000)}${entry.source === "simulação" ? " · simulação" : ""}`;
                list.append(item);
            }
        }

        const bar = element("routineDayBar");
        if (bar) {
            bar.replaceChildren();
            for (const entry of entries) {
                if (entry.id === "ausente") continue;
                const start = entry.start.getHours() * 60 + entry.start.getMinutes() + entry.start.getSeconds() / 60;
                const endDate = entry.end ?? clock;
                let end = endDate.getHours() * 60 + endDate.getMinutes() + endDate.getSeconds() / 60;
                if (end < start) end = 24 * 60;
                const segment = document.createElement("span");
                segment.className = "routine-segment";
                segment.style.left = `${start / 14.4}%`;
                segment.style.width = `${Math.max(0.6, (end - start) / 14.4)}%`;
                segment.style.background = (ACTIVITY_TYPES[entry.id] ?? ACTIVITY_TYPES.sentado).color;
                segment.title = `${entry.label}: ${formatTime(entry.start)} – ${formatTime(endDate)}`;
                bar.append(segment);
            }
            const marker = document.createElement("span");
            marker.className = "routine-now";
            marker.style.left = `${(clock.getHours() * 60 + clock.getMinutes()) / 14.4}%`;
            bar.append(marker);
        }

        const summary = element("routineSummary");
        if (summary) {
            summary.replaceChildren();
            const totals = log.totals(clock);
            if (!totals.length) {
                const empty = document.createElement("span");
                empty.className = "routine-chip muted";
                empty.textContent = "Sem dados ainda";
                summary.append(empty);
            }
            for (const total of totals.slice(0, 6)) {
                const chip = document.createElement("span");
                chip.className = "routine-chip";
                chip.style.setProperty("--activity-color", (ACTIVITY_TYPES[total.id] ?? ACTIVITY_TYPES.sentado).color);
                chip.innerHTML = "<i></i><b></b><span></span>";
                chip.querySelector("b").textContent = total.label;
                chip.querySelector("span").textContent = formatMinutes(total.minutes);
                summary.append(chip);
            }
        }
    };

    element("clearRoutine")?.addEventListener("click", () => {
        log.clear();
        app.renderRoutine(true);
    });
    app.renderRoutine(true);
}
