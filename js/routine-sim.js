// Simulação ilustrada de um dia da rotina (SEM câmera).
//
// Uma pessoa idosa fictícia, a Dona Lúcia, toma café, assiste TV, lê, descansa
// na poltrona e vai dormir. Os pontos do corpo são gerados a partir de um
// esqueleto com proporções humanas e passam pelo MESMO caminho da câmera real:
// postura (regra + Random Forest), movimento, detecção de queda, gravidade e
// classificação de atividade. O desenho da cena é só ilustração; a análise usa
// apenas os 17 pontos, como faria com o MoveNet.

const FRAME = Object.freeze({ width: 640, height: 480 });
const SIZE = Object.freeze({ torso: 100, neck: 16, head: 19, thigh: 82, shin: 84, upper: 56, fore: 52 });
const FAR = Object.freeze({ x: 6, y: -4 });
const DESCENT_LIMIT = 0.45; // mesmo limite nominal de FALL_DETECTION_CONFIG.rapidDownwardSpeedPerSecond
const COCO = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"];

const rad = degrees => degrees * Math.PI / 180;
// Ângulo medido a partir da vertical para baixo; positivo = para a frente (direita).
const seg = (point, length, degrees) => ({ x: point.x + length * Math.sin(rad(degrees)), y: point.y + length * Math.cos(rad(degrees)) });
const lerp = (a, b, k) => a + (b - a) * k;
const ease = k => { const c = Math.min(1, Math.max(0, k)); return c * c * (3 - 2 * c); };
const add = (a, b) => ({ x: a.x + b.x, y: a.y + b.y });

export function blend(a, b, k) {
    const out = { ...b };
    for (const key of Object.keys(a)) {
        if (typeof a[key] === "number" && typeof b[key] === "number") out[key] = lerp(a[key], b[key], k);
        else if (a[key] && b[key] && typeof a[key] === "object") out[key] = { x: lerp(a[key].x, b[key].x, k), y: lerp(a[key].y, b[key].y, k) };
    }
    return out;
}

// ---------------------------------------------------------------- posturas
const BASE = { x: 300, y: 272, rot: 0, lean: 2, headTilt: 0,
    thighN: 0, shinN: 0, thighF: 0, shinF: 0, upperN: 6, foreN: 10, upperF: 4, foreF: 8 };

export const POSES = {
    standing: (x = 300) => ({ ...BASE, x }),
    walking: (x, phase) => {
        const swing = Math.sin(phase) * 19;
        return { ...BASE, x, y: 272 - Math.abs(Math.cos(phase)) * 3, lean: 4,
            thighN: swing, shinN: swing * 0.55 - (swing < 0 ? 8 : 0), thighF: -swing, shinF: -swing * 0.55 - (swing > 0 ? 8 : 0),
            upperN: -swing * 0.8, foreN: -swing * 0.8 + 14, upperF: swing * 0.8, foreF: swing * 0.8 + 14 };
    },
    seated: (x, y = 347) => ({ ...BASE, x, y, lean: 5, thighN: 84, shinN: 2, thighF: 81, shinF: -2,
        upperN: 14, foreN: 60, upperF: 12, foreF: 57 }),
    reading: (x, y = 350) => ({ ...BASE, x, y, lean: 8, headTilt: 18, thighN: 84, shinN: 2, thighF: 81, shinF: -2,
        upperN: 26, foreN: 115, upperF: 22, foreF: 112 }),
    reclined: (x, y = 352) => ({ ...BASE, x, y, lean: -30, headTilt: -6, thighN: 76, shinN: 12, thighF: 73, shinF: 8,
        upperN: 22, foreN: 82, upperF: 18, foreF: 80 }),
    bedSitting: (x, y = 340) => ({ ...BASE, x, y, lean: 3, thighN: 88, shinN: 0, thighF: 85, shinF: -3,
        upperN: 12, foreN: 40, upperF: 10, foreF: 36 }),
    lying: (x, y = 326) => ({ ...BASE, x, y, rot: 90, lean: 0, headTilt: 0, thighN: 0, shinN: 0, thighF: 2, shinF: 1,
        upperN: 6, foreN: 10, upperF: 5, foreF: 8 })
};

/** Constrói os 17 pontos (ordem COCO, como o MoveNet) e as partes usadas no desenho. */
export function buildBody(p, random = Math.random) {
    const R = p.rot ?? 0;
    const hip = { x: p.x, y: p.y };
    const shoulder = seg(hip, SIZE.torso, 180 - p.lean + R);
    const headAngle = 180 - p.lean + R;
    const neck = seg(shoulder, SIZE.neck, headAngle - (p.headTilt ?? 0) * 0.5);
    const head = seg(neck, SIZE.head, headAngle - (p.headTilt ?? 0));
    const forward = 90 + R - (p.headTilt ?? 0);
    const nose = seg(head, 15, forward);
    const eye = seg(head, 12, forward + 35);
    const ear = seg(head, 5, forward + 180);

    const side = (thigh, shin, upper, fore, elbowOverride, wristOverride) => {
        const knee = seg(hip, SIZE.thigh, thigh + R);
        const ankle = seg(knee, SIZE.shin, shin + R);
        const elbow = elbowOverride ?? seg(shoulder, SIZE.upper, upper + R);
        const wrist = wristOverride ?? seg(elbow, SIZE.fore, fore + R);
        return { knee, ankle, elbow, wrist };
    };
    const near = side(p.thighN, p.shinN, p.upperN, p.foreN, p.elbowN, p.wristN);
    const farRaw = side(p.thighF, p.shinF, p.upperF, p.foreF, p.elbowF, p.wristF);
    const far = Object.fromEntries(Object.entries(farRaw).map(([key, point]) => [key, add(point, FAR)]));

    const parts = {
        rot: R, hip, shoulder, neck, head, nose, eye, ear, forward,
        near: { hip, shoulder, ...near },
        far: { hip: add(hip, FAR), shoulder: add(shoulder, FAR), ...far }
    };
    const raw = {
        nose, left_eye: eye, right_eye: add(eye, FAR), left_ear: ear, right_ear: add(ear, FAR),
        left_shoulder: shoulder, right_shoulder: parts.far.shoulder,
        left_elbow: near.elbow, right_elbow: far.elbow, left_wrist: near.wrist, right_wrist: far.wrist,
        left_hip: hip, right_hip: parts.far.hip, left_knee: near.knee, right_knee: far.knee,
        left_ankle: near.ankle, right_ankle: far.ankle
    };
    const keypoints = COCO.map(name => {
        const farSide = name.startsWith("right_");
        const point = raw[name];
        return {
            name,
            x: point.x + (random() - 0.5) * 1.6,
            y: point.y + (random() - 0.5) * 1.6,
            score: (farSide ? 0.62 : 0.84) + random() * 0.12
        };
    });
    return { keypoints, parts };
}

// ---------------------------------------------------------------- roteiro do dia
const breathing = t => Math.sin(t / 650) * 0.8;

export const SEGMENTS = [
    {
        id: "cafe", scene: "cozinha", camera: "Cozinha", start: [7, 30], end: [8, 5], ms: 20_000,
        caption: "Dona Lúcia toma o café da manhã",
        pose(t) {
            const chairX = 250;
            if (t < 3_400) return POSES.walking(lerp(70, chairX, t / 3_400), t / 260);
            if (t < 5_400) return blend(POSES.standing(chairX), POSES.seated(chairX), ease((t - 3_400) / 2_000));
            const base = POSES.seated(chairX);
            const cup = { x: chairX + 112, y: 312 };
            const cycle = (t - 5_400) % 2_700;
            const up = cycle < 600 ? ease(cycle / 600) : cycle < 1_100 ? 1 : cycle < 1_700 ? 1 - ease((cycle - 1_100) / 600) : 0;
            const body = buildBody(base, () => 0.5).parts;
            const mouth = { x: body.nose.x - 2, y: body.nose.y + 9 };
            const elbowRest = { x: chairX + 62, y: 300 };
            const elbowUp = { x: chairX + 58, y: 262 };
            return { ...base, lean: 9, upperF: 30, foreF: 70,
                elbowN: { x: lerp(elbowRest.x, elbowUp.x, up), y: lerp(elbowRest.y, elbowUp.y, up) },
                wristN: { x: lerp(cup.x, mouth.x, up), y: lerp(cup.y, mouth.y, up) },
                prop: { type: "cup", lifted: up } };
        }
    },
    {
        id: "tv", scene: "sala", camera: "Sala", start: [9, 0], end: [9, 50], ms: 15_000,
        caption: "Assiste ao jornal da manhã na TV",
        pose: t => ({ ...POSES.seated(230, 350 + breathing(t) * 0.3), headTilt: -3 + Math.sin(t / 2_300) * 2 })
    },
    {
        id: "leitura", scene: "leitura", camera: "Sala", start: [10, 30], end: [11, 20], ms: 15_000,
        caption: "Lê um livro na poltrona",
        pose(t) {
            const k = ease(t / 1_800);
            const pose = blend(POSES.seated(230, 350), POSES.reading(230, 350), k);
            return { ...pose, y: pose.y + breathing(t) * 0.3, prop: { type: "book", open: k } };
        }
    },
    {
        id: "descanso", scene: "tarde", camera: "Sala", start: [15, 0], end: [16, 10], ms: 15_000,
        caption: "Descansa na poltrona depois do almoço",
        pose(t) {
            const k = ease(t / 2_200);
            const pose = blend(POSES.seated(230, 350), POSES.reclined(230, 352), k);
            return { ...pose, y: pose.y + breathing(t) * 0.25, prop: { type: "blanket", amount: k } };
        }
    },
    {
        id: "dormir", scene: "quarto", camera: "Quarto", start: [22, 0], end: [22, 45], ms: 26_000,
        caption: "Vai dormir: deita-se devagar, sem alerta de queda",
        pose(t) {
            const bedX = 300;
            if (t < 3_000) return POSES.walking(lerp(30, bedX + 40, t / 3_000), t / 260);
            if (t < 5_200) {
                const turn = blend(POSES.standing(bedX + 40), POSES.bedSitting(bedX), ease((t - 3_000) / 2_200));
                return turn;
            }
            if (t < 6_200) return POSES.bedSitting(bedX);
            if (t < 9_700) {
                const k = ease((t - 6_200) / 3_500);
                return { ...blend(POSES.bedSitting(bedX), POSES.lying(bedX), k), prop: { type: "blanket-bed", amount: Math.max(0, (k - 0.7) / 0.3) } };
            }
            return { ...POSES.lying(bedX, 326 + breathing(t) * 0.25), prop: { type: "blanket-bed", amount: 1 } };
        }
    }
];

export const TOTAL_MS = SEGMENTS.reduce((sum, item) => sum + item.ms, 0);

export function segmentAt(elapsed) {
    let start = 0;
    for (const [index, segment] of SEGMENTS.entries()) {
        if (elapsed < start + segment.ms) return { segment, index, local: elapsed - start };
        start += segment.ms;
    }
    const last = SEGMENTS.at(-1);
    return { segment: last, index: SEGMENTS.length - 1, local: last.ms };
}

export function clockAt(segment, local, day = new Date()) {
    const startMinutes = segment.start[0] * 60 + segment.start[1];
    const endMinutes = segment.end[0] * 60 + segment.end[1];
    const minutes = lerp(startMinutes, endMinutes, Math.min(1, local / segment.ms));
    const date = new Date(day);
    date.setHours(0, 0, 0, 0);
    date.setTime(date.getTime() + minutes * 60_000);
    return date;
}

// ---------------------------------------------------------------- desenho
const PALETTE = {
    skin: "#f2c9a8", skinShade: "#dcae8c", hair: "#e4e2e8", hairShade: "#c9c6cf",
    cardigan: "#5d8fd6", cardiganFar: "#4a76b8", pants: "#4b5168", pantsFar: "#3c4156",
    slipper: "#8a5540", glasses: "#3b3f4a"
};

function skyFor(hour) {
    if (hour >= 6 && hour < 8.5) return ["#f7c59f", "#a9d8f2", 0.15];
    if (hour >= 8.5 && hour < 15.5) return ["#7cc6f2", "#d4f0fd", 0];
    if (hour >= 15.5 && hour < 18.5) return ["#f2a35e", "#fde0ab", 0.08];
    if (hour >= 18.5 && hour < 20) return ["#5a4a8c", "#f08c62", 0.3];
    return ["#0b1430", "#253765", 0.55];
}

function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
}

function limb(ctx, points, width, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (const point of points.slice(1)) ctx.lineTo(point.x, point.y);
    ctx.stroke();
}

function drawWindow(ctx, x, y, w, h, hour, time) {
    const [top, bottom] = skyFor(hour);
    ctx.save();
    roundRect(ctx, x - 7, y - 7, w + 14, h + 14, 6);
    ctx.fillStyle = "#fbfaf7";
    ctx.fill();
    roundRect(ctx, x, y, w, h, 3);
    ctx.clip();
    const sky = ctx.createLinearGradient(0, y, 0, y + h);
    sky.addColorStop(0, top);
    sky.addColorStop(1, bottom);
    ctx.fillStyle = sky;
    ctx.fillRect(x, y, w, h);
    const night = hour >= 20 || hour < 6;
    if (night) {
        ctx.fillStyle = "#fff";
        for (let i = 0; i < 18; i++) {
            const sx = x + ((i * 53) % w), sy = y + ((i * 31) % (h * 0.8));
            ctx.globalAlpha = 0.5 + 0.5 * Math.sin(time / 600 + i);
            ctx.fillRect(sx, sy, 1.6, 1.6);
        }
        ctx.globalAlpha = 1;
        ctx.fillStyle = "#f6f1d5";
        ctx.beginPath(); ctx.arc(x + w * 0.7, y + h * 0.3, 13, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = top;
        ctx.beginPath(); ctx.arc(x + w * 0.7 + 6, y + h * 0.3 - 4, 11, 0, Math.PI * 2); ctx.fill();
    } else {
        const progress = Math.min(1, Math.max(0, (hour - 6) / 13));
        const sunX = x + w * (0.15 + 0.7 * progress);
        const sunY = y + h * (0.85 - Math.sin(progress * Math.PI) * 0.6);
        const glow = ctx.createRadialGradient(sunX, sunY, 2, sunX, sunY, 34);
        glow.addColorStop(0, "rgba(255,240,180,.95)");
        glow.addColorStop(1, "rgba(255,240,180,0)");
        ctx.fillStyle = glow;
        ctx.fillRect(x, y, w, h);
        ctx.fillStyle = hour > 15.5 ? "#ffb35c" : "#ffe27a";
        ctx.beginPath(); ctx.arc(sunX, sunY, 11, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = "rgba(255,255,255,.85)";
        const cloudX = x + ((time / 90) % (w + 80)) - 40;
        for (const [dx, dy, r] of [[0, 0, 10], [12, -5, 12], [25, 0, 9]]) {
            ctx.beginPath(); ctx.arc(cloudX + dx, y + h * 0.62 + dy, r, 0, Math.PI * 2); ctx.fill();
        }
    }
    // Hills
    ctx.fillStyle = night ? "#15223f" : hour > 15.5 ? "#c9894e" : "#86c08a";
    ctx.beginPath();
    ctx.moveTo(x, y + h);
    ctx.quadraticCurveTo(x + w * 0.3, y + h * 0.68, x + w * 0.6, y + h * 0.86);
    ctx.quadraticCurveTo(x + w * 0.8, y + h * 0.74, x + w, y + h * 0.82);
    ctx.lineTo(x + w, y + h);
    ctx.fill();
    ctx.restore();
    ctx.strokeStyle = "#e9e3d8";
    ctx.lineWidth = 4;
    ctx.beginPath(); ctx.moveTo(x + w / 2, y); ctx.lineTo(x + w / 2, y + h); ctx.moveTo(x, y + h / 2); ctx.lineTo(x + w, y + h / 2); ctx.stroke();
    // Curtains
    ctx.fillStyle = "rgba(214,120,108,.85)";
    for (const side of [-1, 1]) {
        const cx = side < 0 ? x - 14 : x + w + 2;
        ctx.beginPath();
        ctx.moveTo(cx, y - 12);
        ctx.lineTo(cx + 12, y - 12);
        ctx.quadraticCurveTo(cx + 6 + side * 4, y + h / 2, cx + 12, y + h + 16);
        ctx.lineTo(cx, y + h + 16);
        ctx.fill();
    }
}

function drawClock(ctx, x, y, date) {
    ctx.save();
    ctx.fillStyle = "#fffdf8";
    ctx.strokeStyle = "#7a6a5a";
    ctx.lineWidth = 3;
    ctx.beginPath(); ctx.arc(x, y, 22, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    ctx.fillStyle = "#7a6a5a";
    for (let i = 0; i < 12; i++) {
        const a = i / 12 * Math.PI * 2;
        ctx.fillRect(x + Math.sin(a) * 17 - 1, y - Math.cos(a) * 17 - 1, 2, 2);
    }
    const hours = date.getHours() % 12 + date.getMinutes() / 60;
    const minutes = date.getMinutes();
    ctx.lineCap = "round";
    ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + Math.sin(hours / 12 * Math.PI * 2) * 10, y - Math.cos(hours / 12 * Math.PI * 2) * 10); ctx.stroke();
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + Math.sin(minutes / 60 * Math.PI * 2) * 15, y - Math.cos(minutes / 60 * Math.PI * 2) * 15); ctx.stroke();
    ctx.restore();
}

function drawRoom(ctx, scene, hour, time, clock) {
    const walls = { cozinha: ["#f6ead8", "#efdcc3"], sala: ["#eef1f4", "#dde4ec"], leitura: ["#eef1f4", "#dde4ec"],
        tarde: ["#f6e7d6", "#ecd3b8"], quarto: ["#e3e6f3", "#cfd4ea"] };
    const [wallTop, wallBottom] = walls[scene];
    const wall = ctx.createLinearGradient(0, 0, 0, 360);
    wall.addColorStop(0, wallTop);
    wall.addColorStop(1, wallBottom);
    ctx.fillStyle = wall;
    ctx.fillRect(0, 0, FRAME.width, 362);
    // Wallpaper stripes
    ctx.fillStyle = "rgba(255,255,255,.18)";
    for (let x = 0; x < FRAME.width; x += 34) ctx.fillRect(x, 0, 14, 362);
    // Floor
    const floor = ctx.createLinearGradient(0, 362, 0, 480);
    floor.addColorStop(0, scene === "cozinha" ? "#c9a27c" : "#b98b62");
    floor.addColorStop(1, scene === "cozinha" ? "#a57c58" : "#91684a");
    ctx.fillStyle = floor;
    ctx.fillRect(0, 362, FRAME.width, 118);
    ctx.strokeStyle = "rgba(80,50,30,.18)";
    ctx.lineWidth = 1;
    for (let y = 378; y < 480; y += 18) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(FRAME.width, y); ctx.stroke(); }
    ctx.fillStyle = "#f8f4ee";
    ctx.fillRect(0, 356, FRAME.width, 8);

    if (scene === "cozinha") {
        drawWindow(ctx, 40, 70, 120, 110, hour, time);
        // Cabinets
        ctx.fillStyle = "#ffffff";
        roundRect(ctx, 420, 40, 200, 80, 6); ctx.fill();
        ctx.strokeStyle = "#e3d6c4"; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(487, 44); ctx.lineTo(487, 116); ctx.moveTo(553, 44); ctx.lineTo(553, 116); ctx.stroke();
        ctx.fillStyle = "#c9b8a2";
        for (const hx of [478, 496, 544, 562]) ctx.fillRect(hx, 86, 3, 14);
        // Plant
        ctx.fillStyle = "#d9825b"; roundRect(ctx, 210, 150, 28, 24, 4); ctx.fill();
        ctx.fillStyle = "#5fa46a";
        for (const [dx, dy] of [[-10, -12], [0, -22], [10, -12], [-4, -16], [6, -18]]) { ctx.beginPath(); ctx.ellipse(224 + dx, 150 + dy, 6, 12, dx / 20, 0, Math.PI * 2); ctx.fill(); }
        drawClock(ctx, 330, 86, clock);
    } else if (scene === "quarto") {
        drawWindow(ctx, 430, 60, 120, 110, hour, time);
        // Picture frame
        ctx.fillStyle = "#fff"; roundRect(ctx, 180, 110, 70, 52, 4); ctx.fill();
        ctx.fillStyle = "#9fc6e6"; ctx.fillRect(186, 116, 58, 40);
        ctx.fillStyle = "#7fb07d"; ctx.beginPath(); ctx.moveTo(186, 156); ctx.lineTo(208, 132); ctx.lineTo(228, 150); ctx.lineTo(244, 138); ctx.lineTo(244, 156); ctx.fill();
        drawClock(ctx, 330, 80, clock);
    } else {
        drawWindow(ctx, 380, 64, 120, 110, hour, time);
        // Shelf with books
        ctx.fillStyle = "#a9805e"; ctx.fillRect(60, 120, 120, 7);
        const colors = ["#d9775f", "#5d8fd6", "#e8b65a", "#6ab08a", "#8d78c9"];
        colors.forEach((color, i) => { ctx.fillStyle = color; ctx.fillRect(68 + i * 16, 86 + (i % 2) * 6, 12, 34 - (i % 2) * 6); });
        drawClock(ctx, 260, 82, clock);
    }
}

function drawArmchair(ctx, x, recline = 0) {
    ctx.save();
    ctx.fillStyle = "#b5654f";
    // Back
    ctx.beginPath();
    ctx.moveTo(x - 70, 360);
    ctx.lineTo(x - 72 - recline * 34, 205);
    ctx.quadraticCurveTo(x - 50 - recline * 34, 186, x - 28 - recline * 30, 205);
    ctx.lineTo(x - 22, 350);
    ctx.closePath();
    ctx.fill();
    // Seat
    ctx.fillStyle = "#c97761";
    roundRect(ctx, x - 70, 348, 140, 34, 12); ctx.fill();
    ctx.fillStyle = "#a65a46";
    roundRect(ctx, x - 78, 375, 156, 30, 10); ctx.fill();
    ctx.fillStyle = "#6e4636";
    ctx.fillRect(x - 70, 403, 8, 30); ctx.fillRect(x + 60, 403, 8, 30);
    ctx.restore();
}

function drawArmrest(ctx, x) {
    ctx.fillStyle = "#b86a54";
    roundRect(ctx, x - 82, 318, 34, 64, 12); ctx.fill();
}

function drawFurnitureBack(ctx, scene, segmentState) {
    if (scene === "cozinha") {
        const x = 250;
        ctx.fillStyle = "#9b6b47";
        ctx.fillRect(x - 46, 230, 9, 130);              // encosto
        roundRect(ctx, x - 50, 225, 16, 14, 4); ctx.fill();
        ctx.fillStyle = "#b07a52";
        roundRect(ctx, x - 48, 350, 92, 10, 3); ctx.fill(); // assento
        ctx.fillStyle = "#8a5d3d";
        ctx.fillRect(x - 44, 360, 6, 80); ctx.fillRect(x + 34, 360, 6, 80);
    } else if (scene === "sala" || scene === "leitura" || scene === "tarde") {
        if (scene === "leitura") {
            // Floor lamp
            ctx.fillStyle = "#4a4a52"; ctx.fillRect(108, 150, 4, 290); ctx.fillRect(92, 436, 36, 5);
            const glow = ctx.createRadialGradient(110, 170, 5, 110, 190, 160);
            glow.addColorStop(0, "rgba(255,214,140,.55)"); glow.addColorStop(1, "rgba(255,214,140,0)");
            ctx.fillStyle = glow; ctx.fillRect(0, 40, 300, 330);
            ctx.fillStyle = "#f2d39a"; ctx.beginPath(); ctx.moveTo(88, 172); ctx.lineTo(132, 172); ctx.lineTo(124, 140); ctx.lineTo(96, 140); ctx.fill();
        }
        drawArmchair(ctx, 230, scene === "tarde" ? (segmentState.recline ?? 1) : 0);
        drawArmrest(ctx, 230);
    } else if (scene === "quarto") {
        // Nightstand + lamp
        ctx.fillStyle = "#a9805e"; roundRect(ctx, 22, 330, 66, 100, 5); ctx.fill();
        ctx.fillStyle = "#8e6a4c"; ctx.fillRect(30, 372, 50, 3);
        const glow = ctx.createRadialGradient(55, 300, 4, 55, 300, 150);
        glow.addColorStop(0, "rgba(255,210,140,.6)"); glow.addColorStop(1, "rgba(255,210,140,0)");
        ctx.fillStyle = glow; ctx.fillRect(0, 150, 260, 300);
        ctx.fillStyle = "#7d7f8c"; ctx.fillRect(52, 300, 5, 30);
        ctx.fillStyle = "#f6dca8"; ctx.beginPath(); ctx.moveTo(36, 304); ctx.lineTo(74, 304); ctx.lineTo(66, 276); ctx.lineTo(44, 276); ctx.fill();
        // Bed
        ctx.fillStyle = "#7d5a44"; roundRect(ctx, 96, 236, 22, 200, 6); ctx.fill();   // cabeceira
        ctx.fillStyle = "#7d5a44"; ctx.fillRect(108, 365, 420, 30);
        ctx.fillRect(112, 395, 10, 40); ctx.fillRect(512, 395, 10, 40);
        ctx.fillStyle = "#f4f2fa"; roundRect(ctx, 112, 334, 420, 34, 10); ctx.fill();  // colchão
        ctx.fillStyle = "#ffffff"; roundRect(ctx, 122, 306, 84, 32, 14); ctx.fill();   // travesseiro
        ctx.strokeStyle = "#dcdbe8"; ctx.lineWidth = 2; ctx.stroke();
        ctx.fillStyle = "#c9d3ef"; roundRect(ctx, 112, 334, 420, 11, 6); ctx.fill();   // colcha sobre o colchão
    }
}

function drawFurnitureFront(ctx, scene, prop, time) {
    if (scene === "cozinha") {
        // Table in front of the legs
        ctx.fillStyle = "#f3efe8"; roundRect(ctx, 350, 318, 270, 12, 4); ctx.fill();
        const cloth = ctx.createLinearGradient(0, 328, 0, 372);
        cloth.addColorStop(0, "#e7a9a0"); cloth.addColorStop(1, "#d88d84");
        ctx.fillStyle = cloth;
        ctx.beginPath(); ctx.moveTo(352, 328); ctx.lineTo(618, 328); ctx.lineTo(612, 372);
        for (let x = 612; x > 352; x -= 20) ctx.quadraticCurveTo(x - 10, 378, x - 20, 372);
        ctx.fill();
        ctx.fillStyle = "#8a5d3d"; ctx.fillRect(372, 370, 9, 70); ctx.fillRect(590, 370, 9, 70);
        // Plate with bread and fruit
        ctx.fillStyle = "#ffffff"; ctx.beginPath(); ctx.ellipse(470, 316, 34, 7, 0, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = "#d9a35e"; ctx.beginPath(); ctx.ellipse(462, 309, 16, 8, 0, Math.PI, 0); ctx.fill();
        ctx.fillStyle = "#e7604f"; ctx.beginPath(); ctx.arc(486, 309, 6, 0, Math.PI * 2); ctx.fill();
        // Coffee pot
        ctx.fillStyle = "#3f4b5c"; roundRect(ctx, 540, 278, 34, 40, 8); ctx.fill();
        ctx.fillRect(572, 288, 8, 4);
        // Cup on table when not lifted
        if (!prop || prop.lifted < 0.05) drawCup(ctx, 362, 304, time);
    } else if (scene === "sala" || scene === "leitura" || scene === "tarde") {
        // TV and stand on the right
        ctx.fillStyle = "#6b4f3d"; roundRect(ctx, 470, 360, 160, 60, 6); ctx.fill();
        ctx.fillStyle = "#59402f"; ctx.fillRect(478, 386, 144, 3);
        ctx.fillStyle = "#1e2229"; roundRect(ctx, 462, 220, 172, 112, 8); ctx.fill();
        ctx.fillRect(540, 332, 16, 28);
        const on = scene === "sala";
        if (on) {
            const hue = (time / 40) % 360;
            const screen = ctx.createLinearGradient(470, 228, 626, 324);
            screen.addColorStop(0, `hsl(${hue}, 60%, 55%)`); screen.addColorStop(1, `hsl(${(hue + 80) % 360}, 65%, 45%)`);
            ctx.fillStyle = screen; ctx.fillRect(470, 228, 156, 96);
            ctx.fillStyle = "rgba(255,255,255,.85)"; ctx.fillRect(482, 300, 70, 6); ctx.fillRect(482, 310, 44, 5);
            const light = ctx.createRadialGradient(540, 280, 10, 420, 300, 260);
            light.addColorStop(0, `hsla(${hue}, 70%, 70%, .25)`); light.addColorStop(1, "hsla(0,0%,100%,0)");
            ctx.fillStyle = light; ctx.fillRect(160, 160, 480, 300);
        } else {
            ctx.fillStyle = "#2a2f38"; ctx.fillRect(470, 228, 156, 96);
            ctx.fillStyle = "rgba(255,255,255,.06)"; ctx.beginPath(); ctx.moveTo(470, 228); ctx.lineTo(540, 228); ctx.lineTo(470, 300); ctx.fill();
        }
        if (scene !== "sala") {
            // Side table with tea
            ctx.fillStyle = "#a9805e"; roundRect(ctx, 330, 340, 58, 8, 3); ctx.fill(); ctx.fillRect(355, 348, 8, 90);
            drawCup(ctx, 344, 326, time);
        }
        // Rug
        ctx.fillStyle = "rgba(120,109,192,.35)"; ctx.beginPath(); ctx.ellipse(300, 452, 190, 18, 0, 0, Math.PI * 2); ctx.fill();
    }
}

function drawCup(ctx, x, y, time, steam = true) {
    ctx.fillStyle = "#ffffff";
    roundRect(ctx, x, y, 20, 16, 4); ctx.fill();
    ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.arc(x + 21, y + 8, 5, -Math.PI / 2, Math.PI / 2); ctx.stroke();
    ctx.fillStyle = "#e7604f"; ctx.fillRect(x + 2, y + 6, 16, 3);
    if (!steam) return;
    ctx.strokeStyle = "rgba(255,255,255,.75)"; ctx.lineWidth = 2;
    for (let i = 0; i < 2; i++) {
        ctx.beginPath();
        for (let k = 0; k <= 12; k++) {
            const sy = y - 4 - k * 1.8;
            const sx = x + 6 + i * 8 + Math.sin(time / 260 + k / 2 + i) * 3;
            if (k === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
        }
        ctx.globalAlpha = 0.7;
        ctx.stroke();
        ctx.globalAlpha = 1;
    }
}

function drawPerson(ctx, parts, prop, time, activityId) {
    const { near, far, head, neck, rot, forward } = parts;
    const foot = (ankle, color) => {
        const toe = seg(ankle, 15, 90 + rot);
        limb(ctx, [ankle, toe], 10, color);
    };
    // Far side (behind)
    limb(ctx, [far.hip, far.knee, far.ankle], 17, PALETTE.pantsFar);
    foot(far.ankle, PALETTE.slipper);
    limb(ctx, [far.shoulder, far.elbow, far.wrist], 12, PALETTE.cardiganFar);
    ctx.fillStyle = PALETTE.skinShade;
    ctx.beginPath(); ctx.arc(far.wrist.x, far.wrist.y, 5.5, 0, Math.PI * 2); ctx.fill();
    // Torso
    limb(ctx, [near.hip, near.shoulder], 34, PALETTE.cardigan);
    limb(ctx, [near.hip, { x: (near.hip.x + near.knee.x) / 2, y: (near.hip.y + near.knee.y) / 2 }], 24, PALETTE.pants);
    // Neck & head
    limb(ctx, [near.shoulder, neck], 11, PALETTE.skinShade);
    ctx.fillStyle = PALETTE.skin;
    ctx.beginPath(); ctx.arc(head.x, head.y, SIZE.head, 0, Math.PI * 2); ctx.fill();
    // Hair (back and top half, opposite to the face)
    ctx.fillStyle = PALETTE.hair;
    const back = rad(forward + 180);
    const angle = Math.atan2(Math.cos(back), Math.sin(back));
    ctx.beginPath();
    ctx.arc(head.x, head.y, SIZE.head + 2, angle - 1.3, angle + 1.95);
    ctx.closePath();
    ctx.fill();
    const bun = seg(head, SIZE.head + 4, forward + 205);
    ctx.fillStyle = PALETTE.hairShade;
    ctx.beginPath(); ctx.arc(bun.x, bun.y, 8, 0, Math.PI * 2); ctx.fill();
    // Face details
    const eye = seg(head, 9, forward + 28);
    ctx.fillStyle = "#3a3330";
    const sleeping = activityId === "dormindo";
    if (sleeping) {
        ctx.strokeStyle = "#3a3330"; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(eye.x, eye.y, 2.4, 0.2, Math.PI - 0.2); ctx.stroke();
    } else {
        ctx.beginPath(); ctx.arc(eye.x, eye.y, 1.8, 0, Math.PI * 2); ctx.fill();
    }
    ctx.strokeStyle = PALETTE.glasses; ctx.lineWidth = 1.6;
    ctx.beginPath(); ctx.arc(eye.x, eye.y, 5.2, 0, Math.PI * 2); ctx.stroke();
    const temple = seg(head, 6, forward + 150);
    ctx.beginPath(); ctx.moveTo(eye.x, eye.y); ctx.lineTo(temple.x, temple.y); ctx.stroke();
    const mouth = seg(head, 11, forward - 25);
    ctx.strokeStyle = "#b2655a"; ctx.lineWidth = 1.6;
    ctx.beginPath(); ctx.arc(mouth.x, mouth.y, 2.5, 0, Math.PI); ctx.stroke();
    const cheek = seg(head, 9, forward - 5);
    ctx.fillStyle = "rgba(232,128,118,.35)"; ctx.beginPath(); ctx.arc(cheek.x, cheek.y, 3.5, 0, Math.PI * 2); ctx.fill();
    // Near leg
    limb(ctx, [near.hip, near.knee, near.ankle], 18, PALETTE.pants);
    foot(near.ankle, PALETTE.slipper);
    // Props held in hands
    if (prop?.type === "book" && prop.open > 0.05) {
        const center = { x: (near.wrist.x + far.wrist.x) / 2 + 6, y: (near.wrist.y + far.wrist.y) / 2 - 8 };
        ctx.save();
        ctx.translate(center.x, center.y);
        ctx.rotate(-0.5);
        ctx.globalAlpha = Math.min(1, prop.open * 1.5);
        ctx.fillStyle = "#c0483c"; roundRect(ctx, -4, -20, 8, 40, 2); ctx.fill();
        ctx.fillStyle = "#fffaf0"; roundRect(ctx, -2, -18, 10 * prop.open + 2, 36, 2); ctx.fill();
        ctx.strokeStyle = "rgba(0,0,0,.15)"; ctx.lineWidth = 1;
        for (let i = -12; i <= 12; i += 6) { ctx.beginPath(); ctx.moveTo(1, i); ctx.lineTo(8 * prop.open, i); ctx.stroke(); }
        ctx.restore();
    }
    if (prop?.type === "cup" && prop.lifted >= 0.05) drawCup(ctx, near.wrist.x - 6, near.wrist.y - 12, time, prop.lifted < 0.5);
    // Near arm on top
    limb(ctx, [near.shoulder, near.elbow, near.wrist], 13, PALETTE.cardigan);
    ctx.fillStyle = PALETTE.skin;
    ctx.beginPath(); ctx.arc(near.wrist.x, near.wrist.y, 6, 0, Math.PI * 2); ctx.fill();
    // Blankets
    if (prop?.type === "blanket" && prop.amount > 0) {
        ctx.save();
        ctx.globalAlpha = Math.min(1, prop.amount);
        ctx.fillStyle = "#9cc5a1";
        ctx.beginPath();
        ctx.moveTo(near.hip.x - 6, near.hip.y - 14);
        ctx.quadraticCurveTo(near.knee.x, near.knee.y - 20, near.knee.x + 14, near.knee.y - 2);
        ctx.lineTo(near.ankle.x + 12, near.ankle.y - 18);
        ctx.lineTo(near.ankle.x - 16, near.ankle.y - 18);
        ctx.quadraticCurveTo(near.hip.x, near.knee.y + 22, near.hip.x - 8, near.hip.y + 10);
        ctx.fill();
        ctx.strokeStyle = "rgba(255,255,255,.5)"; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(near.hip.x + 10, near.hip.y); ctx.lineTo(near.knee.x, near.knee.y + 6); ctx.stroke();
        ctx.restore();
    }
    if (prop?.type === "blanket-bed" && prop.amount > 0) {
        ctx.save();
        ctx.globalAlpha = prop.amount;
        const quilt = ctx.createLinearGradient(0, 300, 0, 370);
        quilt.addColorStop(0, "#a7b9e8"); quilt.addColorStop(1, "#7f93cc");
        ctx.fillStyle = quilt;
        roundRect(ctx, near.hip.x - 32, Math.min(near.hip.y, near.ankle.y) - 18, near.ankle.x - near.hip.x + 66, 44, 14);
        ctx.fill();
        ctx.fillStyle = "rgba(255,255,255,.35)";
        for (let x = near.hip.x - 20; x < near.ankle.x + 20; x += 26) { ctx.beginPath(); ctx.arc(x, near.hip.y + 2, 3, 0, Math.PI * 2); ctx.fill(); }
        ctx.restore();
    }
    if (sleeping) {
        ctx.fillStyle = "#5d6fb1";
        ctx.font = "700 16px Manrope, sans-serif";
        for (let i = 0; i < 3; i++) {
            const phase = ((time / 1300) + i / 3) % 1;
            ctx.globalAlpha = 1 - phase;
            ctx.font = `700 ${12 + i * 4}px Manrope, sans-serif`;
            ctx.fillText("z", head.x + 18 + phase * 24 + i * 6, head.y - 18 - phase * 34 - i * 6);
        }
        ctx.globalAlpha = 1;
    }
}

const SKELETON = [["left_shoulder", "right_shoulder"], ["left_shoulder", "left_elbow"], ["left_elbow", "left_wrist"],
    ["right_shoulder", "right_elbow"], ["right_elbow", "right_wrist"], ["left_shoulder", "left_hip"],
    ["right_shoulder", "right_hip"], ["left_hip", "right_hip"], ["left_hip", "left_knee"], ["left_knee", "left_ankle"],
    ["right_hip", "right_knee"], ["right_knee", "right_ankle"]];

function drawSkeleton(ctx, keypoints) {
    const point = name => keypoints.find(item => item.name === name);
    ctx.save();
    ctx.fillStyle = "rgba(10,16,24,.35)";
    ctx.fillRect(0, 0, FRAME.width, FRAME.height);
    ctx.strokeStyle = "#39c99a"; ctx.lineWidth = 3; ctx.lineCap = "round";
    for (const [a, b] of SKELETON) {
        const p = point(a), q = point(b);
        ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y); ctx.stroke();
    }
    ctx.fillStyle = "#f8dfad";
    for (const keypoint of keypoints) { ctx.beginPath(); ctx.arc(keypoint.x, keypoint.y, 4, 0, Math.PI * 2); ctx.fill(); }
    ctx.restore();
}

export function drawScene(ctx, { segment, clock, body, prop, time, activityId, showSkeleton, transition }) {
    const hour = clock.getHours() + clock.getMinutes() / 60;
    drawRoom(ctx, segment.scene, hour, time, clock);
    const recline = segment.id === "descanso" ? (prop?.amount ?? 0) : 0;
    drawFurnitureBack(ctx, segment.scene, { recline });
    drawPerson(ctx, body.parts, prop, time, activityId);
    drawFurnitureFront(ctx, segment.scene, prop, time);
    // Iluminação do horário
    const [, , darkness] = skyFor(hour);
    if (darkness > 0) {
        ctx.fillStyle = `rgba(14,22,48,${darkness * (segment.scene === "quarto" ? 0.75 : 0.5)})`;
        ctx.fillRect(0, 0, FRAME.width, FRAME.height);
        if (segment.scene === "quarto") {
            const lamp = ctx.createRadialGradient(55, 300, 5, 55, 300, 220);
            lamp.addColorStop(0, "rgba(255,206,130,.32)"); lamp.addColorStop(1, "rgba(255,206,130,0)");
            ctx.fillStyle = lamp; ctx.fillRect(0, 100, 420, 380);
        }
    }
    if (hour >= 15.5 && hour < 18.5) {
        ctx.fillStyle = "rgba(255,170,90,.10)"; ctx.fillRect(0, 0, FRAME.width, FRAME.height);
    }
    if (showSkeleton) drawSkeleton(ctx, body.keypoints);
    const vignette = ctx.createRadialGradient(320, 240, 160, 320, 240, 420);
    vignette.addColorStop(0, "rgba(0,0,0,0)"); vignette.addColorStop(1, "rgba(0,0,0,.28)");
    ctx.fillStyle = vignette; ctx.fillRect(0, 0, FRAME.width, FRAME.height);
    if (transition > 0) {
        ctx.fillStyle = `rgba(12,18,26,${transition})`;
        ctx.fillRect(0, 0, FRAME.width, FRAME.height);
    }
}

// ---------------------------------------------------------------- controle
export function installRoutineSimulation(app) {
    const canvas = document.getElementById("routineCanvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const playButton = document.getElementById("routinePlay");
    const restartButton = document.getElementById("routineRestart");
    const skeletonToggle = document.getElementById("routineSkeleton");
    const el = id => document.getElementById(id);

    let running = false;
    let elapsed = 0;
    let lastTick = null;
    let frame = null;
    let currentIndex = -1;
    let centers = [];
    let peakDescent = 0;
    let sleepNoted = false;
    const day = new Date();

    function resizeCanvas() {
        const ratio = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        const width = Math.max(1, Math.round(rect.width * ratio));
        const height = Math.max(1, Math.round(rect.width * 0.75 * ratio));
        if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
        ctx.setTransform(width / FRAME.width, 0, 0, height / FRAME.height, 0, 0);
    }

    function descentSpeed(keypoints, now) {
        const get = name => keypoints.find(item => item.name === name);
        const y = (get("left_shoulder").y + get("left_hip").y) / 2 / FRAME.height;
        centers.push({ t: now, y });
        centers = centers.filter(item => now - item.t <= 700);
        let speed = 0;
        for (const item of centers) {
            const seconds = (now - item.t) / 1000;
            if (seconds > 0.15) speed = Math.max(speed, (y - item.y) / seconds);
        }
        return Math.max(0, speed);
    }

    function setReadout(id, text, state) {
        const node = el(id);
        if (!node) return;
        node.textContent = text;
        if (state) node.dataset.state = state;
    }

    function updateProgress(index) {
        el("routineProgressFill").style.width = `${Math.min(100, elapsed / TOTAL_MS * 100)}%`;
        document.querySelectorAll(".sim-step").forEach((step, i) => {
            step.dataset.state = i < index ? "done" : i === index ? "active" : "next";
        });
    }

    function enterSegment(index) {
        // Troca de cômodo = troca de câmera: a continuidade do movimento recomeça.
        app.resetMotionContinuity?.();
        app.resetFallDetection?.();
        app.resetActivity?.();
        app.postureHistory = [];
        centers = [];
        peakDescent = 0;
        currentIndex = index;
        const segment = SEGMENTS[index];
        el("routineCameraName").textContent = `CÂMERA · ${segment.camera.toUpperCase()}`;
        el("routineCaption").textContent = segment.caption;
        el("routineToast").hidden = true;
        sleepNoted = false;
    }

    function step(now) {
        if (!running) return;
        const delta = lastTick === null ? 0 : Math.min(250, now - lastTick);
        lastTick = now;
        elapsed += delta;
        if (elapsed >= TOTAL_MS) {
            finish();
            return;
        }
        const { segment, index, local } = segmentAt(elapsed);
        if (index !== currentIndex) enterSegment(index);
        const clock = clockAt(segment, local, day);
        app.routineClock = () => new Date(clock);
        const params = segment.pose(local);
        const body = buildBody(params);
        const pose = { score: 0.8, keypoints: body.keypoints };
        const result = app.processSyntheticFrame(pose, FRAME);
        const timestamp = Date.now();
        const speed = descentSpeed(body.keypoints, timestamp);
        peakDescent = Math.max(peakDescent, speed);

        const fadeIn = Math.max(0, 1 - local / 500);
        const fadeOut = Math.max(0, 1 - (segment.ms - local) / 500);
        resizeCanvas();
        drawScene(ctx, { segment, clock, body, prop: params.prop, time: now, activityId: result.activity?.id,
            showSkeleton: skeletonToggle?.checked, transition: Math.max(fadeIn, fadeOut) * 0.85 });

        el("routineClockText").textContent = clock.toLocaleTimeString("pt-BR", { hour: "2-digit", minute: "2-digit" });
        const posture = result.analysisResult.posture.state;
        setReadout("simPosture", posture === "INCONCLUSIVA" ? "Analisando…" : posture, posture === "INCONCLUSIVA" ? "processing" : "success");
        setReadout("simActivity", result.activity?.label ?? "—", "success");
        const severity = result.severity;
        const fallText = severity?.phase === "EM_ACOMPANHAMENTO" ? severity.label
            : severity?.phase === "OBSERVANDO_DESCIDA" ? "Descida rápida · analisando" : "Sem queda";
        setReadout("simFall", fallText, severity?.level >= 2 ? "error" : severity?.level === 1 ? "attention" : "success");
        const ratio = Math.min(1, speed / (DESCENT_LIMIT / 0.75));
        el("simDescentFill").style.width = `${ratio * 100}%`;
        el("simDescentFill").dataset.level = speed >= DESCENT_LIMIT ? "high" : speed >= DESCENT_LIMIT * 0.6 ? "mid" : "low";
        el("simDescentValue").textContent = `${speed.toFixed(2).replace(".", ",")} /s`;
        el("simDescentPeak").textContent = `pico nesta cena: ${peakDescent.toFixed(2).replace(".", ",")} /s · limite ${DESCENT_LIMIT.toFixed(2).replace(".", ",")} /s`;

        if (segment.id === "dormir" && result.activity?.id === "dormindo" && !sleepNoted) {
            sleepNoted = true;
            const alarm = severity?.level >= 1;
            const toast = el("routineToast");
            toast.hidden = false;
            toast.dataset.state = alarm ? "error" : "success";
            toast.querySelector("strong").textContent = alarm ? "Atenção: o deitar gerou alerta" : "Deitou-se para dormir · nenhum alerta";
            toast.querySelector("span").textContent =
                `A descida mais rápida foi ${peakDescent.toFixed(2).replace(".", ",")}/s, abaixo do limite de ${DESCENT_LIMIT.toFixed(2).replace(".", ",")}/s de uma queda. Deitar devagar não é queda.`;
            app.registerEvent?.(`Simulação: deitou-se para dormir com descida máxima de ${peakDescent.toFixed(2)}/s; ${alarm ? "houve alerta" : "nenhum alerta de queda"}.`);
        }
        updateProgress(index);
    }

    function start() {
        if (app.cameraStream) {
            app.registerEvent?.("Desligue a câmera para usar a simulação da rotina.");
            el("routineCaption").textContent = "Desligue a câmera para iniciar a simulação.";
            return;
        }
        app.stopFallDemo?.();
        if (elapsed === 0 || elapsed >= TOTAL_MS) {
            elapsed = 0;
            currentIndex = -1;
            app.routineLog?.clear();
        }
        app.routineSource = "simulação";
        running = true;
        lastTick = null;
        document.getElementById("startCamera").disabled = true;
        el("routineStage").dataset.state = "running";
        playButton.innerHTML = '<span aria-hidden="true">❚❚</span> Pausar';
        if (elapsed === 0) app.registerEvent?.("Simulação da rotina iniciada (gerada por computador, sem câmera).");
        // Temporizador fixo (e não requestAnimationFrame) para manter o ritmo mesmo em aparelhos lentos.
        frame = setInterval(() => step(performance.now()), 33);
    }

    function pause() {
        running = false;
        clearInterval(frame);
        app.interruptMotionContinuity?.();
        document.getElementById("startCamera").disabled = false;
        el("routineStage").dataset.state = "paused";
        playButton.innerHTML = '<span aria-hidden="true">▶</span> Continuar';
    }

    function restore() {
        app.routineSource = "câmera";
        app.routineClock = () => new Date();
        app.demoFrameSize = null;
    }

    function finish() {
        running = false;
        clearInterval(frame);
        elapsed = TOTAL_MS;
        updateProgress(SEGMENTS.length);
        app.resetActivity?.();
        restore();
        document.getElementById("startCamera").disabled = false;
        el("routineStage").dataset.state = "done";
        el("routineCaption").textContent = "Fim do dia simulado. Veja o registro da rotina abaixo.";
        playButton.innerHTML = '<span aria-hidden="true">▶</span> Simular de novo';
        app.registerEvent?.("Simulação da rotina concluída.");
    }

    app.stopRoutineSimulation = () => {
        if (!running && elapsed === 0) return;
        if (running) pause();
        elapsed = 0;
        currentIndex = -1;
        restore();
        el("routineStage").dataset.state = "idle";
        playButton.innerHTML = '<span aria-hidden="true">▶</span> Simular um dia';
    };

    playButton.addEventListener("click", () => (running ? pause() : start()));
    restartButton?.addEventListener("click", () => {
        if (running) pause();
        elapsed = 0;
        currentIndex = -1;
        start();
    });
    document.getElementById("startCamera")?.addEventListener("click", () => app.stopRoutineSimulation(), { capture: true });

    // Quadro inicial ilustrativo.
    function drawPreview() {
        resizeCanvas();
        const segment = SEGMENTS[0];
        const body = buildBody(POSES.seated(250), () => 0.5);
        drawScene(ctx, { segment, clock: clockAt(segment, 6_000, day), body, prop: null, time: 0, activityId: null, showSkeleton: false, transition: 0 });
    }
    drawPreview();
    window.addEventListener("resize", () => { if (!running) drawPreview(); });
    skeletonToggle?.addEventListener("change", () => { if (!running && elapsed === 0) drawPreview(); });
}
