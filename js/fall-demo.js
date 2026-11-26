// Demonstração de queda SEM câmera: gera pontos corporais por computador e os passa
// pelo mesmo caminho da câmera real (postura → movimento → detecção → gravidade).
// Serve para apresentar e conferir os níveis sem precisar cair de verdade.
// Tudo o que aparece durante a demonstração é simulado e a tela avisa isso.

const FRAME = Object.freeze({ width: 640, height: 480 });
const FRAME_INTERVAL_MS = 33;

// Ordem COCO (a mesma do MoveNet): nariz, olhos, orelhas, ombros, cotovelos, punhos, quadris, joelhos, tornozelos.
const STANDING = {
    nose: [320, 70], left_eye: [327, 63], right_eye: [313, 63], left_ear: [333, 68], right_ear: [307, 68],
    left_shoulder: [340, 110], right_shoulder: [300, 110], left_elbow: [350, 175], right_elbow: [290, 175],
    left_wrist: [352, 235], right_wrist: [288, 235], left_hip: [335, 240], right_hip: [305, 240],
    left_knee: [335, 340], right_knee: [305, 340], left_ankle: [335, 440], right_ankle: [305, 440]
};
const LYING = {
    nose: [150, 410], left_eye: [145, 405], right_eye: [145, 415], left_ear: [155, 400], right_ear: [155, 420],
    left_shoulder: [200, 420], right_shoulder: [200, 405], left_elbow: [250, 428], right_elbow: [250, 398],
    left_wrist: [300, 430], right_wrist: [300, 396], left_hip: [330, 420], right_hip: [330, 405],
    left_knee: [430, 421], right_knee: [430, 406], left_ankle: [530, 423], right_ankle: [530, 408]
};
const NAMES = Object.keys(STANDING);
const CONNECTIONS = [
    ["left_shoulder", "right_shoulder"], ["left_shoulder", "left_elbow"], ["left_elbow", "left_wrist"],
    ["right_shoulder", "right_elbow"], ["right_elbow", "right_wrist"], ["left_shoulder", "left_hip"],
    ["right_shoulder", "right_hip"], ["left_hip", "right_hip"], ["left_hip", "left_knee"],
    ["left_knee", "left_ankle"], ["right_hip", "right_knee"], ["right_knee", "right_ankle"]
];

// k = 0 em pé, k = 1 deitada; null = pessoa fora da imagem.
export const DEMO_SCENARIOS = Object.freeze({
    imovel: {
        label: "Cai e fica imóvel no chão",
        durationMs: 20_000,
        position: t => t < 2500 ? 0 : Math.min(1, (t - 2500) / 600)
    },
    levanta: {
        label: "Cai e levanta logo",
        durationMs: 12_000,
        position: t => t < 2500 ? 0 : t < 5000 ? Math.min(1, (t - 2500) / 600) : Math.max(0, 1 - (t - 5000) / 1500)
    },
    some: {
        label: "Cai e sai da imagem",
        durationMs: 14_000,
        position: t => t < 2500 ? 0 : t < 3000 ? Math.min(1, (t - 2500) / 600) : null
    }
});

export function demoPose(k, random = Math.random) {
    if (k === null) return null;
    const ease = k * k * (3 - 2 * k);
    return {
        score: 0.8,
        keypoints: NAMES.map(name => {
            const [ax, ay] = STANDING[name];
            const [bx, by] = LYING[name];
            return {
                name,
                x: ax + (bx - ax) * ease + (random() - 0.5) * 3,
                y: ay + (by - ay) * ease + (random() - 0.5) * 3,
                score: 0.8 + random() * 0.15
            };
        })
    };
}

export function installFallDemo(app) {
    const button = document.getElementById("demoFall");
    const select = document.getElementById("demoScenario");
    const banner = document.getElementById("demoBanner");
    if (!button || !select) return;

    let timer = null;
    let startedAt = 0;
    let scenario = null;

    function drawDemoPose(pose) {
        const canvas = app.poseCanvas;
        const rect = app.videoContainer.getBoundingClientRect();
        canvas.width = Math.max(1, Math.round(rect.width));
        canvas.height = Math.max(1, Math.round(rect.height));
        const context = app.poseContext;
        context.clearRect(0, 0, canvas.width, canvas.height);
        if (!pose) return;
        const scale = Math.min(canvas.width / FRAME.width, canvas.height / FRAME.height);
        const offsetX = (canvas.width - FRAME.width * scale) / 2;
        const offsetY = (canvas.height - FRAME.height * scale) / 2;
        const point = name => {
            const keypoint = pose.keypoints.find(item => item.name === name);
            return { x: offsetX + keypoint.x * scale, y: offsetY + keypoint.y * scale };
        };
        context.strokeStyle = "#39c99a";
        context.lineWidth = 3;
        context.lineCap = "round";
        for (const [first, second] of CONNECTIONS) {
            const a = point(first);
            const b = point(second);
            context.beginPath();
            context.moveTo(a.x, a.y);
            context.lineTo(b.x, b.y);
            context.stroke();
        }
        context.fillStyle = "#f8dfad";
        for (const name of NAMES.slice(5)) {
            const p = point(name);
            context.beginPath();
            context.arc(p.x, p.y, 5, 0, Math.PI * 2);
            context.fill();
        }
    }

    // Mesmo fluxo de pose.js para um quadro com (ou sem) pessoa.
    function processFrame(pose) {
        const timestamp = Date.now();
        const analysisResult = app.createAnalysisResult(pose, timestamp);
        app.recordAnalysisResult(analysisResult);
        const motionSample = app.recordMotionInference({
            pose, analysisResult, videoWidth: FRAME.width, videoHeight: FRAME.height
        });
        app.evaluateFallSeverity(app.evaluateFallDetection(motionSample), motionSample);
        drawDemoPose(pose);
        if (!pose) {
            app.updatePostureHistory(null);
            app.setPoseStatus("Procurando uma pessoa...");
            return;
        }
        const posture = analysisResult.posture.state === "INCONCLUSIVA" ? null : analysisResult.posture.state;
        const stablePosture = app.updatePostureHistory(posture);
        if (stablePosture) app.recordPositionChange(stablePosture, timestamp);
        if (!posture) app.setPoseStatus("Não foi possível identificar a posição.");
        else app.setPoseStatus(stablePosture ? `Posição identificada: ${stablePosture}` : "Pessoa detectada — analisando posição...");
    }

    function tick() {
        const elapsed = Date.now() - startedAt;
        if (elapsed > scenario.durationMs) {
            stop("concluída");
            return;
        }
        processFrame(demoPose(scenario.position(elapsed)));
    }

    function start() {
        if (app.cameraStream) {
            app.registerEvent?.("Desligue a câmera para usar a demonstração de queda.");
            return;
        }
        scenario = DEMO_SCENARIOS[select.value] ?? DEMO_SCENARIOS.imovel;
        app.demoActive = true;
        app.demoFrameSize = FRAME;
        app.resetMotionContinuity?.();
        app.resetFallDetection?.();
        app.postureHistory = [];
        document.getElementById("placeholder").style.display = "none";
        document.getElementById("startCamera").disabled = true;
        select.disabled = true;
        if (banner) banner.hidden = false;
        button.textContent = "■ Parar demonstração";
        app.setAiStatus?.("Demonstração (simulada)");
        app.registerEvent?.(`Demonstração iniciada (simulada, sem câmera): ${scenario.label}.`);
        startedAt = Date.now();
        timer = setInterval(tick, FRAME_INTERVAL_MS);
    }

    function stop(reason = "interrompida") {
        clearInterval(timer);
        timer = null;
        app.demoActive = false;
        app.demoFrameSize = null;
        app.poseContext.clearRect(0, 0, app.poseCanvas.width, app.poseCanvas.height);
        document.getElementById("placeholder").style.display = "flex";
        document.getElementById("startCamera").disabled = false;
        select.disabled = false;
        if (banner) banner.hidden = true;
        button.textContent = "▶ Simular queda";
        app.registerEvent?.(`Demonstração ${reason}. O resultado final continua no cartão de gravidade.`);
    }

    button.addEventListener("click", () => (timer ? stop() : start()));
    // Ligar a câmera encerra uma demonstração em andamento.
    document.getElementById("startCamera")?.addEventListener("click", () => { if (timer) stop(); }, { capture: true });
}
