import { installPosture } from "../../js/posture.js";

const KEYPOINT_NAMES = [
    "left_shoulder", "right_shoulder", "left_hip", "right_hip",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
];
const VISIBILITY_THRESHOLD = 0.25;
const app = { postureHistory: [] };
installPosture(app);

const elements = {
    subject: document.getElementById("subjectId"),
    session: document.getElementById("sessionId"),
    posture: document.getElementById("postureLabel"),
    viewpoint: document.getElementById("viewpointLabel"),
    challenge: document.getElementById("challengeLabel"),
    facing: document.getElementById("cameraFacing"),
    consent: document.getElementById("consent"),
    video: document.getElementById("preview"),
    previewWrap: document.getElementById("previewWrap"),
    overlay: document.getElementById("poseOverlay"),
    debug: document.getElementById("debugPanel"),
    status: document.getElementById("status"),
    quality: document.getElementById("sampleQuality"),
    start: document.getElementById("startCamera"),
    stop: document.getElementById("stopCamera"),
    capture: document.getElementById("captureSample"),
    download: document.getElementById("downloadJsonl"),
    clear: document.getElementById("clearSamples"),
    count: document.getElementById("sampleCount"),
    rows: document.getElementById("sampleRows")
};

let stream = null;
let detector = null;
let running = false;
let inferenceBusy = false;
let latestPose = null;
let animationFrame = null;
let lastInferenceAt = 0;
let sampleSequence = 0;
const samples = [];

function setStatus(message) {
    elements.status.textContent = message;
}

// Diagnóstico visível na própria página: permite descobrir no celular por que o
// esqueleto não aparece sem depender de chrome://inspect. Não registra coordenadas.
const debugState = { stage: "aguardando início", backend: "-", video: "-", canvas: "-", fps: "-", pose: "-", errors: [] };
let inferenceCount = 0;
let fpsWindowStartedAt = performance.now();

function renderDebug() {
    if (!elements.debug) return;
    const lines = [
        `Etapa: ${debugState.stage}`,
        `Contexto seguro (HTTPS/localhost): ${window.isSecureContext ? "sim" : "NÃO — a câmera não funciona"}`,
        `TF.js: ${window.tf ? window.tf.version?.tfjs ?? "carregado" : "NÃO CARREGADO"} · Pose Detection: ${window.poseDetection ? "carregado" : "NÃO CARREGADO"}`,
        `Backend: ${debugState.backend} · Vídeo: ${debugState.video} · Canvas: ${debugState.canvas}`,
        `Inferências/s: ${debugState.fps} · Pose: ${debugState.pose}`
    ];
    elements.debug.replaceChildren(document.createTextNode(lines.join("\n")));
    for (const message of debugState.errors.slice(-5)) {
        const line = document.createElement("div");
        line.className = "error";
        line.textContent = `ERRO: ${message}`;
        elements.debug.append(line);
    }
}

function reportError(message) {
    debugState.errors.push(String(message));
    renderDebug();
}

window.addEventListener("error", event => reportError(`${event.message} (${event.filename?.split("/").pop() ?? "?"}:${event.lineno ?? "?"})`));
window.addEventListener("unhandledrejection", event => reportError(event.reason?.message ?? event.reason));

function setPreviewFacing() {
    elements.previewWrap.classList.toggle("mirrored", elements.facing.value === "user");
}

const SKELETON_CONNECTIONS = [
    ["left_shoulder", "right_shoulder"], ["left_shoulder", "left_elbow"], ["left_elbow", "left_wrist"],
    ["right_shoulder", "right_elbow"], ["right_elbow", "right_wrist"], ["left_shoulder", "left_hip"],
    ["right_shoulder", "right_hip"], ["left_hip", "right_hip"], ["left_hip", "left_knee"],
    ["left_knee", "left_ankle"], ["right_hip", "right_knee"], ["right_knee", "right_ankle"]
];

// Desenha o esqueleto sobre o vídeo. O canvas fica dentro do mesmo contêiner
// espelhado do vídeo, então os pontos originais do MoveNet são usados sem inversão.
function drawSkeleton(pose) {
    const canvas = elements.overlay;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(1, Math.round(rect.width));
    const height = Math.max(1, Math.round(rect.height));
    if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
    }
    debugState.canvas = `${width}x${height}`;
    const context = canvas.getContext("2d");
    context.clearRect(0, 0, width, height);
    const videoWidth = elements.video.videoWidth;
    const videoHeight = elements.video.videoHeight;
    if (!pose || !videoWidth || !videoHeight) return;

    // Mesmo cálculo de object-fit: contain usado pelo <video>.
    const scale = Math.min(width / videoWidth, height / videoHeight);
    const offsetX = (width - videoWidth * scale) / 2;
    const offsetY = (height - videoHeight * scale) / 2;
    const byName = new Map(pose.keypoints.map(point => [point.name, point]));
    const visible = point => point && Number.isFinite(point.x) && Number.isFinite(point.y) &&
        (point.score ?? 0) >= VISIBILITY_THRESHOLD;
    const toCanvas = point => [offsetX + point.x * scale, offsetY + point.y * scale];

    context.lineWidth = 3;
    context.lineCap = "round";
    context.strokeStyle = "#39c99a";
    for (const [first, second] of SKELETON_CONNECTIONS) {
        const a = byName.get(first);
        const b = byName.get(second);
        if (!visible(a) || !visible(b)) continue;
        context.beginPath();
        context.moveTo(...toCanvas(a));
        context.lineTo(...toCanvas(b));
        context.stroke();
    }
    for (const point of pose.keypoints) {
        if (!visible(point)) continue;
        context.fillStyle = KEYPOINT_NAMES.includes(point.name) ? "#f8dfad" : "#8fb8ff";
        context.beginPath();
        context.arc(...toCanvas(point), 5, 0, Math.PI * 2);
        context.fill();
    }
}

function updateCaptureAvailability() {
    const identifiersValid = elements.subject.checkValidity() && elements.session.checkValidity();
    elements.capture.disabled = !running || !latestPose || !identifiersValid || !elements.consent.checked;
}

function pointToRecord(point, width, height) {
    const score = Number.isFinite(point?.score) && point.score >= 0 && point.score <= 1
        ? point.score
        : null;
    if (!point || !Number.isFinite(point.x) || !Number.isFinite(point.y) || score === null ||
        point.x < 0 || point.y < 0 || point.x > width || point.y > height) {
        return { x: null, y: null, score, validity: "missing" };
    }
    const x = point.x / width;
    const y = point.y / height;
    return {
        x,
        y,
        score,
        validity: score >= VISIBILITY_THRESHOLD ? "observed" : "low_confidence"
    };
}

function makeSample(pose) {
    const width = elements.video.videoWidth;
    const height = elements.video.videoHeight;
    if (!width || !height) throw new Error("A câmera ainda não informou as dimensões do vídeo.");
    const points = Object.fromEntries(KEYPOINT_NAMES.map(name => {
        const point = pose.keypoints.find(candidate => candidate.name === name) ?? null;
        return [name, pointToRecord(point, width, height)];
    }));
    const observedCount = Object.values(points).filter(point => point.validity === "observed").length;
    const geometricPrediction = app.classifyPose(pose);
    const predictionMap = { "EM PÉ": "standing", "SENTADA": "sitting", "DEITADA": "lying" };
    sampleSequence += 1;
    return {
        schema_version: 1,
        sample_id: `sample_${Date.now()}_${String(sampleSequence).padStart(4, "0")}`,
        subject_id: elements.subject.value.trim(),
        session_id: elements.session.value.trim(),
        timestamp_ms: Date.now(),
        image_width: width,
        image_height: height,
        data_origin: "camera",
        capture_method: "movenet_singlepose_lightning",
        posture_label: elements.posture.value,
        viewpoint_label: elements.viewpoint.value,
        challenge_label: elements.challenge.value,
        geometric_prediction: predictionMap[geometricPrediction] ?? "inconclusive",
        keypoints: points,
        observedCount
    };
}

function renderSamples() {
    elements.count.textContent = `${samples.length} amostra${samples.length === 1 ? "" : "s"}. Nada é persistido ao fechar ou recarregar a página.`;
    elements.download.disabled = samples.length === 0;
    elements.clear.disabled = samples.length === 0;
    elements.rows.replaceChildren();
    if (samples.length === 0) {
        const row = document.createElement("tr");
        const cell = document.createElement("td");
        cell.colSpan = 6;
        cell.className = "muted";
        cell.textContent = "Nenhuma amostra capturada.";
        row.append(cell);
        elements.rows.append(row);
        return;
    }
    samples.forEach((sample, index) => {
        const row = document.createElement("tr");
        const values = [
            sample.sample_id,
            sample.posture_label,
            sample.viewpoint_label,
            sample.challenge_label,
            `${sample.observedCount}/12`
        ];
        values.forEach(value => {
            const cell = document.createElement("td");
            cell.textContent = value;
            row.append(cell);
        });
        const actionCell = document.createElement("td");
        const discard = document.createElement("button");
        discard.type = "button";
        discard.textContent = "Descartar";
        discard.setAttribute("aria-label", `Descartar amostra ${sample.sample_id}`);
        discard.addEventListener("click", () => {
            samples.splice(index, 1);
            renderSamples();
        });
        actionCell.append(discard);
        row.append(actionCell);
        elements.rows.append(row);
    });
}

async function inferLoop() {
    if (!running) return;
    animationFrame = requestAnimationFrame(inferLoop);
    const now = performance.now();
    if (inferenceBusy || now - lastInferenceAt < 90 || elements.video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
    inferenceBusy = true;
    lastInferenceAt = now;
    try {
        const poses = await detector.estimatePoses(elements.video, { maxPoses: 1, flipHorizontal: false });
        latestPose = poses?.[0]?.keypoints?.length ? poses[0] : null;
        drawSkeleton(latestPose);
        inferenceCount += 1;
        const elapsed = performance.now() - fpsWindowStartedAt;
        if (elapsed >= 1000) {
            debugState.fps = (inferenceCount * 1000 / elapsed).toFixed(1);
            inferenceCount = 0;
            fpsWindowStartedAt = performance.now();
        }
        debugState.stage = "inferindo";
        debugState.video = `${elements.video.videoWidth}x${elements.video.videoHeight}`;
        debugState.pose = latestPose ? "detectada" : "nenhuma pessoa";
        renderDebug();
        const observed = latestPose
            ? KEYPOINT_NAMES.filter(name => {
                const point = latestPose.keypoints.find(candidate => candidate.name === name);
                return point && Number.isFinite(point.x) && Number.isFinite(point.y) &&
                    Number.isFinite(point.score) && point.score >= VISIBILITY_THRESHOLD;
            }).length
            : 0;
        elements.quality.textContent = latestPose
            ? `${observed}/12 articulações com confiança ≥ ${VISIBILITY_THRESHOLD}. Amostra mantida apenas se você capturar.`
            : "Pessoa ou pose não detectada. Nenhuma amostra foi criada.";
        updateCaptureAvailability();
    } catch (error) {
        latestPose = null;
        elements.quality.textContent = "Falha nesta inferência; não foi interpretada como ausência de movimento.";
        setStatus(`Falha de inferência: ${error.message}`);
        reportError(`inferência: ${error.message}`);
        updateCaptureAvailability();
    } finally {
        inferenceBusy = false;
    }
}

async function stopCamera() {
    running = false;
    if (animationFrame !== null) cancelAnimationFrame(animationFrame);
    animationFrame = null;
    latestPose = null;
    if (stream) stream.getTracks().forEach(track => track.stop());
    stream = null;
    elements.video.srcObject = null;
    drawSkeleton(null);
    if (detector) await detector.dispose().catch(() => {});
    detector = null;
    debugState.stage = "câmera desligada";
    renderDebug();
    elements.start.disabled = false;
    elements.stop.disabled = true;
    elements.quality.textContent = "Câmera desligada.";
    updateCaptureAvailability();
    setStatus("Câmera desligada. As amostras em memória permanecem até exportar ou descartar.");
}

async function startCamera() {
    if (!navigator.mediaDevices?.getUserMedia) {
        setStatus("A câmera requer localhost/HTTPS e suporte a getUserMedia neste navegador.");
        return;
    }
    elements.start.disabled = true;
    setPreviewFacing();
    try {
        if (!window.tf || !window.poseDetection) {
            throw new Error("TensorFlow.js ou Pose Detection não carregou (verifique a internet/CDN).");
        }
        debugState.stage = "preparando TensorFlow.js";
        renderDebug();
        await window.tf.ready();
        debugState.backend = window.tf.getBackend?.() ?? "?";
        debugState.stage = "pedindo permissão da câmera";
        renderDebug();
        const streamCandidate = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: { ideal: elements.facing.value } },
            audio: false
        });
        elements.video.srcObject = streamCandidate;
        await elements.video.play();
        debugState.video = `${elements.video.videoWidth}x${elements.video.videoHeight}`;
        debugState.stage = "carregando MoveNet";
        renderDebug();
        const model = window.poseDetection.SupportedModels.MoveNet;
        const modelType = window.poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING;
        const detectorCandidate = await window.poseDetection.createDetector(model, { modelType });
        stream = streamCandidate;
        detector = detectorCandidate;
        running = true;
        elements.stop.disabled = false;
        setStatus("Câmera ativa. Confira as anotações; a inferência usa pontos originais sem espelhamento.");
        animationFrame = requestAnimationFrame(inferLoop);
    } catch (error) {
        if (elements.video.srcObject) {
            elements.video.srcObject.getTracks().forEach(track => track.stop());
            elements.video.srcObject = null;
        }
        if (detector) await detector.dispose().catch(() => {});
        detector = null;
        stream = null;
        elements.start.disabled = false;
        setStatus(`Não foi possível iniciar o MoveNet/câmera: ${error.message}`);
        debugState.stage = "falhou ao iniciar";
        reportError(`início: ${error.name ?? ""} ${error.message}`);
    }
    updateCaptureAvailability();
}

elements.start.addEventListener("click", startCamera);
elements.stop.addEventListener("click", stopCamera);
elements.facing.addEventListener("change", () => {
    setPreviewFacing();
    if (running) {
        stopCamera().then(startCamera);
    }
});
for (const element of [elements.subject, elements.session, elements.consent]) {
    element.addEventListener("input", updateCaptureAvailability);
    element.addEventListener("change", updateCaptureAvailability);
}
elements.capture.addEventListener("click", () => {
    if (!latestPose || !elements.consent.checked) return;
    try {
        const sample = makeSample(latestPose);
        samples.push(sample);
        renderSamples();
        setStatus(sample.observedCount < 6
            ? "Amostra capturada com baixa visibilidade; revise e descarte se estiver duvidosa."
            : "Amostra capturada localmente.");
    } catch (error) {
        setStatus(error.message);
    }
});
elements.download.addEventListener("click", () => {
    if (!samples.length) return;
    const jsonl = samples.map(({ observedCount, ...sample }) => JSON.stringify(sample)).join("\n") + "\n";
    const url = URL.createObjectURL(new Blob([jsonl], { type: "application/x-ndjson;charset=utf-8" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = `vidasegura_${elements.subject.value.trim()}_${elements.session.value.trim()}.jsonl`;
    link.click();
    URL.revokeObjectURL(url);
    setStatus("JSONL baixado. O arquivo contém keypoints normalizados, sem imagens ou vídeo.");
});
elements.clear.addEventListener("click", () => {
    if (!window.confirm("Descartar todas as amostras que ainda estão nesta página?")) return;
    samples.length = 0;
    renderSamples();
    setStatus("Amostras em memória descartadas.");
});

window.addEventListener("beforeunload", () => {
    running = false;
    if (animationFrame !== null) cancelAnimationFrame(animationFrame);
    if (stream) stream.getTracks().forEach(track => track.stop());
    if (detector) detector.dispose().catch(() => {});
});

renderSamples();
setPreviewFacing();
renderDebug();
