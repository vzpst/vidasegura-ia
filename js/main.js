import { installUI } from "./ui.js";
import { installCamera } from "./camera.js";
import { installPose } from "./pose.js";
import { installPosture } from "./posture.js";
import { installPostureModel } from "./posture-model.js";
import { installVisibility } from "./visibility.js";
import { installHistory } from "./history.js";
import { installSensors } from "./sensors.js";
import { installMotionAnalysis } from "./motion-analysis.js";
import { installFallDetection } from "./fall-detection.js";
import { installFallSeverity } from "./fall-severity.js";

const app = {
    camera: document.getElementById("camera"),
    poseCanvas: document.getElementById("poseCanvas"),
    videoContainer: document.querySelector(".video-container"),
    flipCameraButton: document.getElementById("flipCamera"),
    themeToggle: document.getElementById("themeToggle"),
    cameraStream: null,
    poseDetector: null,
    cameraFacingMode: "user",
    cameraRequestedFacingMode: null,
    cameraDeviceId: null,
    cameraOrientationKnown: false,
    availableCameraCount: null,
    cameraSwitching: false,
    poseAnimationFrame: null,
    detectionInProgress: false,
    detectionPromise: null,
    cameraStartingSession: null,
    cameraStopping: false,
    detectionSession: 0,
    postureHistory: [],
    analysisHistory: [],
    positionHistory: [],
    latestAnalysisResult: null,
    lastPersonDetectedAt: null,
    detectionStartedAt: null,
    sensorActive: false,
    previousAcceleration: null,
    lastAlert: 0
};
app.poseContext = app.poseCanvas.getContext("2d");
app.videoContainer.style.position = "relative";
app.poseCanvas.style.pointerEvents = "none";
app.videoContainer.appendChild(app.flipCameraButton);
app.flipCameraButton.setAttribute("aria-label", "Trocar entre câmera frontal e traseira");
app.flipCameraButton.title = "Trocar câmera";
app.flipCameraButton.innerHTML = '<svg aria-hidden="true" viewBox="0 0 24 24" width="23" height="23" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 7h-3l-2-2H9L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2Z"/><circle cx="12" cy="13" r="3"/><path d="M5 3 2 6l3 3M19 21l3-3-3-3"/></svg>';
Object.assign(app.flipCameraButton.style, { position: "absolute", top: "12px", right: "12px", zIndex: "5", width: "58px", height: "58px", minWidth: "58px", minHeight: "58px", padding: "0", display: "grid", placeItems: "center", borderRadius: "50%", border: "2px solid rgba(255,255,255,.9)", background: "rgba(18,28,38,.88)", color: "#fff", boxShadow: "0 3px 12px rgba(0,0,0,.45)", cursor: "pointer", pointerEvents: "auto", touchAction: "manipulation" });
app.flipCameraButton.addEventListener("pointerdown", () => { app.flipCameraButton.style.transform = "scale(.94)"; });
for (const eventName of ["pointerup", "pointercancel", "pointerleave"]) app.flipCameraButton.addEventListener(eventName, () => { app.flipCameraButton.style.transform = ""; });

installUI(app);
installHistory(app);
installPosture(app);
installPostureModel(app);
// Se este texto não aparecer ao lado da versão, o navegador está rodando JavaScript antigo (Ctrl+F5).
app.postureModelReady.then(() => {
    const version = document.getElementById("appVersion");
    if (version) version.textContent += ` · JS atualizado · modelo de postura ${app.postureModelStatus}`;
});
installVisibility(app);
installMotionAnalysis(app);
installFallDetection(app);
installFallSeverity(app);
installPose(app);
installCamera(app);
installSensors(app);

app.applyTheme(document.documentElement.dataset.theme || "light", false);
app.themeToggle.addEventListener("click", () => app.applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));
document.getElementById("startCamera").addEventListener("click", app.startCamera);
document.getElementById("stopCamera").addEventListener("click", app.stopCamera);
document.getElementById("flipCamera").addEventListener("click", app.switchCamera);
document.getElementById("startSensors").addEventListener("click", app.startSensors);
document.getElementById("stopSensors").addEventListener("click", app.stopSensors);
document.getElementById("fallAcknowledge").addEventListener("click", () => app.acknowledgeFallSeverity());
document.getElementById("clearLog").addEventListener("click", () => { document.getElementById("eventLog").innerHTML = '<p class="empty-state">Nenhum evento registrado nesta sessão.</p>'; });

const pageTitles = {
    overview: ["Visão geral", "Acompanhe o monitoramento e o estado do sistema em tempo real."],
    monitoring: ["Monitoramento", "Visualização da câmera e análise de postura."],
    sensors: ["Sensores", "Leituras recebidas dos sensores deste dispositivo."],
    history: ["Histórico de posições", "Mudanças estabilizadas entre posturas corporais."],
    settings: ["Configurações", "Preferências disponíveis em uma etapa futura."]
};
document.querySelectorAll(".nav-link").forEach(link => link.addEventListener("click", () => {
    const title = pageTitles[link.getAttribute("href")?.slice(1)];
    if (!title) return;
    document.getElementById("pageTitle").textContent = title[0];
    document.getElementById("pageSubtitle").textContent = title[1];
    document.querySelectorAll(".nav-link").forEach(item => {
        item.classList.toggle("is-active", item === link);
        if (item === link) item.setAttribute("aria-current", "page");
        else item.removeAttribute("aria-current");
    });
}));

window.addEventListener("beforeunload", () => {
    app.detectionSession++;
    app.resetFallDetection();
    app.cameraStartingSession = null;
    app.cameraStopping = true;
    if (app.poseAnimationFrame !== null) {
        cancelAnimationFrame(app.poseAnimationFrame);
        app.poseAnimationFrame = null;
    }
    if (app.cameraStream) {
        app.cameraStream.getTracks().forEach(track => track.stop());
        app.cameraStream = null;
    }
    app.cameraDeviceId = null;
    app.cameraOrientationKnown = false;
    app.camera.srcObject = null;
    if (app.sensorActive) {
        window.removeEventListener("devicemotion", app.handleMotion);
        app.sensorActive = false;
    }
    const detector = app.poseDetector;
    const inference = app.detectionPromise;
    app.poseDetector = null;
    if (detector) Promise.resolve(inference).catch(() => {}).then(() => detector.dispose()).catch(() => {});
});
