// Funções extraídas do fluxo original; o estado e as dependências são injetados por main.js.
export function installCamera(app) {
    app.updateCameraAvailability = async function(session) {
    try {
        if (!navigator.mediaDevices?.enumerateDevices) {
            app.availableCameraCount = null;
        } else {
            const devices = await navigator.mediaDevices.enumerateDevices();
            if (session !== app.detectionSession) return;
            app.availableCameraCount = devices.filter(device => device.kind === "videoinput").length;
        }
    } catch (error) {
        // A enumeração pode ser bloqueada por alguns navegadores; nesse caso o botão fica indisponível.
        app.availableCameraCount = null;
    }

    const flipButton = document.getElementById("flipCamera");
    flipButton.disabled = !app.cameraStream || app.availableCameraCount === 1 || app.cameraSwitching ||
        app.cameraStartingSession !== null;
    if (app.cameraStream && app.camera.srcObject === app.cameraStream &&
        app.camera.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && !app.camera.paused) {
        const status = app.availableCameraCount === 1 ? "Ativa — câmera única" : "Conectada";
        app.setCameraStatus(app.cameraOrientationKnown ? status : `${status} · orientação não identificada`);
    }

    };
    app.getTrackFacingMode = function(track) {
    const settings = track?.getSettings?.() ?? {};
    if (settings.facingMode === "user" || settings.facingMode === "environment") return settings.facingMode;
    const label = track?.label ?? "";
    if (/front|user|frontal|\bface\b/i.test(label)) return "user";
    if (/back|rear|environment|traseir/i.test(label)) return "environment";
    return null;

    };
    app.identifyTrackFacingMode = async function(track) {
    const knownFacing = app.getTrackFacingMode(track);
    if (knownFacing) return knownFacing;
    const deviceId = track?.getSettings?.().deviceId;
    if (!deviceId || !navigator.mediaDevices?.enumerateDevices) return null;
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const label = devices.find(device => device.kind === "videoinput" && device.deviceId === deviceId)?.label ?? "";
        if (/front|user|frontal|\bface\b/i.test(label)) return "user";
        if (/back|rear|environment|traseir/i.test(label)) return "environment";
    } catch (error) {
        // Sem informação verificável, mantém a orientação desconhecida.
    }
    return null;

    };
    app.requestCameraStream = async function(facingMode, previousDeviceId) {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { exact: facingMode } }, audio: false });
        const track = stream.getVideoTracks()[0];
        const settings = track?.getSettings?.() ?? {};
        const actualFacing = app.getTrackFacingMode(track);
        if (previousDeviceId && settings.deviceId === previousDeviceId) {
            stream.getTracks().forEach(item => item.stop());
            throw new DOMException("O navegador manteve a câmera anterior.", "OverconstrainedError");
        }
        if (actualFacing && actualFacing !== facingMode) {
            stream.getTracks().forEach(item => item.stop());
            throw new DOMException("O navegador abriu uma câmera diferente da solicitada.", "OverconstrainedError");
        }
        return { stream, deviceId: settings.deviceId ?? null, facingMode: actualFacing ?? facingMode, orientationKnown: true };
    } catch (error) {
        if (!(["OverconstrainedError", "ConstraintNotSatisfiedError", "NotFoundError", "TypeError"].includes(error.name))) throw error;
    }

    if (!navigator.mediaDevices?.enumerateDevices) {
        throw new Error(`Não foi possível identificar com segurança a câmera ${facingMode === "user" ? "frontal" : "traseira"}.`);
    }
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cameras = devices.filter(device => device.kind === "videoinput" &&
        (!previousDeviceId || device.deviceId !== previousDeviceId));
    const pattern = facingMode === "user" ? /front|user|frontal|\bface\b/i : /back|rear|environment|traseir/i;
    const matches = cameras.filter(device => pattern.test(device.label));
    let target = matches.length === 1 ? matches[0] : null;
    const uniqueAlternate = !target && matches.length === 0 && cameras.length === 1;
    if (uniqueAlternate) target = cameras[0];
    if (!target?.deviceId) {
        if (cameras.length === 0 && previousDeviceId) {
            throw new Error("Não foi encontrada outra câmera disponível para alternar.");
        }
        throw new Error(`Não foi possível identificar com segurança a câmera ${facingMode === "user" ? "frontal" : "traseira"}. Confira os nomes das câmeras do dispositivo.`);
    }

    const stream = await navigator.mediaDevices.getUserMedia({ video: { deviceId: { exact: target.deviceId } }, audio: false });
    const track = stream.getVideoTracks()[0];
    const settings = track?.getSettings?.() ?? {};
    const actualFacing = app.getTrackFacingMode(track);
    if (settings.deviceId && settings.deviceId === previousDeviceId) {
        stream.getTracks().forEach(item => item.stop());
        throw new Error("O navegador retornou a câmera anterior em vez da alternativa solicitada.");
    }
    if (actualFacing && actualFacing !== facingMode && !uniqueAlternate) {
        stream.getTracks().forEach(item => item.stop());
        throw new Error("A câmera encontrada não corresponde à orientação solicitada.");
    }
    return {
        stream,
        deviceId: settings.deviceId ?? target.deviceId,
        facingMode: actualFacing ?? (pattern.test(target.label) ? facingMode : null),
        orientationKnown: Boolean(actualFacing || pattern.test(target.label))
    };

    };
    app.waitForVideoEvent = function(eventName, stream) {
    return new Promise((resolve, reject) => {
        const readyState = eventName === "loadedmetadata" ? HTMLMediaElement.HAVE_METADATA : HTMLMediaElement.HAVE_CURRENT_DATA;
        let timeout;
        const finish = error => {
            clearTimeout(timeout);
            app.camera.removeEventListener(eventName, onReady);
            app.camera.removeEventListener("error", onError);
            if (error) reject(error);
            else resolve();
        };
        const onReady = () => {
            if (app.camera.srcObject !== stream) finish(new Error("O stream foi substituído durante a inicialização."));
            else if (eventName === "loadedmetadata" && (!app.camera.videoWidth || !app.camera.videoHeight)) finish(new Error("A câmera não forneceu dimensões válidas."));
            else finish();
        };
        const onError = () => finish(new Error("O elemento de vídeo não conseguiu ler o stream da câmera."));
        timeout = setTimeout(() => finish(new Error("A câmera não iniciou o vídeo no tempo esperado.")), 8000);
        app.camera.addEventListener(eventName, onReady, { once: true });
        app.camera.addEventListener("error", onError, { once: true });
        if (app.camera.srcObject === stream && app.camera.readyState >= readyState) queueMicrotask(onReady);
    });

    };
    app.attachCameraStream = async function(stream, session) {
    if (session !== app.detectionSession || app.cameraStopping) return false;
    app.cameraStream = stream;
    app.camera.pause();
    app.camera.srcObject = null;
    app.camera.style.display = "block";
    app.camera.style.transform = app.cameraFacingMode === "user" ? "scaleX(-1)" : "none";
    app.poseCanvas.style.display = "block";
    document.getElementById("placeholder").style.display = "none";
    const metadataReady = app.waitForVideoEvent("loadedmetadata", stream);
    app.camera.srcObject = stream;
    await metadataReady;
    if (session !== app.detectionSession || app.cameraStopping || app.cameraStream !== stream) return false;
    await app.camera.play();
    await app.waitForVideoEvent("loadeddata", stream);
    if (session !== app.detectionSession || app.cameraStopping || app.cameraStream !== stream) return false;
    if (!app.camera.videoWidth || !app.camera.videoHeight) throw new Error("O vídeo iniciou sem dimensões válidas.");
    app.prepareCanvas();
    app.poseContext.clearRect(0, 0, app.poseCanvas.width, app.poseCanvas.height);
    app.monitorCameraTrack(stream, session);
    return true;

    };
    app.monitorCameraTrack = function(stream, session) {
    const track = stream.getVideoTracks()[0];
    if (!track) return;
    const handleEnded = () => {
        if (session !== app.detectionSession || app.cameraStopping || app.cameraStream !== stream) return;
        if (app.cameraSwitching) {
            setTimeout(handleEnded, 50);
            return;
        }
        app.registerEvent("A conexão da câmera foi encerrada pelo dispositivo.");
        app.stopCamera().then(() => {
            if (!app.cameraStream && !app.cameraStopping) {
                app.setCameraStatus("Câmera desconectada");
                app.setPoseStatus("Erro na câmera: conexão encerrada.");
            }
        });
    };
    track.addEventListener("ended", handleEnded, { once: true });

    };
    app.startCamera = async function() {
    if (app.cameraStartingSession !== null || app.cameraStopping || app.cameraSwitching || app.cameraStream) return;
    const session = ++app.detectionSession;
    app.resetMotionContinuity();
    app.resetFallDetection();
    app.postureHistory = [];
    app.resetAnalysisHistory();
    app.resetAnalysisDimensions();
    app.lastPersonDetectedAt = null;
    app.detectionStartedAt = Date.now();
    app.cameraStartingSession = session;
    document.getElementById("startCamera").disabled = true;
    document.getElementById("stopCamera").disabled = false;
    app.setCameraStatus("Inicializando...");
    app.setAiStatus("Aguardando câmera");
    app.setAnalysisPanel("Aguardando análise", "Aguardando câmera", "Inicializando câmera", "processing");

    try {
        if (!navigator.mediaDevices?.getUserMedia) {
            throw new Error("Câmera indisponível. Utilize HTTPS ou localhost.");
        }

        let stream;
        let frontCameraFallback = false;
        let requestedFacingMode = "user";
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { exact: "user" } }, audio: false });
        } catch (frontError) {
            if (!["NotFoundError", "OverconstrainedError", "ConstraintNotSatisfiedError", "TypeError"].includes(frontError.name)) throw frontError;
            frontCameraFallback = true;
            requestedFacingMode = "environment";
            app.setPoseStatus("Câmera frontal indisponível; tentando uma câmera alternativa...");
            stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: "environment" } }, audio: false });
        }

        if (session !== app.detectionSession) {
            stream.getTracks().forEach(track => track.stop());
            return;
        }

        const initialTrack = stream.getVideoTracks()[0];
        const initialSettings = initialTrack?.getSettings?.() ?? {};
        const detectedFacing = await app.identifyTrackFacingMode(initialTrack);
        frontCameraFallback = detectedFacing === "environment" || (frontCameraFallback && detectedFacing === null);
        app.cameraDeviceId = initialSettings.deviceId ?? null;
        app.cameraOrientationKnown = Boolean(detectedFacing);
        app.cameraRequestedFacingMode = requestedFacingMode;
        // facingMode exato confirmado seleciona a frontal mesmo sem metadados; ideal não confirma orientação.
        app.cameraFacingMode = detectedFacing ?? (requestedFacingMode === "user" ? "user" : null);
        app.cameraStream = stream;
        await app.updateCameraAvailability(session);
        if (session !== app.detectionSession || app.cameraStopping) {
            stream.getTracks().forEach(track => track.stop());
            app.cameraStream = null;
            return;
        }
        const videoReady = await app.attachCameraStream(stream, session);
        if (!videoReady || session !== app.detectionSession || app.cameraStopping) {
            stream.getTracks().forEach(track => track.stop());
            if (app.cameraStream === stream) app.cameraStream = null;
            return;
        }
        app.setCameraStatus("Conectada");
        document.getElementById("startCamera").disabled = true;
        document.getElementById("stopCamera").disabled = false;
        app.registerEvent(frontCameraFallback ? "Câmera alternativa ativada; a frontal não estava disponível." : "Câmera ativada.");

        const detector = await app.loadPoseDetector(session);

        if (detector && session === app.detectionSession && !app.cameraStopping) {
            app.setPoseStatus("Procurando uma pessoa...");
            await app.updateCameraAvailability(session);
            app.poseAnimationFrame = requestAnimationFrame(() => app.detectPose(session));
        }
    } catch (error) {
        if (session === app.detectionSession) {
            app.setCameraStatus("Erro na conexão");
            app.setAiStatus("Não inicializada");
            app.setPoseStatus(`Erro na IA ou câmera: ${error.message}`);
            app.registerEvent(`Erro ao iniciar câmera ou IA: ${error.message}`);

            if (app.cameraStream) {
                app.cameraStream.getTracks().forEach(track => track.stop());
                app.cameraStream = null;
            }
            app.cameraDeviceId = null;
            app.cameraOrientationKnown = false;
            app.cameraRequestedFacingMode = null;
            app.camera.srcObject = null;
            app.camera.style.display = "none";
            app.poseCanvas.style.display = "none";
            document.getElementById("placeholder").style.display = "flex";
            document.getElementById("startCamera").disabled = false;
            document.getElementById("stopCamera").disabled = true;
            document.getElementById("flipCamera").disabled = true;
        }
    } finally {
        // Uma inicialização antiga não pode liberar o bloqueio de uma sessão mais nova.
        if (app.cameraStartingSession === session) {
            app.cameraStartingSession = null;
        }
        if (session === app.detectionSession && app.cameraStream && !app.cameraStopping) {
            await app.updateCameraAvailability(session);
        } else if (session === app.detectionSession) {
            app.setCameraStatus(document.getElementById("cameraStatus").textContent);
        }
        app.syncSystemStatus();
    }

    };
    app.switchCamera = async function() {
    if (app.cameraSwitching || app.cameraStopping || app.cameraStartingSession !== null || !app.cameraStream) return;
    if (app.availableCameraCount === 1) {
        app.setPoseStatus("Este dispositivo possui apenas uma câmera disponível.");
        return;
    }

    app.cameraSwitching = true;
    const nextSession = ++app.detectionSession;
    app.resetMotionContinuity();
    app.resetFallDetection();
    const flipButton = document.getElementById("flipCamera");
    let previousDeviceId = app.cameraDeviceId;
    const previousFacingMode = app.cameraFacingMode;
    const previousRequestedFacingMode = app.cameraRequestedFacingMode;
    const previousOrientationKnown = app.cameraOrientationKnown;
    let oldStream = null;
    document.getElementById("startCamera").disabled = true;
    document.getElementById("stopCamera").disabled = true;
    flipButton.disabled = true;
    app.setCameraStatus("Trocando câmera...");
    app.setPoseStatus("Trocando câmera...");
    app.postureHistory = [];
    app.resetAnalysisHistory();
    app.resetAnalysisDimensions();
    app.lastPersonDetectedAt = null;
    app.detectionStartedAt = Date.now();

    if (app.poseAnimationFrame !== null) {
        cancelAnimationFrame(app.poseAnimationFrame);
        app.poseAnimationFrame = null;
    }

    try {
        // Fecha a inferência anterior antes de alterar a fonte do elemento video.
        if (app.detectionPromise) await app.detectionPromise.catch(() => {});
        if (nextSession !== app.detectionSession || app.cameraStopping) return;

        oldStream = app.cameraStream;
        const oldTrack = oldStream?.getVideoTracks()[0];
        previousDeviceId = oldTrack?.getSettings?.().deviceId ?? previousDeviceId;
        oldStream?.getTracks().forEach(track => track.stop());
        app.cameraStream = null;
        app.camera.srcObject = null;
        app.poseContext.clearRect(0, 0, app.poseCanvas.width, app.poseCanvas.height);

        const currentFacing = previousFacingMode ?? previousRequestedFacingMode;
        const requestedFacing = currentFacing === "environment" ? "user" : "environment";
        const selection = await app.requestCameraStream(requestedFacing, previousDeviceId);
        if (nextSession !== app.detectionSession || app.cameraStopping) {
            selection.stream.getTracks().forEach(track => track.stop());
            return;
        }

        app.cameraFacingMode = selection.facingMode;
        app.cameraRequestedFacingMode = requestedFacing;
        const videoReady = await app.attachCameraStream(selection.stream, nextSession);
        if (!videoReady || nextSession !== app.detectionSession || app.cameraStopping) {
            selection.stream.getTracks().forEach(track => track.stop());
            if (app.cameraStream === selection.stream) app.cameraStream = null;
            return;
        }

        app.cameraDeviceId = selection.deviceId;
        app.cameraOrientationKnown = selection.orientationKnown;
        app.setCameraStatus("Conectada");
        app.setAiStatus("Modelo carregado");
        app.setPoseStatus("Procurando uma pessoa...");
        await app.updateCameraAvailability(nextSession);
        if (nextSession === app.detectionSession && app.poseDetector && app.cameraStream && !app.cameraStopping) {
            app.poseAnimationFrame = requestAnimationFrame(() => app.detectPose(nextSession));
        }
    } catch (error) {
        if (nextSession !== app.detectionSession || app.cameraStopping) return;

        if (app.cameraStream) app.cameraStream.getTracks().forEach(track => track.stop());
        app.cameraStream = null;
        app.cameraDeviceId = null;
        app.cameraOrientationKnown = false;
        app.cameraRequestedFacingMode = null;
        app.camera.srcObject = null;
        app.camera.style.display = "none";
        app.poseCanvas.style.display = "none";
        document.getElementById("placeholder").style.display = "flex";
        app.poseContext.clearRect(0, 0, app.poseCanvas.width, app.poseCanvas.height);

        // Tenta reabrir o dispositivo anterior pelo ID exato; nunca ativa um substituto aleatório.
        if (previousDeviceId) {
            try {
                const restoredStream = await navigator.mediaDevices.getUserMedia({
                    video: { deviceId: { exact: previousDeviceId } },
                    audio: false
                });
                const restoredDeviceId = restoredStream.getVideoTracks()[0]?.getSettings?.().deviceId;
                if (restoredDeviceId && restoredDeviceId !== previousDeviceId) {
                    restoredStream.getTracks().forEach(track => track.stop());
                    throw new Error("O navegador não reabriu o dispositivo de câmera anterior.");
                }
                if (nextSession !== app.detectionSession || app.cameraStopping) {
                    restoredStream.getTracks().forEach(track => track.stop());
                    return;
                }
                app.cameraFacingMode = previousFacingMode;
                app.cameraRequestedFacingMode = previousRequestedFacingMode;
                const restored = await app.attachCameraStream(restoredStream, nextSession);
                if (nextSession !== app.detectionSession || app.cameraStopping) {
                    restoredStream.getTracks().forEach(track => track.stop());
                    return;
                }
                if (restored && nextSession === app.detectionSession && !app.cameraStopping) {
                    app.cameraFacingMode = previousFacingMode;
                    app.cameraRequestedFacingMode = previousRequestedFacingMode;
                    app.cameraDeviceId = previousDeviceId;
                    app.cameraOrientationKnown = previousOrientationKnown;
                    app.setAiStatus(app.poseDetector ? "Modelo carregado" : "Não inicializada");
                    app.setPoseStatus("Procurando uma pessoa...");
                    app.setCameraStatus("Câmera anterior restaurada");
                    app.registerEvent("A câmera anterior foi restaurada após falha na troca.");
                    await app.updateCameraAvailability(nextSession);
                    if (app.poseDetector && app.cameraStream && !app.cameraStopping) {
                        app.poseAnimationFrame = requestAnimationFrame(() => app.detectPose(nextSession));
                    }
                    return;
                }
                restoredStream.getTracks().forEach(track => track.stop());
                if (app.cameraStream === restoredStream) app.cameraStream = null;
            } catch (restoreError) {
                if (app.cameraStream) app.cameraStream.getTracks().forEach(track => track.stop());
                app.cameraStream = null;
                error = new Error(`${error.message} A câmera anterior também não pôde ser restaurada: ${restoreError.message}`);
            }
        }

        app.cameraStream = null;
        app.cameraDeviceId = null;
        app.camera.srcObject = null;
        app.camera.style.display = "none";
        app.poseCanvas.style.display = "none";
        document.getElementById("placeholder").style.display = "flex";
        app.setCameraStatus("Câmera indisponível");
        app.setAiStatus(app.poseDetector ? "Modelo carregado" : "Não inicializada");
        app.setPoseStatus(`Não foi possível trocar a câmera: ${error.message}`);
        app.registerEvent(`Erro ao trocar câmera: ${error.message}`);
    } finally {
        app.cameraSwitching = false;
        app.syncSystemStatus();
        if (nextSession === app.detectionSession && !app.cameraStopping) {
            document.getElementById("startCamera").disabled = Boolean(app.cameraStream);
            document.getElementById("stopCamera").disabled = !app.cameraStream;
            if (app.cameraStream) await app.updateCameraAvailability(nextSession);
            else app.setCameraStatus(document.getElementById("cameraStatus").textContent);
        }
    }

    };
    app.stopCamera = async function() {
    if (app.cameraStopping || app.cameraSwitching) return;

    // Invalida a sessão antes de qualquer limpeza para bloquear novos frames e desenhos.
    const invalidatedSession = app.detectionSession;
    app.detectionSession++;
    app.resetMotionContinuity();
    app.resetFallDetection();
    app.postureHistory = [];
    app.resetAnalysisHistory();
    app.resetAnalysisDimensions();
    app.lastPersonDetectedAt = null;
    app.detectionStartedAt = null;
    // Cancela logicamente a inicialização antiga sem esperar getUserMedia terminar.
    if (app.cameraStartingSession === invalidatedSession) {
        app.cameraStartingSession = null;
    }
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
    app.cameraRequestedFacingMode = null;

    app.camera.srcObject = null;
    app.camera.style.display = "none";
    app.poseCanvas.style.display = "none";
    app.poseContext.clearRect(0, 0, app.poseCanvas.width, app.poseCanvas.height);
    document.getElementById("placeholder").style.display = "flex";
    app.setCameraStatus("Desligada");
    app.setAiStatus("Não inicializada");
    document.getElementById("stopCamera").disabled = true;
    document.getElementById("flipCamera").disabled = true;
    app.setPoseStatus("IA corporal aguardando ativação.");

    const detector = app.poseDetector;
    app.poseDetector = null;
    try {
        // Aguarda a inferência em andamento antes de liberar o detector que ela utiliza.
        if (app.detectionPromise) {
            await app.detectionPromise.catch(() => {});
        }

        if (detector) {
            await detector.dispose();
        }
    } catch (error) {
        app.registerEvent(`Erro ao liberar o modelo MoveNet: ${error.message}`);
    } finally {
        // A inferência encerrou antes do dispose; mantém o estado consistente ao reativar.
        if (app.detectionPromise === null) app.detectionInProgress = false;
        app.cameraStopping = false;
        document.getElementById("startCamera").disabled = false;
        app.syncSystemStatus();
    }

    app.registerEvent("Câmera desligada.");

    };
}
