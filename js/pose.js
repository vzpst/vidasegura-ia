// Funções extraídas do fluxo original; o estado e as dependências são injetados por main.js.
export function installPose(app) {
    app.loadPoseDetector = async function(session) {
    if (session !== app.detectionSession || app.cameraStopping) return null;
    if (app.poseDetector) return app.poseDetector;
    app.setAiStatus("Carregando modelo");
    app.setPoseStatus("IA carregando o modelo MoveNet...");

    if (!window.tf || !window.poseDetection) {
        throw new Error("As bibliotecas TensorFlow.js ou Pose Detection não foram carregadas.");
    }

    await window.tf.ready();

    if (session !== app.detectionSession) {
        return null;
    }

    const model = window.poseDetection.SupportedModels?.MoveNet;
    const singlePoseLightning = window.poseDetection.movenet?.modelType?.SINGLEPOSE_LIGHTNING;

    if (!model || !singlePoseLightning) {
        throw new Error("A configuração MoveNet SINGLEPOSE_LIGHTNING não está disponível.");
    }

    const detector = await window.poseDetection.createDetector(model, {
        modelType: singlePoseLightning
    });

    if (session !== app.detectionSession) {
        await detector.dispose();
        return null;
    }

    app.poseDetector = detector;
    app.setAiStatus("Modelo carregado");
    return detector;

    };
    app.prepareCanvas = function() {
    const rect = app.videoContainer.getBoundingClientRect();
    const width = Math.max(1, Math.round(rect.width));
    const height = Math.max(1, Math.round(rect.height));

    if (app.poseCanvas.width !== width || app.poseCanvas.height !== height) {
        app.poseCanvas.width = width;
        app.poseCanvas.height = height;
    }

    const videoWidth = app.camera.videoWidth;
    const videoHeight = app.camera.videoHeight;

    if (!videoWidth || !videoHeight) {
        return null;
    }

    const scale = Math.min(width / videoWidth, height / videoHeight);
    const drawWidth = videoWidth * scale;
    const drawHeight = videoHeight * scale;

    return {
        scale,
        offsetX: (width - drawWidth) / 2,
        offsetY: (height - drawHeight) / 2
    };

    };
    app.drawPose = function(pose) {
    const frame = app.prepareCanvas();
    app.poseContext.clearRect(0, 0, app.poseCanvas.width, app.poseCanvas.height);

    if (!frame || !pose?.keypoints?.length) {
        return false;
    }

    const points = pose.keypoints;
    const threshold = 0.25;
    const connections = [
        [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
        [5, 11], [6, 12], [11, 12], [11, 13], [13, 15],
        [12, 14], [14, 16]
    ];
    const visible = point => point && (point.score ?? 0) >= threshold;
    const toCanvas = point => ({
        // O vídeo frontal já está espelhado; reflete os keypoints uma única vez no canvas.
        x: frame.offsetX + (app.cameraFacingMode === "user" ? app.camera.videoWidth - point.x : point.x) * frame.scale,
        y: frame.offsetY + point.y * frame.scale
    });

    app.poseContext.strokeStyle = "#39c99a";
    app.poseContext.lineWidth = 3;
    app.poseContext.lineCap = "round";

    for (const [startIndex, endIndex] of connections) {
        const start = points[startIndex];
        const end = points[endIndex];

        if (!visible(start) || !visible(end)) continue;

        const a = toCanvas(start);
        const b = toCanvas(end);
        app.poseContext.beginPath();
        app.poseContext.moveTo(a.x, a.y);
        app.poseContext.lineTo(b.x, b.y);
        app.poseContext.stroke();
    }

    app.poseContext.fillStyle = "#f8dfad";
    for (const point of points) {
        if (!visible(point)) continue;

        const position = toCanvas(point);
        app.poseContext.beginPath();
        app.poseContext.arc(position.x, position.y, 5, 0, Math.PI * 2);
        app.poseContext.fill();
    }

    return true;

    };
    app.detectPose = async function(session) {
    // O callback já consumiu este frame; futuros agendamentos passam por um único caminho abaixo.
    if (session === app.detectionSession) app.poseAnimationFrame = null;
    // A sessão pode ter sido encerrada entre o agendamento e a execução do frame.
    if (session !== app.detectionSession || !app.poseDetector || !app.cameraStream || app.cameraStopping) return;
    if (app.detectionInProgress) return;

    app.detectionInProgress = true;
    let inference = null;
    let inferenceFailed = false;
    try {
        if (app.camera.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
            const detector = app.poseDetector;
            // A verificação dentro da promessa fecha a janela entre agendamento e inferência.
            inference = Promise.resolve().then(() => {
                if (session !== app.detectionSession || app.cameraStopping ||
                    app.poseDetector !== detector || !app.cameraStream) {
                    return null;
                }

                return detector.estimatePoses(app.camera);
            });
            app.detectionPromise = inference;
            const poses = await inference;

            // Descarta a inferência se desligaram ou reativaram a câmera enquanto ela rodava.
            if (session !== app.detectionSession || app.cameraStopping) return;
            if (poses === null || poses === undefined) {
                app.interruptMotionContinuity();
                app.interruptFallDetection("FALHA_DE_INFERÊNCIA", Date.now());
                return;
            }

            if (poses[0]) {
                app.drawPose(poses[0]);
                app.setAiStatus("Analisando");
                app.lastPersonDetectedAt = Date.now();
                const analysisResult = app.createAnalysisResult(poses[0]);
                app.recordAnalysisResult(analysisResult);
                const motionSample = app.recordMotionInference({
                    pose: poses[0],
                    analysisResult,
                    videoWidth: app.camera.videoWidth,
                    videoHeight: app.camera.videoHeight
                });
                app.evaluateFallSeverity(app.evaluateFallDetection(motionSample), motionSample);
                app.updateActivity?.(analysisResult, poses[0], app.camera.videoWidth, app.camera.videoHeight);
                const posture = analysisResult.posture.state === "INCONCLUSIVA" ? null : analysisResult.posture.state;
                const stablePosture = app.updatePostureHistory(posture);
                if (stablePosture) app.recordPositionChange(stablePosture, analysisResult.timestamp);
                if (!posture) app.setPoseStatus("Não foi possível identificar a posição.");
                else app.setPoseStatus(stablePosture
                    ? `Posição identificada: ${stablePosture}`
                    : "Pessoa detectada — analisando posição...");
            } else {
                app.setAiStatus("Analisando");
                const analysisResult = app.createAnalysisResult(null);
                app.recordAnalysisResult(analysisResult);
                const motionSample = app.recordMotionInference({
                    pose: null,
                    analysisResult,
                    videoWidth: app.camera.videoWidth,
                    videoHeight: app.camera.videoHeight
                });
                app.evaluateFallSeverity(app.evaluateFallDetection(motionSample), motionSample);
                app.updateActivity?.(analysisResult, null, app.camera.videoWidth, app.camera.videoHeight);
                app.drawPose(null);
                app.updatePostureHistory(null);
                // Pequenas falhas não apagam a mensagem; só declara ausência após 1,5 s.
                if (Date.now() - (app.lastPersonDetectedAt ?? app.detectionStartedAt ?? Date.now()) >= 1500) {
                    app.postureHistory = [];
                    app.setPoseStatus("Procurando uma pessoa...");
                }
            }
        }
    } catch (error) {
        if (session === app.detectionSession && app.cameraStream && !app.cameraStopping) {
            inferenceFailed = true;
            app.interruptMotionContinuity();
            app.interruptFallDetection("FALHA_DE_INFERÊNCIA", Date.now());
            app.drawPose(null);
            app.setAiStatus("Análise indisponível");
            app.setPoseStatus(`Erro no processamento da IA: ${error.message}`);
            app.registerEvent(`Erro no processamento da IA: ${error.message}`);
        }
    } finally {
        // Uma conclusão antiga não pode limpar o estado de uma inferência mais nova.
        if (inference && app.detectionPromise === inference) {
            app.detectionPromise = null;
            app.detectionInProgress = false;
        } else if (!inference && session === app.detectionSession) {
            app.detectionInProgress = false;
        }
    }

    if (session === app.detectionSession && app.cameraStream && !app.cameraStopping && app.poseDetector) {
        if (inferenceFailed) {
            // A pausa evita tentativas contínuas quando a inferência falha repetidamente.
            setTimeout(() => {
                if (session === app.detectionSession && app.cameraStream && !app.cameraStopping && app.poseDetector &&
                    app.poseAnimationFrame === null) {
                    app.poseAnimationFrame = requestAnimationFrame(() => app.detectPose(session));
                }
            }, 750);
        } else if (app.poseAnimationFrame === null) {
            app.poseAnimationFrame = requestAnimationFrame(() => app.detectPose(session));
        }
    }

    };
}
