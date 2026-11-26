// Funções extraídas do fluxo original; o estado e as dependências são injetados por main.js.
export function installUI(app) {
    app.applyTheme = function(theme, savePreference = true) {
    const isDark = theme === "dark";
    const selectedTheme = isDark ? "dark" : "light";
    const toggle = document.getElementById("themeToggle");

    document.documentElement.dataset.theme = selectedTheme;
    toggle.setAttribute("aria-pressed", String(isDark));
    toggle.setAttribute("aria-label", isDark ? "Ativar tema claro" : "Ativar tema escuro");
    toggle.title = isDark ? "Ativar tema claro" : "Ativar tema escuro";
    document.getElementById("themeIconUse").setAttribute("href", isDark ? "#icon-moon" : "#icon-sun");
    document.querySelector('meta[name="theme-color"]').setAttribute("content", isDark ? "#171a21" : "#f7f7f8");

    if (savePreference) {
        try {
            localStorage.setItem("vidaSeguraTheme", selectedTheme);
        } catch (error) {
            // A troca de tema continua ativa mesmo se o navegador bloquear o armazenamento.
        }
    }

    };
    app.setPoseStatus = function(message) {
    document.getElementById("poseStatus").textContent = message;
    const identified = message.match(/^Posição identificada: (.+)$/);
    const normalized = message.toLocaleLowerCase("pt-BR");

    if (identified) {
        app.setAnalysisPanel(identified[1], "Pessoa detectada", "Postura identificada", "success");
    } else if (normalized.includes("ia carregando")) {
        app.setAnalysisPanel("Aguardando análise", "Aguardando primeiro quadro", "Carregando MoveNet", "processing");
    } else if (normalized.startsWith("pessoa detectada")) {
        app.setAnalysisPanel("Analisando posição", "Pessoa detectada", "Analisando postura", "attention");
    } else if (normalized.includes("não foi possível identificar a posição")) {
        app.setAnalysisPanel("INCONCLUSIVA", "Pessoa detectada", "Postura inconclusiva", "attention");
    } else if (normalized.includes("procurando uma pessoa")) {
        app.setAnalysisPanel(app.cameraStream ? "INCONCLUSIVA" : "Aguardando câmera",
            app.cameraStream ? "Procurando pessoa" : "Aguardando câmera",
            app.cameraStream ? "Analisando imagem" : "Câmera desligada",
            app.cameraStream ? "processing" : "waiting");
    } else if (normalized.includes("trocando câmera")) {
        app.setAnalysisPanel("Aguardando análise", "Aguardando câmera", "Trocando câmera", "processing");
    } else if (normalized.includes("erro") || normalized.includes("não foi possível")) {
        app.setAnalysisPanel("Análise indisponível", "Detecção indisponível", "Erro na análise", "error");
    } else {
        app.setAnalysisPanel("Aguardando câmera", "Aguardando câmera", "Câmera desligada", "waiting");
    }

    if (app.latestAnalysisResult) app.renderAnalysisDimensions(app.latestAnalysisResult);

    };
    app.setAnalysisPanel = function(posture, person, state, mood) {
    const card = document.getElementById("analysisCard");
    const badge = document.getElementById("analysisBadge");
    const postureValue = document.getElementById("postureValue");
    const personState = document.getElementById("personState");
    const analysisState = document.getElementById("analysisState");
    const postureHighlight = document.querySelector(".posture-highlight");

    postureValue.textContent = posture;
    personState.textContent = person;
    analysisState.textContent = state;
    card.dataset.state = mood;
    badge.dataset.state = mood;
    postureHighlight.dataset.state = mood;
    postureValue.dataset.state = mood;
    personState.dataset.state = mood;
    analysisState.dataset.state = mood;
    badge.textContent = state;

    };
    app.syncSystemStatus = function() {
    const status = document.getElementById("systemStatus");
    const dot = document.getElementById("systemDot");
    const cameraMessage = document.getElementById("cameraStatus").textContent;
    let message = "Sistema aguardando inicialização";
    let mood = "";

    if (app.cameraSwitching) {
        message = "Trocando câmera...";
    } else if (app.cameraStartingSession !== null) {
        message = "Inicializando monitoramento...";
    } else if (/erro|indisponível/i.test(cameraMessage)) {
        message = "Câmera desconectada";
        mood = "is-error";
    } else if (app.cameraStream && !app.poseDetector) {
        message = "IA carregando";
    } else if (app.cameraStream && app.poseDetector) {
        message = "Sistema em funcionamento";
        mood = "is-active";
    } else if (app.cameraStopping) {
        message = "Monitoramento pausado";
    } else if (app.sensorActive) {
        message = "Sensores ativos · câmera desligada";
        mood = "is-active";
    }

    status.textContent = message;
    dot.classList.remove("is-active", "is-error");
    if (mood) dot.classList.add(mood);

    };
    app.setCameraStatus = function(message) {
    document.getElementById("cameraStatus").textContent = message;
    const liveBadge = document.getElementById("cameraLiveBadge");
    const badgeText = liveBadge.querySelector("span");
    let badgeState = "waiting";

    if (app.cameraSwitching) {
        badgeText.textContent = "TROCANDO";
        badgeState = "processing";
    } else if (app.cameraStartingSession !== null) {
        badgeText.textContent = "INICIANDO";
        badgeState = "processing";
    } else if (/erro|indisponível/i.test(message)) {
        badgeText.textContent = "ERRO";
        badgeState = "error";
    } else if (app.cameraStream) {
        badgeText.textContent = "AO VIVO";
        badgeState = "success";
    } else {
        badgeText.textContent = "DESLIGADA";
    }

    liveBadge.dataset.state = badgeState;
    app.syncSystemStatus();

    };
    app.setAiStatus = function(message) {
    const aiStatus = document.getElementById("aiStatus");
    if (aiStatus.textContent !== message) aiStatus.textContent = message;
    app.syncSystemStatus();

    };
    app.setSensorStatus = function(message) {
    document.getElementById("sensorStatus").textContent = message;
    app.syncSystemStatus();

    };
    app.registerEvent = function(message) {
    const log = document.getElementById("eventLog");

    if (log.querySelector("p")) {
        log.innerHTML = "";
    }

    const event = document.createElement("div");
    event.className = "event";
    event.textContent = `${new Date().toLocaleTimeString("pt-BR")} — ${message}`;
    log.prepend(event);

    };
    app.renderAnalysisDimensions = function(result) {
    const personState = document.getElementById("personState");
    const postureValue = document.getElementById("postureValue");
    const visibilityState = document.getElementById("visibilityState");
    const fallRiskState = document.getElementById("fallRiskState");
    if (personState) {
        personState.textContent = result.personDetection.state;
        personState.dataset.state = result.personDetection.detected ? "success" : "waiting";
    }
    if (!result.personDetection.detected && postureValue) {
        postureValue.textContent = "INCONCLUSIVA";
        postureValue.dataset.state = "attention";
    }
    if (!visibilityState || !fallRiskState) return;

    const counts = result.visibility.regionCounts;
    const visibilityDetails = result.visibility.state === "NÃO AVALIÁVEL"
        ? ""
        : ` · corpo ${result.visibility.observedCount}/${result.visibility.totalPoints} · cabeça ${result.visibility.facialObservedCount}/${result.visibility.facialTotalPoints} · tronco ${counts.torso}/4 · braços ${counts.arms}/4 · pernas ${counts.legs}/4`;
    visibilityState.textContent = `${result.visibility.state}${visibilityDetails}`;
    visibilityState.dataset.state = result.visibility.state === "BOA VISIBILIDADE" ? "success" :
        result.visibility.state === "VISIBILIDADE PARCIAL" ? "attention" : "waiting";
    // Com o módulo de gravidade instalado, o cartão é atualizado por renderFallSeverity.
    if (app.latestFallSeverity) return;
    fallRiskState.textContent = result.fallRisk.state;
    fallRiskState.dataset.state = "waiting";

    };

    const severityStates = { 0: "waiting", 1: "processing", 2: "attention", 3: "error" };
    const outcomeLabels = {
        RECUPERADA_LOGO_APÓS_IMPACTO: "recuperação logo após o impacto",
        RECUPERADA: "recuperação",
        PESSOA_INFORMOU_ESTAR_BEM: "pessoa informou estar bem"
    };
    let lastRenderedLevel = 0;
    let lastBeepLevel = 0;

    function beep() {
        try {
            const AudioContextClass = window.AudioContext || window.webkitAudioContext;
            if (!AudioContextClass) return;
            const context = new AudioContextClass();
            const oscillator = context.createOscillator();
            const gain = context.createGain();
            oscillator.frequency.value = 880;
            gain.gain.value = 0.08;
            oscillator.connect(gain).connect(context.destination);
            oscillator.start();
            oscillator.stop(context.currentTime + 0.4);
            oscillator.onended = () => context.close();
        } catch (error) {
            // Sem áudio disponível: o alerta visual continua ativo.
        }
    }

    app.renderFallSeverity = function(result) {
        const state = document.getElementById("fallRiskState");
        const alert = document.getElementById("fallAlert");
        if (!state || !result) return;

        const seconds = Number.isFinite(result.elapsedSinceImpactMs) ? Math.round(result.elapsedSinceImpactMs / 1000) : null;
        let text = result.label;
        if (result.phase === "OBSERVANDO_DESCIDA") text = "Descida rápida detectada · analisando";
        else if (result.level === 0 && result.phase === "SEM_EVENTO" && result.evidence === "insuficiente") {
            text = "Monitorando · pessoa fora de vista ou pouco visível";
        } else if (result.level === 0 && result.phase === "SEM_EVENTO") text = "Monitorando · sem queda";
        if (result.phase === "EM_ACOMPANHAMENTO" && seconds !== null) text += ` · ${seconds}s após o impacto`;
        if (result.trackingLost) text += " · pessoa saiu de vista após a possível queda";
        else if (result.evidence === "insuficiente" && result.phase === "EM_ACOMPANHAMENTO") {
            text += " · evidência visual insuficiente";
        }
        if (result.phase === "ENCERRADO" && result.outcome) text += ` · encerrado: ${outcomeLabels[result.outcome] ?? result.outcome}`;
        state.textContent = text;
        state.dataset.state = severityStates[result.level] ?? "waiting";

        if (result.phase === "EM_ACOMPANHAMENTO" && result.level !== lastRenderedLevel && result.level >= 2) {
            app.registerEvent(`${result.label} (${seconds ?? "?"}s após o impacto). Verificar a pessoa.`);
        }
        lastRenderedLevel = result.phase === "EM_ACOMPANHAMENTO" ? result.level : 0;

        if (!alert) return;
        // Nível 1 em acompanhamento já aparece (sem som) para a pessoa poder tocar em "Estou bem".
        const showAlert = result.alarm === "aviso" || result.alarm === "alerta" ||
            (result.phase === "EM_ACOMPANHAMENTO" && result.level >= 1);
        alert.hidden = !showAlert;
        alert.dataset.level = String(result.level);
        if (showAlert) {
            document.getElementById("fallAlertTitle").textContent = result.label;
            document.getElementById("fallAlertText").textContent = result.level >= 3
                ? "A pessoa parece estar em posição baixa há muito tempo ou imóvel. Verifique imediatamente."
                : result.level === 1
                    ? "Observando se a pessoa se levanta. Se estiver tudo bem, toque em \"Estou bem\"."
                : result.trackingLost
                    ? "A pessoa saiu do campo de visão após a possível queda e não voltou. Verifique."
                    : "Posição baixa mantida após a carência. Se estiver tudo bem, toque em \"Estou bem\".";
        }
        if (result.level >= 2 && lastBeepLevel < result.level && result.phase === "EM_ACOMPANHAMENTO") beep();
        lastBeepLevel = result.phase === "EM_ACOMPANHAMENTO" ? result.level : 0;
    };

    app.recordFallSeverityEvent = function(summary) {
        const outcome = outcomeLabels[summary.outcome] ?? summary.outcome;
        app.registerEvent(`Possível queda encerrada · nível máximo ${summary.maxLevel} · ${outcome}.`);
    };
    app.resetAnalysisDimensions = function() {
    const visibilityState = document.getElementById("visibilityState");
    const fallRiskState = document.getElementById("fallRiskState");
    if (visibilityState) {
        visibilityState.textContent = "Aguardando inferência";
        visibilityState.dataset.state = "waiting";
    }
    if (fallRiskState) {
        fallRiskState.textContent = "Aguardando câmera";
        fallRiskState.dataset.state = "waiting";
    }

    };

    app.renderPositionHistory = function(positions) {
        const list = document.getElementById("positionHistory");
        if (!list) return;
        list.replaceChildren();

        if (!positions.length) {
            const empty = document.createElement("li");
            empty.className = "empty-state";
            empty.textContent = "Nenhuma posição registrada nesta sessão.";
            list.appendChild(empty);
            return;
        }

        const postureLabels = { "EM PÉ": "Em pé", SENTADA: "Sentada", DEITADA: "Deitada", OUTRA: "Outra (ajoelhada ou inclinada)" };
        const today = new Date();
        positions.forEach(position => {
            const capturedAt = new Date(position.timestamp);
            const includeDate = capturedAt.toDateString() !== today.toDateString();
            const formattedTime = capturedAt.toLocaleTimeString("pt-BR");
            const formattedDate = includeDate ? `${capturedAt.toLocaleDateString("pt-BR")} · ` : "";
            const item = document.createElement("li");
            const time = document.createElement("time");
            const label = document.createElement("span");
            time.dateTime = capturedAt.toISOString();
            time.textContent = `${formattedDate}${formattedTime}`;
            label.textContent = postureLabels[position.posture];
            item.className = "position-entry";
            item.append(time, label);
            list.appendChild(item);
        });
    };
}
