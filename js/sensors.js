// Funções extraídas do fluxo original; o estado e as dependências são injetados por main.js.
export function installSensors(app) {
    app.startSensors = async function() {
    try {
        if (!("DeviceMotionEvent" in window)) {
            throw new Error("Sensor de movimento não disponível neste dispositivo.");
        }

        if (typeof DeviceMotionEvent.requestPermission === "function") {
            const permission = await DeviceMotionEvent.requestPermission();
            if (permission !== "granted") {
                throw new Error("Permissão dos sensores negada.");
            }
        }

        window.addEventListener("devicemotion", app.handleMotion);
        app.sensorActive = true;
        app.previousAcceleration = null;
        app.setSensorStatus("Ativos");
        document.getElementById("startSensors").disabled = true;
        document.getElementById("stopSensors").disabled = false;
        document.getElementById("sensorHelp").textContent =
            "Sensores ativos. Movimente o celular suavemente.";
        app.registerEvent("Sensores ativados.");
    } catch (error) {
        app.setSensorStatus("Indisponíveis");
        document.getElementById("sensorHelp").textContent = error.message;
        app.registerEvent(error.message);
    }

    };
    app.handleMotion = function(event) {
    // Prefere aceleração sem gravidade e usa a leitura incluindo gravidade como alternativa.
    const acceleration = event.acceleration || event.accelerationIncludingGravity;

    if (!acceleration) {
        app.setSensorStatus("Sem dados");
        document.getElementById("sensorHelp").textContent =
            "O navegador não forneceu dados de aceleração.";
        return;
    }

    const x = acceleration.x ?? 0;
    const y = acceleration.y ?? 0;
    const z = acceleration.z ?? 0;
    document.getElementById("accX").textContent = x.toFixed(2);
    document.getElementById("accY").textContent = y.toFixed(2);
    document.getElementById("accZ").textContent = z.toFixed(2);

    if (app.previousAcceleration !== null) {
        const dx = x - app.previousAcceleration.x;
        const dy = y - app.previousAcceleration.y;
        const dz = z - app.previousAcceleration.z;
        const variation = Math.sqrt(dx * dx + dy * dy + dz * dz);
        document.getElementById("motionDelta").textContent = variation.toFixed(2);

        // Este alerta descreve movimento brusco do aparelho, não uma queda humana.
        if (variation > 12 && Date.now() - app.lastAlert > 5000) {
            app.lastAlert = Date.now();
            document.getElementById("riskStatus").textContent = "Movimento brusco";
            app.registerEvent("Movimento brusco do aparelho detectado. Verificar manualmente.");
        } else if (variation <= 12) {
            document.getElementById("riskStatus").textContent = "Sem evento";
        }
    }

    app.previousAcceleration = { x, y, z };

    };
    app.stopSensors = function() {
    window.removeEventListener("devicemotion", app.handleMotion);
    app.sensorActive = false;
    app.previousAcceleration = null;
    app.setSensorStatus("Aguardando ativação");
    document.getElementById("startSensors").disabled = false;
    document.getElementById("stopSensors").disabled = true;
    document.getElementById("sensorHelp").textContent = "Sensores desativados.";
    app.registerEvent("Sensores desligados.");

    };
}
