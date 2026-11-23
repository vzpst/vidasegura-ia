const modelField = document.getElementById("modelUrl");
const result = document.getElementById("result");
const params = new URLSearchParams(location.search);
modelField.value = params.get("model") ?? "";

let loadedModel = null;

document.getElementById("verify").addEventListener("click", async () => {
    const modelValue = modelField.value.trim();
    if (!modelValue) {
        result.textContent = "Informe a URL local de model.json.";
        return;
    }
    result.textContent = "Carregando model.json e pesos…";
    try {
        const modelUrl = new URL(modelValue, location.href);
        const metadataUrl = new URL("metadata.json", modelUrl);
        const metadataResponse = await fetch(metadataUrl);
        if (!metadataResponse.ok) throw new Error(`metadata.json retornou HTTP ${metadataResponse.status}`);
        const metadata = await metadataResponse.json();
        if (metadata.schema_version !== 1 || !metadata.feature_transform?.feature_count || !metadata.class_order?.length) {
            throw new Error("metadata.json ausente ou incompatível com o esquema 1.");
        }
        if (loadedModel) loadedModel.dispose();
        loadedModel = await window.tf.loadLayersModel(modelUrl.href);
        const inputSize = metadata.feature_transform.feature_count;
        const classCount = metadata.class_order.length;
        const inputShape = loadedModel.inputs[0]?.shape;
        if (inputShape?.at(-1) !== inputSize) {
            throw new Error(`Entrada do modelo (${inputShape?.at(-1)}) não corresponde ao vetor (${inputSize}).`);
        }
        const input = window.tf.zeros([1, inputSize]);
        const output = loadedModel.predict(input);
        const outputs = Array.isArray(output) ? output : [output];
        const primary = outputs[0];
        const outputShape = primary.shape;
        const finite = Array.from(primary.dataSync()).every(Number.isFinite);
        input.dispose();
        outputs.forEach(tensor => tensor.dispose());
        if (outputShape.at(-1) !== classCount || !finite) {
            throw new Error("Saída do modelo incompatível com as classes ou contém valores inválidos.");
        }
        result.textContent = [
            "Smoke test concluído: tf.loadLayersModel carregou o modelo e predict executou.",
            `Versão: ${metadata.model_version}`,
            `Entrada: ${inputSize} características (${metadata.feature_transform.version})`,
            `Saída: ${classCount} classes (${metadata.class_order.join(", ")})`,
            "A entrada neutra só verifica compatibilidade do formato; não mede desempenho nem valida o modelo."
        ].join("\n");
    } catch (error) {
        if (loadedModel) {
            loadedModel.dispose();
            loadedModel = null;
        }
        result.textContent = `Falha na verificação: ${error.message}`;
    }
});
