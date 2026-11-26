# Laboratório local de posturas

Este diretório contém um pipeline experimental separado do monitoramento principal. Ele coleta rótulos manuais, valida JSONL, prepara características versionadas, treina um MLP com Keras e avalia um holdout separado por participantes e sessões. Nenhum modelo é integrado à aplicação nesta etapa.

O classificador geométrico atual é usado somente para gravar uma previsão de referência na amostra e comparar resultados. A coleta não altera `js/posture.js`, a câmera ou o MoveNet da aplicação. O risco da aplicação permanece `NÃO AVALIADO`.

## Estrutura

- `tools/pose-capture.html` e `tools/pose-capture.js`: coleta local com MoveNet e anotações manuais.
- `dataset.schema.v1.json`: esquema JSON Schema do formato JSONL.
- `scripts/common.py`: validação, características v1, agrupamento e gates de diversidade.
- `tools/pose-features-v1.mjs`: mesma transformação `pose-features-v1` para uso futuro no navegador.
- `scripts/validate_dataset.py`: relatório de validação e rejeições.
- `scripts/prepare_dataset.py`: criação de matriz NumPy e metadados.
- `scripts/train_posture.py`: MLP experimental com separação agrupada.
- `scripts/evaluate.py`: métricas do holdout, por perspectiva, participante e situação anotada.
- `scripts/export_model.py`: exportação condicionada à avaliação e revisão humana.
- `tools/verify-tfjs-export.html`: smoke test local com `tf.loadLayersModel()`.

## Compatibilidade e versões

O ambiente documentado no laboratório usa Python 3.12.10, TensorFlow 2.19.0, Keras 3.15.1, NumPy 2.1.3 e Scikit-learn 1.6.1. A faixa do TensorFlow 2.19 inclui Python 3.12; TF-DF 1.12.0 declara suporte a Python 3.12 e compatibilidade com TF 2.19. Os detalhes podem mudar em releases futuros: consulte a [matriz oficial do TensorFlow](https://www.tensorflow.org/install/source) e as [notas de TF-DF](https://github.com/tensorflow/decision-forests/releases).

O TensorFlow.js Converter está fixado em 4.22.0. Seus requisitos incluem TensorFlow, `tf-keras`, TF-DF e dependências de conversão; o projeto TF-DF recomenda Linux/WSL no Windows porque não mantém wheels nativos do Windows ([requisitos do converter](https://raw.githubusercontent.com/tensorflow/tfjs/master/tfjs-converter/python/requirements.txt), [instalação de TF-DF](https://github.com/tensorflow/decision-forests/blob/main/documentation/installation.md)). Por isso, o arquivo de requisitos instala o converter somente em Linux/WSL. A coleta e os testes não precisam do converter.

Os pins são uma base reproduzível para este laboratório, não uma garantia de compatibilidade com toda combinação de sistema operacional e hardware. O host onde os testes foram preparados tinha somente Python 3.15.2; essa versão não atende à faixa do TensorFlow 2.19. Não tente instalar este stack no Python global. Instale Python 3.12.10 e use o ambiente virtual abaixo. O instalador oficial do Windows está em [python.org](https://www.python.org/downloads/windows/).

## 1–3. Instalar Python, criar o ambiente e instalar dependências

Abra PowerShell na raiz do projeto:

```powershell
Set-Location "C:\Users\usuario\Downloads\vz\vida segura IA"
py -0p
py -3.12 --version
py -3.12 -m venv ml\.venv
.\ml\.venv\Scripts\python.exe -m pip install --upgrade pip
.\ml\.venv\Scripts\python.exe -m pip install -r ml\requirements.txt
```

`py -0p` lista os interpretadores disponíveis; `py -3.12 --version` deve indicar Python 3.12.10. Se não encontrar essa versão, instale Python 3.12.10, reabra o PowerShell e repita. Chamadas pelo caminho `ml\.venv\Scripts\python.exe` evitam depender da política de ativação do PowerShell e mantêm os pacotes dentro do projeto.

Pins principais:

| Pacote | Versão | Uso |
|---|---:|---|
| Python | 3.12.10 | Interpretador compatível com a pilha escolhida |
| TensorFlow | 2.19.0 | Treinamento e avaliação Keras |
| Keras | 3.15.1 | MLP experimental |
| NumPy | 2.1.3 | Matrizes de características |
| Scikit-learn | 1.6.1 | Separação agrupada, pesos e métricas |
| TensorFlow.js Converter | 4.22.0 | Conversão Keras para Layers Model, somente Linux/WSL |
| TF-DF / tf-keras | 1.12.0 / 2.19.0 | Dependências da conversão, somente Linux/WSL |

## 4–6. Abrir a ferramenta e coletar amostras

Na raiz do projeto, inicie um servidor local em um terminal PowerShell:

```powershell
py -3.12 -m http.server 8765 --bind 127.0.0.1
```

Abra `http://127.0.0.1:8765/ml/tools/pose-capture.html` no navegador e permita a câmera. A ferramenta usa as mesmas versões de TFJS 4.22.0 e Pose Detection 2.1.3 da aplicação, com MoveNet `SINGLEPOSE_LIGHTNING`. O vídeo é processado ao vivo; imagens, vídeo e keypoints não são enviados nem gravados. Os arquivos JS do modelo são carregados do CDN definido na página. As amostras ficam somente na memória da aba até baixar ou descartar; fechar/recarregar a página as remove.

Use identificadores anônimos como `P001` e `S001`; não use nomes ou dados de contato. Obtenha consentimento antes de coletar. Para cada amostra, anote:

- `posture_label`: `standing`, `sitting`, `lying` ou `other`.
- `viewpoint_label`: `front`, `left_profile`, `right_profile` ou `oblique`.
- `challenge_label` opcional: situação comum ou condição alvo `side_lying`, `squat`, `kneeling` ou `leaning`. Esta etiqueta adicional permite examinar os casos difíceis separadamente; ela não substitui a postura principal.

Revise o contador de articulações observadas. Amostras duvidosas podem ser descartadas individualmente ou em conjunto. Use **Baixar JSONL** para salvar as linhas localmente. Coloque manualmente os arquivos baixados em `ml\data\raw\`; esse diretório é ignorado pelo Git.

Cada linha possui `schema_version`, IDs anônimos, `timestamp_ms`, dimensões do vídeo, `data_origin`, `capture_method`, rótulos, previsão geométrica e os 12 keypoints: ombros, quadris, cotovelos, punhos, joelhos e tornozelos esquerdo/direito. `x` e `y` são normalizados de 0 a 1 a partir das coordenadas originais do MoveNet; `score` preserva a confiança original; `validity` é `observed`, `low_confidence` ou `missing`. Quando faltam coordenadas, elas ficam nulas; não são estimadas. A previsão geométrica é armazenada como referência e pode ser `inconclusive`.

## 7. Validar o JSONL

No PowerShell, encontre os arquivos sem depender de expansão de curingas por um programa nativo:

```powershell
$datasetFiles = @(Get-ChildItem -Path ml\data\raw -Filter *.jsonl -File | ForEach-Object { $_.FullName })
if ($datasetFiles.Count -eq 0) { throw "Nenhum JSONL encontrado em ml\data\raw" }
& .\ml\.venv\Scripts\python.exe ml\scripts\validate_dataset.py $datasetFiles
```

O relatório padrão é `ml\reports\dataset-validation.json`. Ele contabiliza rótulos, participantes, sessões, perspectivas, duplicatas prováveis, amostras utilizáveis, problemas de esquema e diversidade para treinamento. Linhas inválidas não são descartadas silenciosamente; cada rejeição inclui arquivo, linha e motivo. Um código de saída diferente de zero indica rejeições/erros de leitura, não uma avaliação do modelo.

## 8. Preparar os dados

Depois de revisar o relatório, prepare apenas as linhas válidas:

```powershell
& .\ml\.venv\Scripts\python.exe ml\scripts\prepare_dataset.py $datasetFiles --output-dir ml\data\processed\pose-features-v1
```

Por padrão, qualquer rejeição interrompe a preparação. Se decidiu conscientemente continuar apenas com as linhas válidas, acrescente `--allow-invalid-records`; o relatório de preparação ainda registra cada rejeição. Possíveis duplicatas são sinalizadas para revisão.

A transformação v1 cria 118 valores, nesta ordem:

1. Para cada keypoint na ordem `left_shoulder`, `right_shoulder`, `left_hip`, `right_hip`, `left_elbow`, `right_elbow`, `left_wrist`, `right_wrist`, `left_knee`, `right_knee`, `left_ankle`, `right_ankle`: `[x_centered_scaled, y_centered_scaled, confidence, mask]` (48 valores).
2. Para cada par na ordem abaixo: `[dx_scaled, dy_scaled, distance_scaled, angle_from_vertical_norm, mask]` (70 valores).

Pares: ombro esquerdo–direito; quadril esquerdo–direito; ombro esquerdo–quadril esquerdo; ombro direito–quadril direito; quadril esquerdo–joelho esquerdo; quadril direito–joelho direito; joelho esquerdo–tornozelo esquerdo; joelho direito–tornozelo direito; ombro esquerdo–cotovelo esquerdo; ombro direito–cotovelo direito; cotovelo esquerdo–punho esquerdo; cotovelo direito–punho direito; ombro esquerdo–quadril direito; ombro direito–quadril esquerdo.

O centro é a média dos quadris observados, ou dos ombros se não houver quadris, ou dos pontos observados restantes. A escala é a maior extensão observada entre X e Y, limitada inferiormente a `1e-6`. Os ângulos são medidos sem direção em relação ao eixo vertical da imagem: `0` é vertical e `90°` é horizontal, normalizados para `[0, 1]`. Pontos abaixo de 0.25 ou ausentes recebem coordenadas neutras e máscara zero; coordenadas ausentes nunca são inventadas. O vetor e a ordem estão implementados em Python e JavaScript em `pose-features-v1`; qualquer futura inferência no navegador deve usar esta mesma versão.

## 9. Treinar o modelo

O treino só começa quando o dataset contém, no mínimo provisório, cinco participantes, cinco sessões, cinco grupos independentes participante/sessão, 20 amostras por classe e cada classe presente em pelo menos três grupos. O agrupamento conecta registros que compartilham participante **ou** sessão. Esses gates são salvaguardas operacionais para reduzir avaliação frágil; não são limiares científicos.

```powershell
& .\ml\.venv\Scripts\python.exe ml\scripts\train_posture.py --prepared ml\data\processed\pose-features-v1\dataset.npz
```

O script usa `StratifiedGroupKFold` para criar treino, validação e teste sem compartilhar participantes/sessões. Ele exige as quatro classes em cada partição e recusa treinar se não conseguir uma divisão adequada. A rede inicial tem camadas densas 128/64, dropout, regularização L2, pesos por classe e parada antecipada. Os números e a arquitetura são uma configuração experimental, não um modelo aprovado. Artefatos ficam sob `ml\artifacts\` e são ignorados pelo Git.

## 10. Avaliar

Após o treino, escolha o diretório do run criado e avalie o holdout:

```powershell
$runDir = Get-ChildItem -Path ml\artifacts -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
& .\ml\.venv\Scripts\python.exe ml\scripts\evaluate.py --run-dir $runDir --prepared ml\data\processed\pose-features-v1\dataset.npz
```

`evaluation.json` contém acurácia, precisão, recall, F1 por classe, macro F1, matriz de confusão, métricas por perspectiva e participante, e comparação com o classificador geométrico (incluindo cobertura). `by_challenge_condition` mostra separadamente deitada de lado, agachamento, ajoelhamento e inclinação quando essas condições estiverem anotadas. Categorias sem exemplos no holdout aparecem como `NOT_OBSERVED_IN_HOLDOUT`; resultados com poucos exemplos são apenas descritivos. O relatório também explicita limitações e vieses possíveis. Ele não mede aprovação clínica ou segurança e não usa a acurácia de treino como resultado de generalização.

## 11. Exportar para TensorFlow.js

A exportação só é permitida após treino, holdout completo, correspondência dos hashes e confirmação explícita de revisão humana. Execute no Linux/WSL2, onde o converter e TF-DF estão instalados:

```powershell
wsl
```

No shell Linux/WSL, a partir da pasta compartilhada do projeto (ajuste a unidade se necessário):

```bash
cd "/mnt/c/Users/usuario/Downloads/vz/vida segura IA"
python3.12 -m venv ml/.venv-wsl
source ml/.venv-wsl/bin/activate
python -m pip install --upgrade pip
python -m pip install -r ml/requirements.txt
```

Use o mesmo diretório de artefatos produzido pelo treino e confirme que revisou `evaluation.json`:

```bash
python ml/scripts/export_model.py --run-dir "ml/artifacts/ID_DO_RUN" --confirm-evaluation-reviewed
```

O conversor cria `tfjs_model/model.json`, pesos e `metadata.json`. Ainda não integra o modelo à aplicação principal. Para o smoke test, mantenha o servidor local ativo e abra `http://127.0.0.1:8765/ml/tools/verify-tfjs-export.html?model=../artifacts/ID_DO_RUN/tfjs_model/model.json`. O teste carrega `metadata.json`, chama `tf.loadLayersModel()` e executa uma entrada neutra para verificar formato e dimensões; isso não mede desempenho nem substitui a avaliação do holdout.

## 12. Testes e dados insuficientes

Testes unitários usam poses sintéticas somente como fixtures de código; o validador rejeita `data_origin` diferente de `camera`, e nenhuma fixture é usada para treinar ou avaliar. Rode:

```powershell
& .\ml\.venv\Scripts\python.exe -m unittest discover -s ml\tests -p test_*.py -v
node --test tests/*.test.mjs
```

Os dados são insuficientes quando o relatório mostra amostras rejeitadas, classes ou perspectivas sem cobertura, poucos participantes/sessões/grupos, menos de 20 exemplos por classe, alguma classe em menos de três grupos, ou quando a divisão agrupada não consegue manter as quatro classes no teste. Colete mais participantes e sessões independentes; não contorne os gates repetindo frames ou usando dados simulados. Um holdout que não contém uma perspectiva ou condição desafiadora não permite concluir desempenho para ela.

Até agora nenhum modelo foi treinado ou exportado: isso só deve ocorrer depois da coleta, validação e revisão de dados reais rotulados. A etapa termina neste laboratório isolado; a integração na aplicação fica para uma etapa futura, após validação.

## Random Forest com corpo parcialmente visível (apresentação)

`scripts/train_forest.py` treina e compara três abordagens com validação cruzada agrupada por participante/sessão (`StratifiedGroupKFold`):

- regra geométrica atual (`geometric_prediction` salva pelo coletor);
- regressão logística (baseline);
- Random Forest (`class_weight="balanced"`).

Para lidar com oclusão, o **treino** de cada fold recebe cópias com articulações escondidas (`scripts/occlusion.py`: um lado do corpo, pernas, braços, 25% e 50% aleatórios, baixa confiança). O **teste** é medido sem oclusão e em cada cenário. Abaixo de um limiar de probabilidade a resposta passa a ser "inconclusivo"; o relatório mostra a cobertura e o acerto para cada limiar.

```powershell
.\ml\.venv\Scripts\Activate.ps1
python ml\scripts\train_forest.py ml\data\raw\*.jsonl
```

Saída em `ml/artifacts/forest_<data>/` (ignorado pelo Git): `relatorio.md`, `relatorio.json` e `random_forest.joblib`. O treino é recusado se a diversidade mínima não for atingida.

Ensaio do pipeline sem dados reais (bonecos sintéticos; **as métricas não têm valor e não devem ir para os slides**):

```powershell
python ml\scripts\train_forest.py --demo-sintetico
```

Para medir o efeito da oclusão simulada, rode também com `--sem-oclusao-no-treino` e compare os cenários `lado_*_oculto` e `aleatorio_50`.
