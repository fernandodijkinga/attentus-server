# Attentus Server — GenMate Field Intelligence

Servidor central de coleta de dados para o piloto de monitoramento de bezerras.
Recebe dados da estação meteorológica (ESP32+BME280+BH1750), calf monitor de baias (ESP32-CAM)
e eventos do brete inteligente Perspicuus (RFID + múltiplas vistas de câmera)
via HTTP, armazena com timestamps e disponibiliza para análise e treinamento de IA.

---

## Deploy no Render

### 1. Criar repositório

```bash
git init attentus-server
cd attentus-server
# copiar todos os arquivos deste pacote
git add . && git commit -m "init"
git remote add origin https://github.com/fernandodijkinga/attentus-server.git
git push -u origin main
```

### 2. Criar Web Service no Render

1. Acesse [render.com](https://render.com) → **New → Web Service**
2. Conecte o repositório GitHub
3. Render detecta o `render.yaml` automaticamente
4. Em **Environment Variables**, defina:
   - `ADMIN_PASS` → sua senha (ex: `Attentus@2024!`)
   - `API_KEY` → chave para os ESP32 (ex: `esp32-piloto-2024`, opcional)
5. Clique **Deploy**
6. Aguarde o build (≈2 min)
7. URL pública: `https://attentus-server.onrender.com`

> **Importante:** O disco persistente (`/data`) está incluído no `render.yaml` (10 GB).
> O plano Render com disco custa ~$14/mês (web $7 + disco $7).
> Para o piloto sem custo, remova o bloco `disk:` — os dados serão perdidos a cada deploy,
> mas funciona para testes iniciais.

---

## Configurar os dispositivos ESP32

### Estação Meteorológica (esp32_sensores_attentus2.ino)

No firmware, configure o `serverURL` via interface web (AP mode) ou compile diretamente:

```cpp
// Na config ou no campo via formulário web:
serverURL = "https://attentus-server.onrender.com/api/sensors"
```

Se usar API_KEY, adicione no firmware antes do `http.POST(...)`:
```cpp
http.addHeader("X-API-Key", "sua-api-key-aqui");
```

Payload JSON esperado (já gerado pelo firmware). Temperatura e umidade vêm do **DHT22**; pressão e altitude do **BMP280**. Podes usar chaves explícitas `dht22_temp_c` / `dht22_humidity` ou os nomes legados abaixo (o servidor aceita ambos):
```json
{
  "device": "SensorNode-01",
  "bh1750_lux": 312.5,
  "dht22_temp_c": 25.3,
  "dht22_humidity": 68.1,
  "bmp280_press_hpa": 997.4,
  "bmp280_alt_m": 891.2,
  "uptime_s": 3600,
  "rssi": -65
}
```

Equivalente com chaves legadas (mesmo conteúdo que o exemplo acima):
```json
{
  "device": "SensorNode-01",
  "bh1750_lux": 312.5,
  "bmp280_temp_c": 25.3,
  "bme280_humidity": 68.1,
  "bmp280_press_hpa": 997.4,
  "bmp280_alt_m": 891.2
}
```

### Calf Monitor de Baia (ESP32CAM_FieldCapture3.ino)

```cpp
serverURL = "https://attentus-server.onrender.com/api/upload"
```

Se usar API_KEY, adicione antes do `http.POST(...)`:
```cpp
http.addHeader("X-API-Key", "sua-api-key-aqui");
```

Upload multipart esperado (já gerado pelo firmware): campos `image`, `device_name`, `capture_id`, `rssi`.

---

## Endpoints da API

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| `POST` | `/api/sensors` | Recebe JSON da estação meteo |
| `POST` | `/api/upload` | Recebe imagem do calf monitor |
| `POST` | `/api/perspicuus/events` | Recebe evento do brete inteligente (RFID + frames) |
| `POST` | `/api/record/perspicuus/<id>/infer` | Inferência MK1 (YOLO + ONNX lateral/posterior); requer login web |
| `GET`  | `/api/weather/data?hours=24&device=X` | JSON para gráficos |
| `GET`  | `/api/calf-monitor/latest` | Última imagem por baia monitorada |
| `GET`  | `/api/image/<device>/<filename>` | Servir imagem |
| `GET`  | `/health` | Healthcheck (sem auth) |
| `GET`  | `/download/weather?device=X` | CSV de dados meteo |
| `GET`  | `/download/images?device=X` | ZIP de imagens com manifest.csv |

---

### Payload Perspicuus (exemplo)

`POST /api/perspicuus/events`

- **application/json:** corpo único com o objeto abaixo (metadados; `path` em cada frame pode ser só texto).
- **multipart/form-data:** campo de formulário **`json`** (string JSON do evento; use `"images": {}` se for enviar ficheiros) + um ficheiro por campo nomeado **`frontal_1`**, **`lateral_2`**, **`posterior_1`**, **`superior_1`**, etc. O servidor grava em disco e substitui `images` pelos URLs `/api/perspicuus/media/...` (visualização na UI autenticada).

```json
{
  "event_id": "brete_01_2026-04-15T14-33-55-182",
  "timestamp_utc": "2026-04-15T17:33:55.182Z",
  "station_id": "brete_01",
  "device_id": "attentus_edge_01",
  "animal": {
    "rfid": "982000123456789",
    "status": "nova passagem",
    "repetition": 1
  },
  "images": {
    "frontal": [{"frame_index": 1, "path": "capturas_brinco/frontal/..." }],
    "lateral": [{"frame_index": 1, "path": "capturas_brinco/lateral/..." }],
    "posterior": [{"frame_index": 1, "path": "capturas_brinco/posterior/..." }],
    "superior": [{"frame_index": 1, "path": "capturas_brinco/superior/..." }]
  },
  "inference_ready": true
}
```

> O endpoint faz upsert por `event_id`: reenvio do mesmo evento atualiza o registro.

---

## Interface Web

| Página | Descrição |
|--------|-----------|
| `/login` | Autenticação |
| `/` | Dashboard com estatísticas e últimas leituras |
| `/weather` | Gráficos temporais de temperatura, lux, pressão, umidade |
| `/calf-monitor` | Grid do calf monitor com última imagem de cada baia |
| `/perspicuus` | Eventos de brete inteligente com RFID e frames por vista |
| `/perspicuus/inferencias` | Gestão das inferências MK1: filtros, médias por vista, JSON e recalcular |
| `/database?tab=weather` | Tabela paginada de dados meteo (editável, deletável) |
| `/database?tab=images` | Tabela paginada de imagens (com prévia, notas, download) |

---

## Estrutura de Dados

### Tabela `weather`
```
id, received_at (UTC), device_name, lux, temp_c, press_hpa, alt_m, humidity, uptime_s, rssi, raw_json
```

### Tabela `images`
```
id, received_at (UTC), device_name, capture_id, filename, filesize, rssi, notes
```

Imagens armazenadas em: `/data/uploads/<device_name>/<device>_<timestamp>_<capture_id>.jpg`

### Tabela `perspicuus_events`
```
event_id, received_at, timestamp_utc, station_id, device_id, animal_rfid, animal_status,
animal_repetition, inference_ready, frontal_json, lateral_json, posterior_json, superior_json,
total_images, raw_json, inference_json, inference_at
```

`inference_json` guarda o pipeline MK1: por vista, **frames** (score por imagem, em sequência) e **traits_mean** (média dos traits nas imagens válidas daquela pose). Preenchido automaticamente no ingest (se `PERSPICUUS_AUTO_INFER=1` e ONNX OK) ou por `POST /api/record/perspicuus/<id>/infer` / botão **Recalcular**.

---

## Variáveis de Ambiente

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `SECRET_KEY` | (gerado) | Chave Flask para sessões |
| `ADMIN_USER` | `admin` | Usuário da interface web |
| `ADMIN_PASS` | `attentus2024` | **Altere antes do deploy!** |
| `API_KEY` | `` (vazio) | Chave dos ESP32 (vazio = sem auth) |
| `DATA_DIR` | `/data` | Diretório do banco e imagens |
| `PERSPICUUS_YOLO_ONNX` | — | Path ao `CowView.onnx` (crop do animal) |
| `PERSPICUUS_LATERAL_ONNX` | — | Modelo iudicium (traits) para vista **lateral** |
| `PERSPICUUS_POSTERIOR_ONNX` | — | Modelo iudicium para vista **posterior** |
| `PERSPICUUS_LATERAL_METADATA_JSON` | — | Opcional (`trait_names`, `input_size`, …) |
| `PERSPICUUS_POSTERIOR_METADATA_JSON` | — | Opcional |
| `PERSPICUUS_YOLO_BBOX_FORMAT` | auto | `xywh` ou `xyxy` se o auto-detetor falhar |
| `PERSPICUUS_AUTO_INFER` | `1` | Se `1`, após cada `POST /api/perspicuus/events` com frames lateral ou posterior, agenda inferência em background (uma imagem de cada vez; média por trait). `0` desliga. |

### Modelos ONNX — nomes de ficheiro e onde colocar

O código **não exige nomes fixos**: cada variável acima deve ser o **caminho absoluto** até ao ficheiro no disco (ex.: `/data/models/cowview.onnx`). Podes usar qualquer nome; uma convenção clara ajuda:

| Ficheiro | Exemplo de nome | Variável |
|----------|-----------------|----------|
| YOLO (CowView / deteção) | `cowview.onnx` ou `NeloreView.onnx` | `PERSPICUUS_YOLO_ONNX` |
| iudicium vista lateral | `iudicium_lateral_fp32.onnx` | `PERSPICUUS_LATERAL_ONNX` |
| iudicium vista posterior | `iudicium_posterior_fp32.onnx` | `PERSPICUUS_POSTERIOR_ONNX` |
| metadata (opcional) | `metadata_lateral.json` | `PERSPICUUS_LATERAL_METADATA_JSON` |
| metadata (opcional) | `metadata_posterior.json` | `PERSPICUUS_POSTERIOR_METADATA_JSON` |

**Local / dev:** coloca os `.onnx` numa pasta (ex.: `attentus-server/data/models/`) e define no ambiente, por exemplo:

```bash
export PERSPICUUS_YOLO_ONNX="/Users/voce/caminho/CowView.onnx"
export PERSPICUUS_LATERAL_ONNX="/Users/voce/caminho/output-ECC/onnx_export/iudicium_fp32.onnx"
export PERSPICUUS_POSTERIOR_ONNX="..."   # outro ficheiro se treinaste modelo separado
export PERSPICUUS_LATERAL_METADATA_JSON="/Users/voce/caminho/output-ECC/onnx_export/metadata.json"
```

**Render.com (disco persistente):** com `DATA_DIR=/data` e disco montado em `/data`, cria uma pasta e copia os ficheiros para lá (não entram no Git por serem pesados):

1. No dashboard: **Shell** do serviço, ou SSH se ativado.
2. Exemplo: `mkdir -p /data/models` e faz **upload** com `scp` a partir do teu PC:  
   `scp CowView.onnx iudicium_*.onnx metadata.json user@render-host:/data/models/`
3. No Render → **Environment**, define por exemplo:  
   `PERSPICUUS_YOLO_ONNX=/data/models/cowview.onnx`  
   `PERSPICUUS_LATERAL_ONNX=/data/models/iudicium_lateral_fp32.onnx`  
   `PERSPICUUS_POSTERIOR_ONNX=/data/models/iudicium_posterior_fp32.onnx`  
   (e os JSON de metadata se existirem.)
4. **Redeploy** ou reinicia o serviço para carregar as novas variáveis.

**Nota:** se lateral e posterior usarem **o mesmo** ONNX por agora, podes apontar as duas variáveis para o **mesmo caminho** de ficheiro. O tamanho do disco em `render.yaml` pode precisar de subir se os modelos forem grandes.

---

## Exportação para Treino de IA

### Dados meteorológicos (CSV)
```
GET /download/weather
```
Colunas: `id, received_at_utc, device_name, lux_lx, temp_c, press_hpa, alt_m, humidity_pct, uptime_s, rssi_dbm`

### Imagens + metadados (ZIP)
```
GET /download/images
```
Contém:
- `manifest.csv` — metadados de cada imagem com timestamps e IDs
- `<device_name>/<filename>.jpg` — imagens organizadas por baia monitorada

---

## Desenvolvimento Local

```bash
cd attentus-server
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Windows:
# venv\Scripts\activate

ADMIN_PASS=admin python app.py
# Acesse: http://localhost:5000
```
Os dados serão salvos em `./data/` (criado automaticamente).

# ATUALIZAR GITHUB
# cd /Users/fernandojeandijkinga/codigos/Render-Attentus/attentus-server
# git add -A
# git commit -m "Descrição clara do que mudou"
# git push origin main