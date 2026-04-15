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
| `GET`  | `/api/weather/data?hours=24&device=X` | JSON para gráficos |
| `GET`  | `/api/calf-monitor/latest` | Última imagem por baia monitorada |
| `GET`  | `/api/image/<device>/<filename>` | Servir imagem |
| `GET`  | `/health` | Healthcheck (sem auth) |
| `GET`  | `/download/weather?device=X` | CSV de dados meteo |
| `GET`  | `/download/images?device=X` | ZIP de imagens com manifest.csv |

---

### Payload Perspicuus (exemplo)

`POST /api/perspicuus/events`

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
total_images, raw_json
```

---

## Variáveis de Ambiente

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `SECRET_KEY` | (gerado) | Chave Flask para sessões |
| `ADMIN_USER` | `admin` | Usuário da interface web |
| `ADMIN_PASS` | `attentus2024` | **Altere antes do deploy!** |
| `API_KEY` | `` (vazio) | Chave dos ESP32 (vazio = sem auth) |
| `DATA_DIR` | `/data` | Diretório do banco e imagens |

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
