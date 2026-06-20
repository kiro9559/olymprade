import websocket
import json
import time
import pandas as pd
import pickle
import os
import base64
import requests
from sklearn.ensemble import RandomForestClassifier
import numpy as np

APP_ID = "1089"
GITHUB_TOKEN = ''
GITHUB_REPO  = ''
GITHUB_BRANCH = "main"
print(f"TOKEN OK: {bool(os.environ.get('GITHUB_TOKEN'))}")
print(f"TOKEN OK: {bool(os.environ.get('GITHUB_REPO'))}")
datos = []
ultimo_proceso = 0
INTERVALO = 15
modelo = None

# ── GitHub ────────────────────────────────────────────────

def guardar_en_github(filepath, contenido_bytes, mensaje):
    try:
        url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{filepath}"
        headers = {
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json"
        }
        r = requests.get(url, headers=headers)
        sha = r.json().get('sha') if r.status_code == 200 else None
        data = {
            "message": mensaje,
            "content": base64.b64encode(contenido_bytes).decode(),
            "branch": GITHUB_BRANCH
        }
        if sha:
            data["sha"] = sha
        requests.put(url, headers=headers, json=data)
        print(f"✓ {filepath} guardado en GitHub")
    except Exception as e:
        print(f"⚠️ Error GitHub: {e}")

def cargar_de_github(filepath):
    try:
        url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{filepath}"
        headers = {
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json"
        }
        r = requests.get(url, headers=headers)
        if r.status_code == 200:
            return base64.b64decode(r.json()['content'])
    except Exception as e:
        print(f"⚠️ Error cargando GitHub: {e}")
    return None

# ── Persistencia ──────────────────────────────────────────

def guardar_modelo(m):
    import io
    buf = io.BytesIO()
    pickle.dump(m, buf)
    contenido = buf.getvalue()
    with open('modelo_ia.pkl', 'wb') as f:
        f.write(contenido)
    guardar_en_github('modelo_ia.pkl', contenido, '🤖 Update modelo')
    print("✓ Modelo guardado")

def cargar_modelo():
    contenido = cargar_de_github('modelo_ia.pkl')
    if contenido:
        import io
        try:
            m = pickle.load(io.BytesIO(contenido))
            print("✓ Modelo cargado desde GitHub")
            return m
        except Exception:
            pass
    if os.path.exists('modelo_ia.pkl'):
        try:
            with open('modelo_ia.pkl', 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass
    print("⚠️ Sin modelo previo, empezando desde cero")
    return None

def guardar_datos(df):
    contenido = df.to_csv(index=False).encode()
    with open('datos_historicos.csv', 'wb') as f:
        f.write(contenido)
    guardar_en_github('datos_historicos.csv', contenido, '📊 Update datos')
    print("✓ Datos guardados")

def cargar_datos():
    contenido = cargar_de_github('datos_historicos.csv')
    if contenido:
        import io
        try:
            df = pd.read_csv(io.BytesIO(contenido))
            if not df.empty:
                print(f"✓ {len(df)} velas cargadas desde GitHub")
                return df.to_dict('records')
        except Exception:
            pass
    if os.path.exists('datos_historicos.csv'):
        try:
            df = pd.read_csv('datos_historicos.csv')
            if not df.empty:
                print(f"✓ {len(df)} velas cargadas desde disco")
                return df.to_dict('records')
        except Exception:
            pass
    return []

# ── Indicadores ───────────────────────────────────────────

def calcular_rsi(serie, periodo=14):
    delta = serie.diff()
    ganancia = delta.where(delta > 0, 0).rolling(periodo).mean()
    perdida  = (-delta.where(delta < 0, 0)).rolling(periodo).mean()
    return 100 - (100 / (1 + ganancia / perdida))

def agregar_indicadores(df):
    df['media_7']       = df['close'].rolling(7).mean()
    df['media_20']      = df['close'].rolling(20).mean()
    df['rsi']           = calcular_rsi(df['close'])
    df['donchian_alto'] = df['high'].rolling(15).max()
    df['donchian_bajo'] = df['low'].rolling(15).min()
    df['donchian_medio']= (df['donchian_alto'] + df['donchian_bajo']) / 2
    df['momentum']      = df['close'].pct_change(5)
    df['volatilidad']   = df['close'].rolling(10).std()
    df['rango']         = df['high'] - df['low']
    df['pos_canal']     = (df['close'] - df['donchian_bajo']) / (df['donchian_alto'] - df['donchian_bajo'] + 0.0001)
    df['cruce_media']   = df['media_7'] - df['media_20']
    return df

# ── IA ────────────────────────────────────────────────────

FEATURES = [
    'open', 'high', 'low', 'close', 'rango',
    'media_7', 'media_20', 'rsi',
    'donchian_alto', 'donchian_bajo', 'donchian_medio',
    'momentum', 'volatilidad', 'pos_canal', 'cruce_media'
]

def preparar_datos(df):
    df = agregar_indicadores(df.copy())
    df['direccion'] = (df['close'] - df['open']).apply(
        lambda x: 2 if x > 0.5 else (0 if x < -0.5 else 1)
    )
    df = df.dropna().reset_index(drop=True)
    if len(df) < 2:
        return None, None
    return df[FEATURES].values[:-1], df['direccion'].values[1:]

def entrenar_modelo(df):
    if len(df) < 30:
        return None
    X, y = preparar_datos(df)
    if X is None:
        return None
    m = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    m.fit(X, y)
    print("✓ Modelo entrenado")
    return m

def predecir(m, df):
    df = agregar_indicadores(df.copy()).dropna().reset_index(drop=True)
    if df.empty:
        return
    X = np.array([[df.iloc[-1][f] for f in FEATURES]])
    pred  = m.predict(X)[0]
    proba = m.predict_proba(X)[0]
    proba_dict = {c: p for c, p in zip(m.classes_, proba)}
    dirs = {0: '🔴 Baja', 1: '🟡 Lateral', 2: '🟢 Sube'}
    print(f"\n📊 Predicción: {dirs[pred]}")
    print(f"🔴 Baja: {proba_dict.get(0,0)*100:.0f}% | 🟡 Lateral: {proba_dict.get(1,0)*100:.0f}% | 🟢 Sube: {proba_dict.get(2,0)*100:.0f}%\n")

# ── WebSocket ─────────────────────────────────────────────

def on_message(ws, message):
    global datos, modelo, ultimo_proceso
    data = json.loads(message)

    if data.get('msg_type') == 'ping':
        ws.send(json.dumps({
            "ticks_history": "1HZ100V",
            "style": "candles",
            "granularity": 3600,
            "count": 100,
            "end": "latest",
            "subscribe": 1
        }))

    elif data.get('msg_type') in ('candles', 'ticks_history', 'ohlc'):
        ahora = time.time()
        if ahora - ultimo_proceso < INTERVALO:
            return
        ultimo_proceso = ahora

        velas_raw = data.get('candles') or data.get('ohlc')
        if not velas_raw:
            return
        if isinstance(velas_raw, dict):
            velas_raw = [velas_raw]

        for v in velas_raw:
            datos.append({
                'tiempo': v.get('epoch', v.get('open_time')),
                'open':   float(v['open']),
                'high':   float(v['high']),
                'low':    float(v['low']),
                'close':  float(v['close'])
            })

        df = pd.DataFrame(datos).drop_duplicates('tiempo').reset_index(drop=True)
        datos = df.to_dict('records')

        print(f"✓ Velas totales: {len(df)}")
        guardar_datos(df)

        if len(df) >= 30:
            modelo = entrenar_modelo(df)
            if modelo:
                guardar_modelo(modelo)
                predecir(modelo, df)

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws, *args):
    print("Conexión cerrada")

def on_open(ws):
    global ultimo_proceso, datos, modelo
    ultimo_proceso = 0
    datos  = cargar_datos()
    modelo = cargar_modelo()
    print("✓ Conectado!")
    ws.send(json.dumps({"ping": 1}))

# ── Inicio ────────────────────────────────────────────────

while True:
    try:
        websocket.WebSocketApp(
            f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}",
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        ).run_forever()
    except Exception as e:
        print(f"Error: {e}")
    print("🔄 Reconectando en 5 segundos...")
    time.sleep(5)
