import websocket
import json
import time
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
import numpy as np

APP_ID = "1089"
TOKEN = ""
datos = []
ultimo_proceso = 0
INTERVALO = 1020
modelo = None

# ── Persistencia ──────────────────────────────────────────

def guardar_modelo(m):
    with open('modelo_ia.pkl', 'wb') as f:
        pickle.dump(m, f)
    print("✓ Modelo guardado")

def cargar_modelo():
    if os.path.exists('modelo_ia.pkl'):
        with open('modelo_ia.pkl', 'rb') as f:
            m = pickle.load(f)
        print("✓ Modelo cargado desde disco")
        return m
    print("⚠️ No hay modelo guardado, empezando desde cero")
    return None

def guardar_datos(df):
    df.to_csv('datos_historicos.csv', index=False)
    print("✓ Datos guardados")

def cargar_datos():
    if os.path.exists('datos_historicos.csv'):
        df = pd.read_csv('datos_historicos.csv')
        print(f"✓ {len(df)} velas cargadas desde disco")
        return df.to_dict('records')
    return []

# ── Indicadores ───────────────────────────────────────────

def calcular_rsi(serie, periodo=14):
    delta = serie.diff()
    ganancia = delta.where(delta > 0, 0).rolling(periodo).mean()
    perdida = (-delta.where(delta < 0, 0)).rolling(periodo).mean()
    rs = ganancia / perdida
    return 100 - (100 / (1 + rs))

def agregar_indicadores(df):
    # Medias móviles
    df['media_7']  = df['close'].rolling(7).mean()
    df['media_20'] = df['close'].rolling(20).mean()

    # RSI
    df['rsi'] = calcular_rsi(df['close'])

    # Donchian Channel
    df['donchian_alto'] = df['high'].rolling(15).max()
    df['donchian_bajo'] = df['low'].rolling(15).min()
    df['donchian_medio'] = (df['donchian_alto'] + df['donchian_bajo']) / 2

    # Momentum y volatilidad
    df['momentum']    = df['close'].pct_change(5)
    df['volatilidad'] = df['close'].rolling(10).std()
    df['rango']       = df['high'] - df['low']

    # Posición dentro del canal Donchian
    df['pos_canal'] = (df['close'] - df['donchian_bajo']) / (
        df['donchian_alto'] - df['donchian_bajo'] + 0.0001
    )

    # Cruce de medias
    df['cruce_media'] = (df['media_7'] - df['media_20'])

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
    X = df[FEATURES].values[:-1]
    y = df['direccion'].values[1:]
    return X, y

def entrenar_modelo(df):
    if len(df) < 30:
        print("⚠️ Pocos datos para entrenar")
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
    df = agregar_indicadores(df.copy())
    df = df.dropna().reset_index(drop=True)
    if len(df) == 0:
        return
    ultima = df.iloc[-1]
    X = np.array([[ultima[f] for f in FEATURES]])
    pred  = m.predict(X)[0]
    proba = m.predict_proba(X)[0]
    clases = m.classes_
    proba_dict = {c: p for c, p in zip(clases, proba)}
    direcciones = {0: '🔴 Baja', 1: '🟡 Lateral', 2: '🟢 Sube'}
    print(f"\n📊 Predicción: {direcciones[pred]}")
    print(f"🔴 Baja: {proba_dict.get(0,0)*100:.0f}% | "
          f"🟡 Lateral: {proba_dict.get(1,0)*100:.0f}% | "
          f"🟢 Sube: {proba_dict.get(2,0)*100:.0f}%\n")

# ── WebSocket ─────────────────────────────────────────────

def on_message(ws, message):
    global datos, modelo, ultimo_proceso
    data = json.loads(message)

    if data.get('msg_type') == 'ping':
        print("✓ Ping OK - solicitando velas...")
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
        print(df.tail(3))

        guardar_datos(df)

        if len(df) >= 30:
            modelo = entrenar_modelo(df)
            if modelo:
                guardar_modelo(modelo)
                predecir(modelo, df)

def on_error(ws, error):
    print("Error:", error)

def on_close(ws, close_status, close_msg):
    print("Conexión cerrada — datos y modelo guardados en disco")

def on_open(ws):
    global ultimo_proceso, datos, modelo
    ultimo_proceso = 0
    datos = cargar_datos()
    modelo = cargar_modelo()
    print("✓ Conectado!")
    ws.send(json.dumps({"ping": 1}))

# ── Inicio ────────────────────────────────────────────────

while True:
    try:
        ws = websocket.WebSocketApp(
            f"wss://ws.derivws.com/websockets/v3?app_id={APP_ID}",
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        ws.run_forever()
    except Exception as e:
        print(f"Error inesperado: {e}")

    print("🔄 Reconectando en 5 segundos...")
    time.sleep(5)
