"""
config.py
Configuración centralizada del proyecto: rutas, semilla de reproducibilidad,
listas de columnas y mapeo de macro-géneros.
"""

from pathlib import Path

# ----------------------------------------------------------------------
# RUTAS DEL PROYECTO
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
IMAGES_DIR = BASE_DIR / "images"
RESULTS_DIR = BASE_DIR / "results" / "metrics"

RAW_DATA_PATH = DATA_RAW_DIR / "dataset.csv"
CLEAN_DATA_PATH = DATA_PROCESSED_DIR / "dataset_limpio.csv"
TRAIN_DATA_PATH = DATA_PROCESSED_DIR / "train.csv"
TEST_DATA_PATH = DATA_PROCESSED_DIR / "test.csv"

# ----------------------------------------------------------------------
# REPRODUCIBILIDAD
# ----------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20

# ----------------------------------------------------------------------
# VARIABLE OBJETIVO
# ----------------------------------------------------------------------
TARGET = "popularity"

# ----------------------------------------------------------------------
# COLUMNAS A DESCARTAR (identificadores / texto libre, no predictivas)
# ----------------------------------------------------------------------
DROP_COLS = [
    "Unnamed: 0",
    "track_id",
    "track_name",
    "album_name",
    "artists",
]

# ----------------------------------------------------------------------
# COLUMNAS NUMÉRICAS DE AUDIO (usadas en PCA / K-Means y como features)
# ----------------------------------------------------------------------
AUDIO_COLS = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]

# Numéricas adicionales generadas en feature engineering (fila a fila)
NUMERIC_ENGINEERED_COLS = [
    "duration_min",   # OJO: NO usar junto con duration_ms (colineales, misma info)
    "n_artistas",
    "n_versiones"
]

NUMERIC_COLS = AUDIO_COLS + NUMERIC_ENGINEERED_COLS

# ----------------------------------------------------------------------
# COLUMNA PARA TARGET ENCODING (Alta cardinalidad o señal fuerte)
# ----------------------------------------------------------------------
CAT_TARGET_ENC = ["track_genre"]

# ----------------------------------------------------------------------
# COLUMNAS PARA ONE-HOT ENCODING (Baja cardinalidad)
# ----------------------------------------------------------------------
CAT_ONEHOT = ["key", "time_signature"]



# ----------------------------------------------------------------------
# COLUMNAS BINARIAS (se castean a int, no pasan por OneHot ni escalado)
# ----------------------------------------------------------------------
BINARY_COLS = [
    "explicit",
    "mode",
    "es_colaboracion",
]

# ----------------------------------------------------------------------
# COLUMNA RESERVADA SOLO PARA AUDITORÍA ÉTICA (no entra como feature)
# ----------------------------------------------------------------------
ETHICS_COL = "macro_genero"

# ----------------------------------------------------------------------
# MAPA DE MACRO-GÉNERO (consolida los 114 track_genre en 9 grupos legibles)
# Ajusta/completa esta lista con el mapeo final que definiste en la Fase 4
# del EDA (incluye las correcciones: 'alternative' -> 'rock-metal',
# 'mpb' -> 'world-regional').
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# MAPA DE MACRO-GÉNERO (definido en la celda 51 del EDA, 01_EDA.ipynb)
# Consolida los 114 track_genre en 9 grupos legibles para la auditoría
# ética. NO entra al modelo (eso lo hace track_genre con One-Hot).
# ----------------------------------------------------------------------
MAPA_MACRO_GENERO = {
    # Pop / mainstream
    'pop': 'pop-mainstream', 'pop-film': 'pop-mainstream', 'k-pop': 'pop-mainstream',
    'j-pop': 'pop-mainstream', 'mandopop': 'pop-mainstream', 'cantopop': 'pop-mainstream',
    'indie-pop': 'pop-mainstream', 'synth-pop': 'pop-mainstream', 'power-pop': 'pop-mainstream',
    'j-idol': 'pop-mainstream',

    # Electrónica / dance
    'edm': 'electronica', 'house': 'electronica', 'deep-house': 'electronica',
    'progressive-house': 'electronica', 'chicago-house': 'electronica', 'detroit-techno': 'electronica',
    'techno': 'electronica', 'trance': 'electronica', 'dubstep': 'electronica', 'garage': 'electronica',
    'drum-and-bass': 'electronica', 'electro': 'electronica', 'minimal-techno': 'electronica',
    'club': 'electronica', 'dance': 'electronica', 'breakbeat': 'electronica', 'hardstyle': 'electronica',
    'trip-hop': 'electronica', 'idm': 'electronica', 'electronic': 'electronica', 'dub': 'electronica',

    # Rock / metal
    'rock': 'rock-metal', 'alt-rock': 'rock-metal', 'hard-rock': 'rock-metal', 'punk-rock': 'rock-metal',
    'psych-rock': 'rock-metal', 'indie': 'rock-metal', 'grunge': 'rock-metal', 'metal': 'rock-metal',
    'metalcore': 'rock-metal', 'death-metal': 'rock-metal', 'heavy-metal': 'rock-metal',
    'black-metal': 'rock-metal', 'punk': 'rock-metal', 'rock-n-roll': 'rock-metal', 'goth': 'rock-metal',
    'industrial': 'rock-metal', 'hardcore': 'rock-metal', 'emo': 'rock-metal', 'ska': 'rock-metal',
    'rockabilly': 'rock-metal', 'grindcore': 'rock-metal', 'j-rock': 'rock-metal', 'british': 'rock-metal',

    # Urbano / hip-hop
    'hip-hop': 'urbano', 'reggaeton': 'urbano', 'r-n-b': 'urbano', 'trap': 'urbano',
    'dancehall': 'urbano', 'reggae': 'urbano', 'afrobeat': 'urbano',

    # Clásica / instrumental
    'classical': 'clasica-instrumental', 'opera': 'clasica-instrumental', 'piano': 'clasica-instrumental',
    'acoustic': 'clasica-instrumental', 'ambient': 'clasica-instrumental', 'new-age': 'clasica-instrumental',
    'sleep': 'clasica-instrumental', 'study': 'clasica-instrumental', 'guitar': 'clasica-instrumental',

    # World / regional
    'latin': 'world-regional', 'latino': 'world-regional', 'salsa': 'world-regional', 'samba': 'world-regional',
    'tango': 'world-regional', 'brazil': 'world-regional', 'sertanejo': 'world-regional',
    'pagode': 'world-regional', 'forro': 'world-regional', 'spanish': 'world-regional',
    'turkish': 'world-regional', 'indian': 'world-regional', 'iranian': 'world-regional',
    'malay': 'world-regional', 'french': 'world-regional', 'german': 'world-regional',
    'swedish': 'world-regional', 'world-music': 'world-regional', 'j-dance': 'world-regional',
    'anime': 'world-regional', 'disney': 'world-regional',

    # Folk / singer-songwriter
    'folk': 'folk-singer', 'singer-songwriter': 'folk-singer', 'songwriter': 'folk-singer',
    'country': 'folk-singer', 'honky-tonk': 'folk-singer', 'bluegrass': 'folk-singer', 'gospel': 'folk-singer',
    'blues': 'folk-singer',

    # Jazz / soul / funk
    'jazz': 'jazz-soul-funk', 'soul': 'jazz-soul-funk', 'funk': 'jazz-soul-funk', 'groove': 'jazz-soul-funk',
    'disco': 'jazz-soul-funk',

    # Otros / nicho
    'comedy': 'otros', 'children': 'otros', 'kids': 'otros', 'show-tunes': 'otros',
    'happy': 'otros', 'sad': 'otros', 'chill': 'otros', 'party': 'otros', 'romance': 'otros',
}

MACRO_GENERO_DEFAULT = 'otros'