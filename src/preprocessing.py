"""
preprocessing.py
ETL inicial y feature engineering determinista (fila a fila), aplicados
ANTES del split. No usa estadísticos agregados entre filas, por lo que
no hay riesgo de fuga de datos en este archivo.

El escalado, la imputación por mediana/moda y el One-Hot Encoding NO van
aquí: se ajustan únicamente sobre Train dentro del pipeline (ver
src/pipeline.py).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# IMPORT CORREGIDO: AUDIO_COLS movido al encabezado (PEP-8)
from src.config import (
    RAW_DATA_PATH,
    TARGET,
    RANDOM_STATE,
    TEST_SIZE,
    MAPA_MACRO_GENERO,
    MACRO_GENERO_DEFAULT,
    AUDIO_COLS 
)


def cargar_datos(path=RAW_DATA_PATH) -> pd.DataFrame:
    """Carga el dataset crudo desde disco."""
    return pd.read_csv(path)


def limpieza_estructural(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza estructural básica (sin tocar escalado ni imputación):
    - Elimina columna basura.
    - Elimina la fila con nulos triples (artists/track_name/album_name).
    - Elimina la canción con duration_ms == 0.
    """
    df = df.copy()

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df = df.dropna(subset=["artists", "track_name", "album_name"], how="all")
    df = df[df["duration_ms"] != 0]

    return df.reset_index(drop=True)


def feature_engineering_por_fila(df: pd.DataFrame) -> pd.DataFrame:
    """
    Operaciones deterministas fila a fila aplicadas antes de agregar.
    """
    df = df.copy()

    df["tempo"] = df["tempo"].replace(0, np.nan)
    df["time_signature"] = df["time_signature"].where(df["time_signature"] >= 3, np.nan)
    df["duration_min"] = df["duration_ms"] / 60000
    df = df.drop(columns=["duration_ms"])

    df["n_artistas"] = df["artists"].fillna("").apply(lambda x: len(x.split(";")) if x else 1)
    df["es_colaboracion"] = (df["n_artistas"] > 1).astype(int)

    df["macro_genero"] = df["track_genre"].map(MAPA_MACRO_GENERO).fillna(MACRO_GENERO_DEFAULT)

    return df


def deduplicar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Consolidación semántica en 2 pasos:
    1. Elimina duplicados exactos de sistema.
    2. Agrega duplicados semánticos calculando el máximo de popularidad,
       promediando el perfil acústico y conservando la trazabilidad del track_id.
    """
    df = df.copy()

    # Paso 1: Limpieza de sistema
    df = df.drop_duplicates(subset=["track_id"], keep="first")

    # Truco de Pandas: Crear columna temporal para sumar y obtener n_versiones
    df["n_versiones"] = 1

    # Paso 2: Agregación Semántica
    agg_dict = {
        TARGET: "max",
        "track_id": "first",     # CONSERVAMOS EL ID (para auditar outliers luego)
        "n_versiones": "sum",    # CONTEO REAL DE VERSIONES (1, 2, 3...)
        "macro_genero": lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0],
        "track_genre": lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0],
        "explicit": "max",
        "mode": lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0],
        "key": lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0],
        "time_signature": lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0],
        "duration_min": "mean",
        "n_artistas": "first",
        "es_colaboracion": "first"
    }

    # Asignar promedio simple a todos los atributos acústicos
    for col in AUDIO_COLS:
        agg_dict[col] = "mean"

    # Agrupamos rompiendo empates aleatorios y ordenando por popularidad
    df = df.sample(frac=1, random_state=RANDOM_STATE).sort_values(TARGET, ascending=False)
    df_agg = df.groupby(["track_name", "artists"], as_index=False).agg(agg_dict)

    # Si quisieras la bandera binaria (redundante si ya tienes n_versiones continuas)
    # df_agg["es_version_multiple"] = (df_agg["n_versiones"] > 1).astype(int)

    return df_agg


def dividir_train_test(df: pd.DataFrame):
    """
    Split 80/20, ejecutado DESPUÉS de deduplicar y ANTES de cualquier
    imputación, escalado o encoding.
    """
    df = df.copy()
    bins = [-1, 0, 20, 40, 60, 80, 100]
    popularity_bucket = pd.cut(df[TARGET], bins=bins)

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=popularity_bucket,
    )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def ejecutar_pipeline_preparacion(path=RAW_DATA_PATH):
    """
    Orquesta Fase 3 (hasta el split): carga -> limpieza -> feature
    engineering por fila -> deduplicación -> split.
    Devuelve (df_limpio, train_df, test_df).
    """
    df = cargar_datos(path)
    df = limpieza_estructural(df)
    df = feature_engineering_por_fila(df)
    df_limpio = deduplicar(df)
    train_df, test_df = dividir_train_test(df_limpio)

    return df_limpio, train_df, test_df


if __name__ == "__main__":
    from src.config import CLEAN_DATA_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH

    df_limpio, train_df, test_df = ejecutar_pipeline_preparacion()

    CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    df_limpio.to_csv(CLEAN_DATA_PATH, index=False)
    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)

    print(f"dataset_limpio: {df_limpio.shape}")
    print(f"train: {train_df.shape}")
    print(f"test:  {test_df.shape}")