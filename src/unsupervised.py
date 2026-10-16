"""
unsupervised.py
Funciones para el aprendizaje no supervisado (Fase 4a de CRISP-DM):
PCA sobre variables de audio escaladas y Clustering con K-Means,
incluyendo el cálculo de Codo (inercia) y Coeficiente de Silueta
para justificar el K óptimo.

IMPORTANTE: PCA y K-Means se ajustan (.fit) únicamente sobre Train,
igual que el resto del pipeline, para mantener Test como datos no vistos.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.config import AUDIO_COLS, RANDOM_STATE


# ------------------------------------------------------------------
# PCA
# ------------------------------------------------------------------

def escalar_audio(df: pd.DataFrame, scaler: StandardScaler = None):
    """
    Escala exclusivamente las columnas de audio (AUDIO_COLS) para PCA
    y K-Means. Imputa valores faltantes con la mediana y escala; si no
    se pasa un transformador ya ajustado, se ajusta aquí sobre el df
    recibido (debe ser Train).
    """
    X_audio = df[AUDIO_COLS].copy()

    if scaler is None:
        scaler = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        X_scaled = scaler.fit_transform(X_audio)
    else:
        X_scaled = scaler.transform(X_audio)

    return X_scaled, scaler


def aplicar_pca(X_scaled: np.ndarray, varianza_objetivo: float = 0.85):
    """
    Ajusta PCA sobre X_scaled (Train) reteniendo suficientes componentes
    para alcanzar varianza_objetivo (por defecto 85%, dentro del rango
    80%-90% exigido). Devuelve el objeto PCA ajustado y la proyección.
    """
    pca_full = PCA(random_state=RANDOM_STATE).fit(X_scaled)
    varianza_acumulada = np.cumsum(pca_full.explained_variance_ratio_)
    n_componentes = int(np.argmax(varianza_acumulada >= varianza_objetivo) + 1)

    pca = PCA(n_components=n_componentes, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    return pca, X_pca, varianza_acumulada


def graficar_varianza_explicada(varianza_acumulada: np.ndarray,
                                 guardar_en=None):
    """Grafica varianza explicada individual y acumulada por componente."""
    varianza_individual = np.diff(np.insert(varianza_acumulada, 0, 0))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(1, len(varianza_individual) + 1), varianza_individual,
           alpha=0.6, label="Varianza individual")
    ax.plot(range(1, len(varianza_acumulada) + 1), varianza_acumulada,
            marker="o", color="orange", label="Varianza acumulada")
    ax.axhline(0.80, color="gray", linestyle="--", linewidth=1, label="80%")
    ax.axhline(0.90, color="gray", linestyle=":", linewidth=1, label="90%")
    ax.set_xlabel("Componente principal")
    ax.set_ylabel("Varianza explicada")
    ax.set_title("PCA - Varianza explicada individual y acumulada")
    ax.legend()
    fig.tight_layout()

    if guardar_en:
        fig.savefig(guardar_en, dpi=150)

    return fig


# ------------------------------------------------------------------
# K-MEANS: BÚSQUEDA DE K ÓPTIMO (Codo + Silueta)
# ------------------------------------------------------------------

def evaluar_rango_k(X: np.ndarray, k_min: int = 2, k_max: int = 10, sample_size: int = 2000):
    """Estima inercia y silueta sobre una muestra fija de X."""
    resultados = {"k": [], "inercia": [], "silueta": []}
    n_muestra = min(sample_size, len(X))
    indices = np.random.default_rng(RANDOM_STATE).choice(
        len(X), size=n_muestra, replace=False
    )
    X_muestra = X[indices]

    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(X_muestra)

        resultados["k"].append(k)
        resultados["inercia"].append(kmeans.inertia_)
        resultados["silueta"].append(
            silhouette_score(X_muestra, labels)
        )

    return pd.DataFrame(resultados)


def graficar_codo_y_silueta(resultados: pd.DataFrame, guardar_en=None):
    """Grafica lado a lado la curva del codo (inercia) y el coef. de silueta."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(resultados["k"], resultados["inercia"], marker="o")
    axes[0].set_xlabel("K (n° de clusters)")
    axes[0].set_ylabel("Inercia")
    axes[0].set_title("Método del Codo")

    axes[1].plot(resultados["k"], resultados["silueta"], marker="o", color="green")
    axes[1].set_xlabel("K (n° de clusters)")
    axes[1].set_ylabel("Coeficiente de Silueta")
    axes[1].set_title("Coeficiente de Silueta por K")

    fig.tight_layout()

    if guardar_en:
        fig.savefig(guardar_en, dpi=150)

    return fig


# ------------------------------------------------------------------
# K-MEANS FINAL (con K óptimo ya elegido)
# ------------------------------------------------------------------

def entrenar_kmeans_final(X: np.ndarray, k_optimo: int):
    """Ajusta el K-Means definitivo con el K óptimo seleccionado."""
    kmeans = KMeans(n_clusters=k_optimo, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(X)
    return kmeans, labels


def graficar_clusters_pca_2d(X_pca: np.ndarray, labels: np.ndarray,
                              guardar_en=None):
    """Proyecta los clusters sobre los dos primeros componentes de PCA."""
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels,
                          cmap="tab10", alpha=0.6, s=10)
    ax.set_xlabel("Componente Principal 1")
    ax.set_ylabel("Componente Principal 2")
    ax.set_title("Clusters K-Means proyectados en PCA (2D)")
    legend = ax.legend(*scatter.legend_elements(), title="Cluster")
    ax.add_artist(legend)
    fig.tight_layout()

    if guardar_en:
        fig.savefig(guardar_en, dpi=150)

    return fig


def perfilar_clusters(df: pd.DataFrame, labels: np.ndarray,
                       columnas_perfil=None) -> pd.DataFrame:
    """
    Calcula el promedio de las variables clave (incluida popularity) por
    cluster, para darles un "nombre de negocio" en el notebook.
    """
    if columnas_perfil is None:
        columnas_perfil = ["danceability", "energy", "acousticness",
                            "valence", "tempo", "popularity"]

    df_perfil = df.copy()
    df_perfil["cluster"] = labels

    return df_perfil.groupby("cluster")[columnas_perfil].mean().round(3)
