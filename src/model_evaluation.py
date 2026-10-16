"""
model_evaluation.py
Fase 5 (Evaluación): métricas de regresión sobre Test, diagnóstico de
residuos, feature importance y auditoría ética de sesgo por macro_genero.

El modelo se evalúa aquí con datos NO vistos (Test). macro_genero se usa
únicamente en este archivo, para desglosar el error por grupo — nunca
entra como feature del modelo.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.config import TARGET, ETHICS_COL


# ------------------------------------------------------------------
# MÉTRICAS DE REGRESIÓN
# ------------------------------------------------------------------

def calcular_metricas(y_true, y_pred) -> dict:
    """Calcula MAE, MSE, RMSE y R2."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def tabla_comparativa_train_test(nombre_modelo: str, pipeline: Pipeline,
                                  X_train, y_train, X_test, y_test) -> pd.DataFrame:
    """
    Genera una fila comparando métricas en Train vs. Test, para
    evidenciar si el modelo controla el overfitting.
    """
    pred_train = pipeline.predict(X_train)
    pred_test = pipeline.predict(X_test)

    m_train = calcular_metricas(y_train, pred_train)
    m_test = calcular_metricas(y_test, pred_test)

    return pd.DataFrame({
        "modelo": [nombre_modelo],
        "MAE_train": [m_train["MAE"]], "MAE_test": [m_test["MAE"]],
        "MSE_train": [m_train["MSE"]], "MSE_test": [m_test["MSE"]],
        "RMSE_train": [m_train["RMSE"]], "RMSE_test": [m_test["RMSE"]],
        "R2_train": [m_train["R2"]], "R2_test": [m_test["R2"]],
    })


def construir_tabla_metricas(resultados: list) -> pd.DataFrame:
    """Concatena varias filas de tabla_comparativa_train_test en una sola tabla."""
    return pd.concat(resultados, ignore_index=True)


# ------------------------------------------------------------------
# DIAGNÓSTICO: PREDICHO VS REAL Y RESIDUOS
# ------------------------------------------------------------------

def graficar_predicho_vs_real(y_test, y_pred, guardar_en=None):
    """Grafica predicciones vs. valores reales y la distribución de residuos."""
    residuos = y_test - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(y_test, y_pred, alpha=0.3, s=10)
    lim_min = min(y_test.min(), y_pred.min())
    lim_max = max(y_test.max(), y_pred.max())
    axes[0].plot([lim_min, lim_max], [lim_min, lim_max], color="red", linestyle="--")
    axes[0].set_xlabel("Popularity real")
    axes[0].set_ylabel("Popularity predicha")
    axes[0].set_title("Predicho vs. Real")

    axes[1].hist(residuos, bins=40, alpha=0.7)
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_xlabel("Residuo (real - predicho)")
    axes[1].set_title("Distribución de residuos")

    fig.tight_layout()

    if guardar_en:
        fig.savefig(guardar_en, dpi=150)

    return fig


# ------------------------------------------------------------------
# FEATURE IMPORTANCE
# ------------------------------------------------------------------

def graficar_feature_importance(pipeline: Pipeline, top_n: int = 20, guardar_en=None):
    """
    Extrae la importancia de variables del mejor modelo (asume que el
    paso final del pipeline se llama 'modelo' y expone feature_importances_
    o coef_, típico de RandomForest y Ridge respectivamente).
    """
    preprocesador = pipeline.named_steps["preprocesador"]
    modelo = pipeline.named_steps["modelo"]
    nombres = preprocesador.get_feature_names_out()

    if hasattr(modelo, "feature_importances_"):
        importancias = modelo.feature_importances_
    elif hasattr(modelo, "coef_"):
        importancias = np.abs(modelo.coef_)
    else:
        raise AttributeError(
            "El modelo no expone feature_importances_ ni coef_."
        )

    df_importancia = (
        pd.DataFrame({"feature": nombres, "importancia": importancias})
        .sort_values("importancia", ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(8, max(5, top_n * 0.3)))
    ax.barh(df_importancia["feature"][::-1], df_importancia["importancia"][::-1])
    ax.set_xlabel("Importancia")
    ax.set_title(f"Feature Importance (top {top_n})")
    fig.tight_layout()

    if guardar_en:
        fig.savefig(guardar_en, dpi=150)

    return fig, df_importancia


# ------------------------------------------------------------------
# AUDITORÍA ÉTICA: ERROR POR MACRO_GÉNERO
# ------------------------------------------------------------------

def auditoria_etica_por_genero(df_test: pd.DataFrame, y_test, y_pred) -> pd.DataFrame:
    """
    Calcula MAE, error promedio (sesgo) y popularity real/predicha
    promedio por macro_genero, para detectar si el modelo castiga
    injustamente a ciertos grupos musicales.

    df_test debe contener la columna ETHICS_COL (macro_genero), alineada
    por índice con y_test/y_pred.
    """
    df_aud = pd.DataFrame({
        ETHICS_COL: df_test[ETHICS_COL].values,
        "real": np.asarray(y_test),
        "predicho": np.asarray(y_pred),
    })
    df_aud["error_abs"] = (df_aud["real"] - df_aud["predicho"]).abs()
    df_aud["error_signado"] = df_aud["predicho"] - df_aud["real"]

    resumen = (
        df_aud.groupby(ETHICS_COL)
        .agg(
            n_canciones=("real", "size"),
            popularity_real_media=("real", "mean"),
            popularity_predicha_media=("predicho", "mean"),
            MAE=("error_abs", "mean"),
            sesgo_promedio=("error_signado", "mean"),
        )
        .sort_values("MAE", ascending=False)
        .round(3)
    )

    return resumen


def graficar_auditoria_etica(resumen_auditoria: pd.DataFrame, guardar_en=None):
    """Grafica el MAE por macro_genero, ordenado de mayor a menor error."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(resumen_auditoria.index[::-1], resumen_auditoria["MAE"][::-1])
    ax.set_xlabel("MAE")
    ax.set_title("Auditoría ética: Error (MAE) por macro_genero")
    fig.tight_layout()

    if guardar_en:
        fig.savefig(guardar_en, dpi=150)

    return fig