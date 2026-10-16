"""
model_training.py
Entrenamiento y optimización de hiperparámetros para la Fase 4b
(Aprendizaje Supervisado): Baseline, Ridge y Random Forest.

Todos los modelos se entrenan dentro de un Pipeline completo
(preprocesador + estimador), ajustado SOLO sobre Train, con
validación cruzada para el tuning.
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor

from src.config import TARGET, RANDOM_STATE
from src.pipeline import construir_preprocesador_lineal, construir_preprocesador_arboles


# ------------------------------------------------------------------
# BASELINE
# ------------------------------------------------------------------

def entrenar_baseline(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Entrena un DummyRegressor(strategy='mean') como piso de comparación."""
    pipeline = Pipeline(steps=[
        ("preprocesador", construir_preprocesador_lineal()),
        ("modelo", DummyRegressor(strategy="mean")),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


# ------------------------------------------------------------------
# MODELO 1: RIDGE (lineal regularizado)
# ------------------------------------------------------------------

def entrenar_ridge(X_train: pd.DataFrame, y_train: pd.Series,
                    cv: int = 5, n_iter: int = 20):
    """
    Entrena Ridge dentro del pipeline completo, con RandomizedSearchCV
    sobre 'alpha' (controla la regularización ante los múltiples dummies
    de track_genre). Devuelve el mejor pipeline y el detalle de la
    búsqueda (para la tabla pre/post tuning).
    """
    pipeline = Pipeline(steps=[
        ("preprocesador", construir_preprocesador_lineal()),
        ("modelo", Ridge(random_state=RANDOM_STATE)),
    ])

    param_grid = {
        "modelo__alpha": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
    }

    busqueda = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=min(n_iter, len(param_grid["modelo__alpha"])),
        cv=cv,
        scoring="neg_root_mean_squared_error",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    busqueda.fit(X_train, y_train)

    return busqueda.best_estimator_, busqueda


# ------------------------------------------------------------------
# MODELO 2: RANDOM FOREST (no lineal)
# ------------------------------------------------------------------

def entrenar_random_forest(X_train: pd.DataFrame, y_train: pd.Series,
                            cv: int = 5, n_iter: int = 10):
    """
    Entrena Random Forest dentro del pipeline completo, con
    RandomizedSearchCV sobre los hiperparámetros clave para controlar
    el dilema sesgo-varianza. Se incluye max_features para manejar la
    alta dimensionalidad del One-Hot Encoder.
    """
    pipeline = Pipeline(steps=[
        ("preprocesador", construir_preprocesador_arboles()),
        ("modelo", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=8)),
    ])

    param_grid = {
        "modelo__n_estimators": [100, 200, 300], 
        "modelo__max_depth": [None, 5, 10, 15, 20],
        "modelo__min_samples_split": [2, 5, 10],
        "modelo__min_samples_leaf": [1, 2, 4],
        "modelo__max_features": [1.0, "sqrt", 0.33], 
    }

    busqueda = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        random_state=RANDOM_STATE,
        n_jobs=8,
    )
    busqueda.fit(X_train, y_train)

    return busqueda.best_estimator_, busqueda

# ------------------------------------------------------------------
# MODELO 3: GRADIENT BOOSTING
# ------------------------------------------------------------------

def entrenar_gradient_boosting(X_train: pd.DataFrame, y_train: pd.Series,
                               cv: int = 5, n_iter: int = 15):
    """
    Entrena HistGradientBoostingRegressor (algoritmo ultra rápido basado
    en histogramas, similar a LightGBM) para corregir los errores de 
    los árboles anteriores de forma secuencial.
    """
    pipeline = Pipeline(steps=[
        ("preprocesador", construir_preprocesador_arboles()),
        ("modelo", HistGradientBoostingRegressor(random_state=RANDOM_STATE)),
    ])

    param_grid = {
        "modelo__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "modelo__max_iter": [100, 200, 300],
        "modelo__max_depth": [5, 10, 15, None],
        "modelo__l2_regularization": [0.0, 0.1, 1.0, 5.0],
    }

    busqueda = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        random_state=RANDOM_STATE,
        n_jobs=8,
    )
    busqueda.fit(X_train, y_train)

    return busqueda.best_estimator_, busqueda

# ------------------------------------------------------------------
# TABLA COMPARATIVA PRE / POST TUNING
# ------------------------------------------------------------------

def tabla_tuning_comparison(nombre_modelo: str, busqueda: RandomizedSearchCV,
                             pipeline_base: Pipeline,
                             X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """
    Compara el RMSE (CV) del modelo con hiperparámetros por defecto
    (pipeline_base, sin tuning) contra el mejor modelo encontrado por
    la búsqueda, usando el mismo esquema de validación cruzada.
    """
    from sklearn.model_selection import cross_val_score

    rmse_base = -cross_val_score(
        pipeline_base, X_train, y_train,
        scoring="neg_root_mean_squared_error", cv=busqueda.cv
    ).mean()

    rmse_tuned = -busqueda.best_score_

    return pd.DataFrame({
        "modelo": [nombre_modelo],
        "rmse_cv_default": [rmse_base],
        "rmse_cv_tuned": [rmse_tuned],
        "mejores_hiperparametros": [str(busqueda.best_params_)],
    })