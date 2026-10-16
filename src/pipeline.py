"""
pipeline.py
Ensambla el ColumnTransformer + Pipeline de scikit-learn. Todo lo que
aprende estadísticos (medianas, modas, categorías de One-Hot, escalas)
se ajusta ÚNICAMENTE sobre Train (vía .fit / .fit_transform), nunca
sobre el dataset completo.

macro_genero NO entra aquí: se reserva para la auditoría ética en
model_evaluation.py y nunca se usa como feature del modelo.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, TargetEncoder

from src.config import NUMERIC_COLS, CAT_ONEHOT, CAT_TARGET_ENC, BINARY_COLS


def _to_int(X):
    """Castea columnas binarias (0/1) a int, sin escalar ni imputar."""
    return X.astype(int)


def construir_preprocesador_lineal() -> ColumnTransformer:
    """Preprocesador para Ridge y Baseline: Usa One-Hot Encoding para TODAS las categóricas."""
    rama_numerica = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    rama_onehot = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    rama_binaria = Pipeline(steps=[
        ("to_int", FunctionTransformer(_to_int, feature_names_out="one-to-one")),
    ])

    # Unimos CAT_TARGET_ENC (track_genre) y CAT_ONEHOT (key, time_signature) para pasarlos por One-Hot
    todas_categoricas = CAT_TARGET_ENC + CAT_ONEHOT

    return ColumnTransformer(
        transformers=[
            ("num", rama_numerica, NUMERIC_COLS),
            ("ohe", rama_onehot, todas_categoricas),
            ("bin", rama_binaria, BINARY_COLS),
        ],
        remainder="drop",
    )

def construir_preprocesador_arboles() -> ColumnTransformer:
    """Preprocesador para RF y GB: Usa Target Encoding para track_genre y One-Hot para el resto."""
    rama_numerica = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    rama_onehot = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    rama_target = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("target_enc", TargetEncoder(target_type="continuous", smooth="auto")),
    ])

    rama_binaria = Pipeline(steps=[
        ("to_int", FunctionTransformer(_to_int, feature_names_out="one-to-one")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", rama_numerica, NUMERIC_COLS),
            ("ohe", rama_onehot, CAT_ONEHOT),
            ("te", rama_target, CAT_TARGET_ENC),
            ("bin", rama_binaria, BINARY_COLS),
        ],
        remainder="drop",
    )

def obtener_nombres_features(preprocesador: ColumnTransformer) -> list:
    """
    Devuelve los nombres de las columnas de salida del ColumnTransformer
    ya ajustado (útil para Feature Importance en model_evaluation.py).
    Requiere que el preprocesador ya haya sido fit().
    """
    return list(preprocesador.get_feature_names_out())