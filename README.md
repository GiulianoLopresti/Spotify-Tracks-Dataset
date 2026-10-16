# Predicción de Popularidad de Canciones en Spotify
## Caso C — Inteligencia musical y predicción de popularidad de canciones

**Asignatura:** MLY1101 — Machine Learning | Evaluación Parcial N°2
**Metodología:** CRISP-DM

---

## 1. Descripción del problema de negocio

Una discográfica/distribuidora digital necesita decidir en qué canciones de su catálogo invertir presupuesto de marketing y promoción editorial (playlists, A&R), pero no puede evaluar manualmente miles de pistas. El problema de negocio es **priorizar, de forma objetiva y escalable, qué canciones tienen mayor potencial de éxito** a partir de sus atributos de audio y metadatos, antes de comprometer presupuesto.

No se busca predecir el valor exacto de popularidad (tarea de alta incertidumbre, dado que factores externos al audio —fama del artista, presupuesto de marketing, viralidad en redes— explican gran parte de la varianza), sino usar el modelo como **herramienta de filtrado y priorización de catálogo**.

## 2. Objetivos del proyecto

- Construir un pipeline reproducible y sin fuga de datos que prepare el dataset de Spotify Tracks para modelamiento.
- Implementar y comparar modelos de aprendizaje supervisado (regresión) para predecir `popularity`.
- Aplicar una técnica de aprendizaje no supervisado (PCA + K-Means) para descubrir perfiles sonoros/segmentos de canciones.
- Evaluar el desempeño de los modelos con métricas apropiadas y auditar posibles sesgos éticos por género musical.

## 3. Definición de KPIs

| KPI | Definición | Resultado obtenido |
|---|---|---|
| **KPI técnico — RMSE** | Error cuadrático medio de la predicción de `popularity` en Test (escala 0–100). | Random Forest: 14.58 / Gradient Boosting: 14.73 / Ridge: 15.08 (vs. 19.44 del Baseline) |
| **KPI de negocio — Captura de Top 20%** | Si el equipo comercial invierte solo en el 20% de canciones mejor puntuadas por el modelo, ¿qué % de los verdaderos éxitos (Top 20% real) logra capturar, comparado con una selección aleatoria (20%)? | Recall Top 20% ≈ 53% (lift de ~2.7x sobre selección aleatoria); popularidad real promedio de la cartera priorizada +16.6 puntos sobre el promedio del catálogo |
| **KPI ético — Equidad de error por género** | MAE desagregado por `macro_genero`, para detectar si el modelo penaliza sistemáticamente a géneros de nicho. | MAE entre 7.3 (world-regional) y 16.9 (jazz-soul-funk); parte de la brecha se explica por tamaño muestral desigual entre grupos |

## 4. Descripción de las fuentes de datos

- **Dataset:** [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) (Kaggle), 114.000 filas × 21 columnas.
- **Variables:** metadatos (`track_name`, `artists`, `album_name`, `track_genre`), variable objetivo (`popularity`, 0–100), y 9 atributos de audio provistos por la API de Spotify (`danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`), además de `key`, `mode`, `time_signature`, `duration_ms` y `explicit`.
- **Limitación conocida:** el dataset no incluye variables de contexto comercial (presupuesto de marketing, fecha de lanzamiento, streams históricos, presencia en playlists editoriales), lo que acota el techo de desempeño predictivo alcanzable solo con audio.

## 5. Preparación y análisis exploratorio de los datos (EDA)

Desarrollado en `notebooks/01_EDA.ipynb`. Hallazgos clave:

- **Calidad de datos:** 1 fila con nulos triples (`artists`/`track_name`/`album_name`) y 1 canción con `duration_ms = 0`, eliminadas en limpieza estructural. `tempo = 0` (157 casos) y `time_signature < 3` (1.136 casos) identificados como fallas de detección del algoritmo de Spotify, tratados como `NaN` e imputados posteriormente con mediana/moda.
- **Duplicados:** se detectaron dos niveles — duplicados de sistema (mismo `track_id`, atributos idénticos) y duplicados semánticos (misma canción con distinto `track_id`: remixes, relanzamientos, versiones acústicas). Consolidados en una sola fila por canción (ver sección 7), reduciendo el dataset de 113.999 a 81.343 canciones únicas (-28.6%).
- **Señal predictiva dominante:** `track_genre` explica el 27.4% de la varianza de `popularity`, muy por encima de cualquier atributo de audio individual o de `macro_genero` (agrupación en 9 categorías), que explica solo 2.2%. Esta brecha motivó usar `track_genre` como variable de alta cardinalidad en el modelo, reservando `macro_genero` exclusivamente para la auditoría ética (114 grupos son ilegibles en una auditoría; 9 son interpretables).
- **Multicolinealidad:** `energy`, `loudness` y `acousticness` están fuertemente correlacionadas entre sí, justificando el uso de PCA antes del clustering no supervisado.
- **Distribución del target:** `popularity` tiene un 14.1% de valores en cero y una distribución multimodal, por lo que el split Train/Test se estratificó por tramos (bins) de popularidad para preservar esa forma en ambos conjuntos.
- **Impacto ético preliminar:** se identificó que algunos géneros (especialmente los regionales/de nicho) tienen menor representación en el dataset, lo que anticipaba el riesgo de mayor error de predicción en esos grupos — confirmado posteriormente en la Fase 5.

## 6. Metodología utilizada (CRISP-DM)

El proyecto siguió las 6 fases de CRISP-DM:

1. **Comprensión del negocio:** definición del problema, KPIs y objetivos (secciones 1–3 de este documento).
2. **Comprensión de los datos:** exploración inicial, calidad de datos, identificación de variables relevantes (`01_EDA.ipynb`).
3. **Preparación de los datos** (`02_Preparacion_y_No_Supervisado.ipynb`, `src/preprocessing.py`):
   - Limpieza estructural (nulos, `duration_ms = 0`).
   - Feature engineering fila a fila, sin fuga: `tempo`/`time_signature` anómalos → `NaN`; `duration_min` (reemplaza a `duration_ms`); `n_artistas` y `es_colaboracion`; `macro_genero` (solo para auditoría ética).
   - **Deduplicación en dos pasos:** (a) eliminación de duplicados exactos por `track_id`; (b) agregación de duplicados semánticos (`track_name` + `artists`) — variables de audio con **promedio simple** (nunca ponderado por `popularity`, para evitar fuga de información del target hacia las features), `popularity` con **máximo** del grupo, `n_versiones` como **conteo real**, y categóricas con **moda determinista**.
   - Split Train/Test 80/20 estratificado por tramos de `popularity`, ejecutado **después** de deduplicar y **antes** de cualquier imputación/escalado/encoding (que se ajustan únicamente sobre Train dentro del pipeline, evitando la fuga de datos detectada en la EP1).
4. **Modelado:**
   - **No supervisado** (`src/unsupervised.py`): `StandardScaler` + PCA (ajustados solo en Train) sobre las 9 variables de audio, reteniendo componentes hasta ~85–90% de varianza explicada. K-Means evaluado con método del Codo y Coeficiente de Silueta para distintos K, seleccionando el K final balanceando métrica e interpretabilidad de negocio.
   - **Supervisado** (`src/model_training.py`): Baseline (`DummyRegressor`), Regresión **Ridge** (lineal, regularización L2) y **Random Forest** Regressor (no lineal, ensamble), ambos con `RandomizedSearchCV` para tuning de hiperparámetros. Se añadió además **Gradient Boosting** (`HistGradientBoostingRegressor`) como tercer modelo exploratorio.
   - **Encoding diferenciado por familia de modelo** (`src/pipeline.py`): Ridge usa One-Hot Encoding para todas las categóricas (necesario para que el modelo lineal capture el efecto de cada género por separado, dada su alta importancia explicativa). Random Forest y Gradient Boosting usan Target Encoding para `track_genre` (alta cardinalidad) + One-Hot para `key`/`time_signature` (baja cardinalidad), aprovechando que los modelos de árboles manejan mejor variables categóricas condensadas.
5. **Evaluación** (`src/model_evaluation.py`): métricas de regresión (MAE, MSE, RMSE, R²) comparando Train vs. Test, diagnóstico de residuos, feature importance, evaluación del KPI de negocio (captura de Top 20%) y auditoría ética de sesgo por `macro_genero`.
6. **Despliegue / entrega:** este informe técnico, notebooks reproducibles, modelos serializados (`models/*.pkl`) y estructura de proyecto documentada.

## 7. Resultados del modelado

### 7.1 Aprendizaje no supervisado (PCA + K-Means)

PCA redujo las 9 variables de audio a componentes que explican ~85–90% de la varianza. K-Means se evaluó entre K=2 y K=10; aunque la silueta máxima se dio en K=2, se priorizó un K mayor por interpretabilidad de negocio, obteniendo grupos con perfiles sonoros diferenciados (ej. clusters dominados por metal/hardstyle, por música brasileña/regional, y por ambient/clásica/new-age). Los clusters, formados únicamente con variables de audio, mostraron una correspondencia clara con géneros musicales reales sin haber usado `track_genre` como input — validando que las variables de audio capturan señal sonora genuina, aunque limitada para predecir popularidad específicamente.

### 7.2 Aprendizaje supervisado

| Modelo | MAE Test | RMSE Test | R² Test | R² Train | Brecha Train–Test (R²) |
|---|---|---|---|---|---|
| Baseline (Dummy) | 16.15 | 19.44 | 0.00 | 0.00 | — |
| Ridge (tuned) | 10.68 | 15.08 | 0.398 | 0.401 | 0.003 (sin overfitting) |
| Random Forest (tuned) | 10.11 | 14.58 | 0.438 | 0.730 | 0.292 (overfitting notorio) |
| Gradient Boosting (tuned) | 10.27 | 14.73 | 0.426 | 0.493 | 0.067 (mejor generalización) |

**Modelo seleccionado:** Random Forest, por el menor RMSE en Test. Sin embargo, presenta la mayor brecha train-test de los tres modelos entrenados, indicando cierto sobreajuste pese al tuning de hiperparámetros (profundidad, mínimo de muestras por hoja). Gradient Boosting, aunque no ganó en RMSE absoluto, mostró el mejor equilibrio sesgo-varianza y es una alternativa a considerar si la prioridad fuera robustez ante datos futuros no vistos.

**Variables más relevantes (Random Forest):** `track_genre` (vía Target Encoding) domina la importancia (~0.50), seguido de `acousticness`, `loudness`, `instrumentalness`, `duration_min` y `danceability` — consistente con el hallazgo del EDA de que el género es la señal más fuerte del dataset.

### 7.3 Impacto en el KPI de negocio

Usando el modelo para priorizar el 20% superior del catálogo por popularidad predicha, se captura ~53% de los verdaderos éxitos (Top 20% real) — un lift de 2.7x respecto a una selección aleatoria — y la cartera resultante tiene una popularidad real promedio 16.6 puntos superior al promedio del catálogo completo. Esto valida el modelo como herramienta de apoyo a decisiones de priorización, no como predictor exacto.

## 8. Auditoría ética de sesgo

Se evaluó el error del modelo (MAE y sesgo promedio) desagregado por `macro_genero` (9 categorías, usadas exclusivamente para esta auditoría, nunca como feature del modelo). Los géneros con mayor error absoluto (`jazz-soul-funk`, `clasica-instrumental`) tienden a ser los de menor representación muestral en el dataset, mientras que géneros con mayor volumen de datos (`world-regional`, `otros`) presentan menor MAE. Esto sugiere que parte del error diferencial responde a desbalance de datos, no necesariamente a un sesgo editorial del modelo en sí.

**Recomendación ética:** no usar las predicciones del modelo en bruto para automatizar decisiones de promoción, ya que esto podría crear un bucle de retroalimentación que invisibilice aún más a los géneros de nicho o regionales ya subrepresentados. Se recomienda normalizar el puntaje predicho por percentiles **dentro de cada macro-género** antes de tomar decisiones editoriales, para no penalizar sistemáticamente a estilos musicales con menor presencia en el catálogo histórico.

## 9. Limitaciones y trabajo futuro

- El modelo explica entre el 40–44% de la varianza de `popularity`; el resto corresponde a factores exógenos no capturados por el dataset (marketing, fama previa, viralidad, inclusión en playlists editoriales).
- No se implementó búsqueda de hiperparámetros con frameworks de optimización bayesiana (ej. Optuna) por restricciones de tiempo; `RandomizedSearchCV` fue suficiente para el alcance de esta evaluación.
- Como trabajo futuro, se podría extender el modelo con **XGBoost/LightGBM** y validar si una búsqueda de hiperparámetros más exhaustiva reduce aún más el RMSE, así como incorporar variables externas (fecha de lanzamiento, métricas de redes sociales) si estuvieran disponibles.

## 10. Reproducibilidad y estructura del proyecto