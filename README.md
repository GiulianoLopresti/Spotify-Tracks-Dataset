# Spotify-Tracks-Dataset

# EP1 - Machine Learning 001D
## Caso C: Inteligencia musical y predicción de popularidad de canciones (Spotify Tracks)
### Integrantes: Marco Bona - Giuliano Lopresti - Lorena Ormeño

## Descripción del problema de negocio

Predecir la popularidad de una canción (valor entre 0 y 100) a partir de sus atributos de audio, con el fin de apoyar decisiones de marketing musical, curaduría de playlists y estrategia de lanzamiento de artistas en plataformas de streaming.

## Objetivos del proyecto

- Identificar qué atributos de audio se relacionan con la popularidad de una canción.
- Preparar un dataset limpio y transformado, apto para un futuro proceso de modelamiento.
- Evaluar los sesgos y aspectos éticos asociados al uso de estos datos.

## KPIs que resolverán el problema de negocio

- Correlación entre variables de audio y `popularity`.
- Porcentaje de valores nulos y duplicados eliminados en la etapa de limpieza.
- Error de predicción del modelo (a definir en la etapa de modelamiento, fuera del alcance de este EP1).

## Descripción de las fuentes de datos

**Spotify Tracks Dataset**, con 114.000 registros y 21 columnas, incluyendo metadatos de la canción (`track_id`, `artists`, `album_name`, `track_name`, `track_genre`) y atributos de audio (`danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `duration_ms`, `key`, `mode`, `time_signature`, `explicit`), además de la variable objetivo `popularity`.

## Preparación y análisis exploratorio de los datos (EDA)

### Calidad de datos

- **Valores nulos:** mínimos, solo 1 registro con nulos en `artists`, `album_name` y `track_name`.
- **Duplicados:** no hay filas totalmente duplicadas, pero sí **24.259 registros con `track_id` repetido** (mismo tema publicado en distintos álbumes/singles), los cuales se consolidaron conservando la versión de mayor popularidad.
- **Shape original:** (114.000, 21) → **Shape tras limpieza:** (89.740, 21).

### Variables categóricas

- **114 géneros musicales únicos** (`track_genre`), distribuidos de forma pareja (1.000 canciones por género aprox.), sin un género dominante extremo.
- El artista más frecuente en el dataset es **The Beatles** (279 canciones).

### Distribución de la variable objetivo (`popularity`)

- Fuertemente sesgada hacia 0: cerca de 20.000 canciones tienen `popularity = 0`.
- El resto de la distribución es multimodal, con concentraciones entre 20 y 60 puntos.
- Esto sugiere que una parte relevante del catálogo corresponde a canciones poco o nada reproducidas recientemente.

### Correlaciones

- **Ninguna variable de audio muestra correlación fuerte con `popularity`** (todas entre -0.04 y 0.04), lo que indica que la popularidad depende más de factores externos (marketing, artista, momento de lanzamiento) que de las características sonoras del tema.
- Sí existen correlaciones fuertes entre variables de audio entre sí: `energy`-`loudness` (0.76) y `energy`-`acousticness` (-0.73).

### Outliers

- Se identificaron outliers en `duration_ms` y `tempo`, tratados mediante *winsorización* (recorte a percentiles 1% y 99%).

## Preparación de datos para modelamiento

- Eliminación de nulos y de duplicados por `track_id`.
- Tratamiento de outliers en `duration_ms` y `tempo`.
- Codificación: `explicit` a entero, `track_genre` mediante `LabelEncoder`.
- Escalado estándar (`StandardScaler`) de las variables de audio y `duration_ms`.
- División train/test: **71.792 registros de entrenamiento** y **17.948 de prueba** (80/20).
- Dataset procesado exportado a `data/dataset_procesado.csv`.

## Evaluación de sesgos, ética y privacidad

**Sesgos:**
- *Exposición/plataforma:* `popularity` refleja reproducciones recientes, favoreciendo lanzamientos nuevos sobre canciones antiguas de igual calidad.
- *Artista/discográfica:* artistas con mayor respaldo de marketing o presencia en playlists editoriales acumulan más reproducciones, generando un sesgo de distribución más que de calidad musical.
- *Geográfico/cultural:* posible sobre-representación de mercados anglosajones.

**Aspectos éticos:**
- Usar `popularity` como objetivo puede reforzar un círculo vicioso que perjudica a artistas emergentes o de nicho si se usa para decidir qué promocionar.
- Debe declararse explícitamente que el modelo predice popularidad *según el algoritmo de Spotify*, no valor artístico ni éxito comercial real.

**Privacidad:**
- Dataset público y agregado, sin información personal de usuarios (no contiene IDs de usuario, ubicación ni datos personales).
- Los datos de artistas y canciones ya son información pública publicada en la plataforma.

## Metodología utilizada (CRISP-DM)

1. **Comprensión del negocio:** definición del problema, objetivos y KPIs.
2. **Comprensión de los datos:** exploración inicial, `info()`, `describe()`, identificación de nulos y duplicados.
3. **Preparación de los datos:** limpieza, tratamiento de outliers, codificación y escalado.
4. **Análisis exploratorio (EDA):** distribuciones, correlaciones y detección de patrones.
5. **Evaluación:** análisis de sesgos, ética y privacidad de los datos.
6. *(Etapas de modelamiento y despliegue quedan fuera del alcance de este EP1.)*

## Estructura del proyecto

```
proyecto-spotify/
├── data/
│   ├── dataset.csv
│   └── dataset_procesado.csv
├── notebooks/
│   └── 01_eda_preparacion.ipynb
├── models/
├── images/
└── README.md
```