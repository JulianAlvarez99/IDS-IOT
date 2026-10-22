# Figuras e imágenes

Todas las imágenes del proyecto, agrupadas por la etapa que las produce. La columna
**Origen** indica el notebook que genera (o del que se exportó) cada figura.

> Los notebooks guardan sus figuras con nombre suelto en el directorio de trabajo
> (`savefig("x.png")`). Al regenerarlas, muévelas aquí a la carpeta de su etapa y
> añade la fila correspondiente a esta tabla.

## Etapa 2 — Análisis estadístico y selección de características

| Archivo | Origen | Contenido |
|---|---|---|
| `Etapa_2/Distribucion de clases CIC-IOT2023.png` | `notebooks/Etapa_2/E2_01_analisis_estadistico.ipynb` | Distribución de clases del dataset CIC-IoT-2023 |
| `Etapa_2/Analisis de valores de features.png` | `notebooks/Etapa_2/E2_01_analisis_estadistico.ipynb` | Mapa de calor de valores problemáticos por característica: % de NaN, Inf y outliers |
| `Etapa_2/Percentiles/*.png` (36) | `notebooks/Etapa_2/E2_01_analisis_estadistico.ipynb` | Distribución por percentiles de cada característica; un archivo por feature |
| `Etapa_2/estructura_arbol_0.png` | `notebooks/Etapa_2/E2_02_feature_importance_binario.ipynb` | Primer árbol del XGBoost binario |
| `Etapa_2/SHAP_BarPlot_Binary.png` | `notebooks/Etapa_2/E2_02_feature_importance_binario.ipynb` | Importancia SHAP media por característica (binario) |
| `Etapa_2/SHAP_Beeswarm_binary.png` | `notebooks/Etapa_2/E2_02_feature_importance_binario.ipynb` | Beeswarm SHAP (binario) |

`Percentiles/` está en `.gitignore`: existe solo en local.

## Etapa 3 — Entrenamiento y evaluación del MLP

| Archivo | Origen | Contenido |
|---|---|---|
| `Etapa_3/arquitectura_ganadora_binario.png` | `notebooks/Etapa_3/E3_05_entrenamiento_binario.ipynb` (`plot_model`; también la regeneran `E3_05b` y `E3_05d`) | Topología del MLP elegido por el NAS (`E3_04`) |
| `Etapa_3/51epochs_128batch/*` | `notebooks/Etapa_3/E3_05_entrenamiento_binario.ipynb` | Curvas de métricas y matriz de confusión del modelo `models/51epochs_128batch/` |
| `Etapa_3/71epochs_64batch/*` | `notebooks/Etapa_3/E3_05_entrenamiento_binario.ipynb` | Ídem para `models/71epochs_64batch/` (incluye matriz sobre test) |
| `Etapa_3/92epochs_32batch/*` | `notebooks/Etapa_3/E3_05_entrenamiento_binario.ipynb` | Ídem para `models/92epochs_32batch/` (incluye matriz sobre test) |
| `Etapa_3/mlp_v2/mlp_binario_v2_{metrics,AUC,recall}.png` | `notebooks/Etapa_3/E3_05b_entrenamiento_binario_mejorado.ipynb` | Curvas de entrenamiento de `models/mlp_v2/` |
| `Etapa_3/mlp_v2/class_weight_*_matrix_confusion.png` | `notebooks/Etapa_3/E3_05b_entrenamiento_binario_mejorado.ipynb` | Matrices de confusión del experimento 1 (class weight), umbral normal y óptimo, train y holdout |
| `Etapa_3/mlp_v2/focal_loss_*_matrix_confusion.png` | `notebooks/Etapa_3/E3_05b_entrenamiento_binario_mejorado.ipynb` | Ídem para el experimento 2 (focal loss) |
| `Etapa_3/mlp_v2/False_negative_classes_focal_loss_model.png` | `notebooks/Etapa_3/E3_05c_analisis_errores_holdout.ipynb` | Clases de ataque con más falsos negativos en el holdout |

## Etapa 4 — Compresión

| Archivo | Origen | Contenido |
|---|---|---|
| `Etapa_4/Destillation_15epochs_128batch/Metricas_modelo_Destilado.png` | `notebooks/Etapa_4/E4_07b_compresion_Destillation.ipynb` | Métricas del estudiante destilado (`models/Destillation_15epochs_128batch/`) |
| `Etapa_4/PQAT_15epochs_128batch/Metricas_modelo_PQAT.png` | `notebooks/Etapa_4/E4_07c_compresion_PQAT.ipynb` | Métricas del modelo PQAT (`models/PQAT_15epochs_128batch/`) |

El origen de las figuras de Etapa 3 y 4 se deduce del nombre de la carpeta del
modelo y del notebook que lo entrena; los notebooks no las guardan con ese nombre
exacto, así que estas imágenes se exportaron a mano.
