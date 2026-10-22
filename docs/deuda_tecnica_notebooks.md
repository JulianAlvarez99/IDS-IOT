# Deuda técnica — notebooks del proyecto IDS-IoT

Auditoría de los 15 notebooks del repositorio (3.307 líneas de código en celdas,
sin contar markdown). Fecha: 27 de agosto de 2026.

El objetivo es doble: reducir el riesgo de perder trabajo o de manejar números
irreproducibles antes de la defensa, y dejar el código en un estado que un tribunal
pueda leer.

---

## 1 · Resumen

| Métrica | Valor |
|---|---|
| Notebooks auditados | 15 |
| Líneas de código en celdas | 3.307 |
| Líneas duplicadas literalmente entre notebooks | ~600 (18 %) |
| Copias de `FEATURES_SELECCIONADAS` | 11 |
| Copias de `focal_loss` | 7 (**2 implementaciones matemáticamente distintas**) |
| Copias del bloque split + `StandardScaler` | 9 |
| Rutas absolutas de una sola máquina | 7 (`Z:/TFG-Alvarez_Julian/...`) |
| Raíces de datos incompatibles conviviendo | 4 |
| Notebooks de entrenamiento con semilla fijada | **0 de 9** |
| Salidas de celda almacenadas dentro de los `.ipynb` | 7,4 MB |
| Celdas que fallan al ejecutarse (verificado) | 3 |

**Los tres hallazgos que atenderías primero si solo tuvieras una tarde:**

1. `E2_01_analisis_estadistico.ipynb` **borra `data/df_binary_balanced.feather`** al
   ejecutarse de principio a fin (D-01). Ese archivo no está en git, alimenta las
   etapas 3, 4 y 5, y regenerarlo exige reprocesar 4 GB de CSV.
2. Ninguna celda del proyecto reproduce el número **0,9933 frente a 0,5223** que
   justifica toda la Etapa 5 (D-03). Existe solo como comentario.
3. Ningún notebook de entrenamiento fija semilla (D-04). Volver a ejecutar `07b`
   produce un modelo estudiante distinto, y con él métricas de FPGA distintas.

---

## 2 · Lista priorizada

Prioridad = (Impacto + Riesgo) × (6 − Esfuerzo), con cada eje de 1 a 5.
Un esfuerzo bajo sube la prioridad: son las victorias baratas.

| # | Hallazgo | Categoría | Imp | Rie | Esf | **Pri** |
|---|---|---|:-:|:-:|:-:|:-:|
| D-01 | El notebook 01 borra el dataset balanceado | Código | 4 | 5 | 1 | **45** |
| D-03 | No hay validación numérica HLS frente a Keras | Test | 5 | 5 | 2 | **40** |
| D-04 | Ningún entrenamiento fija semilla | Test | 4 | 4 | 1 | **40** |
| D-02 | Cuatro raíces de rutas incompatibles | Código | 5 | 4 | 2 | **36** |
| D-09 | El `scaler` de despliegue no se persiste | Arquitectura | 4 | 5 | 2 | **36** |
| E-03 | La latencia de CPU se mide con `time.time()` | Test | 4 | 4 | 2 | **32** |
| D-08 | El holdout de `05c` no se limpia como en `05b` | Código | 3 | 3 | 1 | **30** |
| C-02 | El tipo de dato se deduce de una cadena de texto | Código | 3 | 3 | 1 | **30** |
| D-06 | Dos `focal_loss` divergentes con el mismo nombre | Código | 3 | 4 | 2 | **28** |
| D-07 | El umbral se ajusta sobre el conjunto de test | Test | 3 | 4 | 2 | **28** |
| E-02 | SHAP se calcula dos veces sobre 1.000.000 de filas | Eficiencia | 3 | 2 | 1 | **25** |
| C-03 | `07a` roto convive con `07a_corregido` | Documentación | 2 | 3 | 1 | **25** |
| C-04 | El docstring de `05d` describe otra arquitectura | Documentación | 2 | 3 | 1 | **25** |
| I-01 | 7,4 MB de salidas versionadas en los `.ipynb` | Infraestructura | 3 | 2 | 1 | **25** |
| I-03 | Colisión de nombres de artefactos | Infraestructura | 2 | 3 | 1 | **25** |
| I-02 | Los artefactos no tienen trazabilidad | Infraestructura | 4 | 4 | 3 | **24** |
| C-01 | ~600 líneas duplicadas entre notebooks | Código | 4 | 3 | 3 | **21** |
| D-05 | Tres celdas que fallan al ejecutarse | Código | 2 | 2 | 1 | **20** |
| E-01 | `append_to_feather` es O(n²) | Eficiencia | 3 | 2 | 2 | **20** |
| E-05 | Correlación de Pearson sobre 45 M de filas | Eficiencia | 2 | 2 | 1 | **20** |
| E-04 | `BATCH_SIZE = 32` con un comentario que dice lo contrario | Eficiencia | 2 | 1 | 1 | **15** |

---

## 3 · Detalle

### D-01 · El notebook 01 borra `data/df_binary_balanced.feather` · Pri 45

**Evidencia** (`notebooks/Etapa_2/E2_01_analisis_estadistico.ipynb`):

```
celda  6: FEATHER_TEMP_PATH: Path = Path("dataset_eda_temp.feather")
celda  8: FEATHER_TEMP_PATH: Path = Path("./data/df_binary_balanced.feather")
celda 32: FEATHER_TEMP_PATH.unlink()
```

La celda 6 construye el buffer temporal y documenta con cuidado que la celda de
limpieza final lo eliminará. La celda 8 —añadida después, como atajo para saltarse
la carga de 4 GB— **reasigna la misma variable** a otro archivo. La celda 32 sigue
confiando en el contrato de la celda 6 y borra lo que apunte la variable.

Resultado: «Reiniciar y ejecutar todo» destruye los 227 MB del dataset balanceado.
No está en git (`.gitignore` excluye `*.feather`) y es la entrada de las etapas 3, 4 y 5.

La celda 8 además descarta silenciosamente el `df` que produjo la celda 6, así que
todo el análisis posterior del notebook describe un dataset distinto del que el
título anuncia.

**Corrección.** Separar las dos variables y no borrar nunca a través de una variable
reasignable:

```python
# celda 6
FEATHER_TEMP_PATH = Path("dataset_eda_temp.feather")   # buffer, se borra

# celda 8 — ruta de análisis alternativa, NO es el buffer
RUTA_ANALISIS = DATOS / "df_binary_balanced.feather"
df = feather.read_feather(RUTA_ANALISIS)

# celda 32 — borrar solo el buffer, comprobando el nombre literal
if FEATHER_TEMP_PATH.name == "dataset_eda_temp.feather" and FEATHER_TEMP_PATH.exists():
    FEATHER_TEMP_PATH.unlink()
```

**Justificación.** Tres líneas de esfuerzo frente a la pérdida de un artefacto que
cuesta horas de reproceso y del que no existe copia versionada.

---

### D-03 · No hay validación numérica HLS frente a Keras · Pri 40

**Evidencia** (`notebooks/Etapa_5/E5_05a_HLS4ML.ipynb`): el notebook llama a
`hls_model.write()` y `hls_model.compile()`, pero `predict` no aparece en ninguna
celda. Las cifras que sostienen toda la etapa —

```
Keras float32 (referencia) ........................ 0,9913
ap_fixed<8,2> global + AP_WRAP (config anterior) ... 0,5223
esta configuración ................................ 0,9933
```

— viven únicamente dentro de un comentario de la celda 2.

**Riesgo.** Es el número central de la memoria y nada en el repositorio lo
regenera. Cualquier cambio en `PRECISION_*` o `FACTORES_REUSO` puede degradar la
exactitud sin que ninguna celda lo detecte; el fallo aparecería en la placa.

**Corrección.** Añadir una celda tras `hls_model.compile()`:

```python
# 4c. VALIDACIÓN NUMÉRICA — reproduce la tabla de la celda 2
X_val_hls = np.load(f'{DIR_MODELOS}/X_holdout_40k.npy')
y_val_hls = np.load(f'{DIR_MODELOS}/y_holdout_40k.npy')

p_keras = modelo_keras.predict(X_val_hls, batch_size=1024).ravel()
p_hls   = hls_model.predict(np.ascontiguousarray(X_val_hls)).ravel()

acc_keras = ((p_keras > 0.5) == y_val_hls).mean()
acc_hls   = ((p_hls   > 0.5) == y_val_hls).mean()
print(f'Keras float32 : {acc_keras:.4f}')
print(f'HLS punto fijo: {acc_hls:.4f}   (delta {acc_hls - acc_keras:+.4f})')
print(f'Discrepancias : {((p_keras > 0.5) != (p_hls > 0.5)).sum()} de {len(y_val_hls)}')

assert acc_hls >= acc_keras - 0.005, 'REGRESIÓN: revisa PRECISION_* antes de sintetizar'
```

El `assert` convierte el comentario en una prueba: si alguien vuelve a poner
`granularity='model'` o `AP_WRAP`, la celda falla en el servidor y no en la FPGA.
Requiere guardar una sola vez el subconjunto de 40.000 paquetes como `.npy` junto
al modelo.

---

### D-04 · Ningún entrenamiento fija semilla · Pri 40

**Evidencia.** `np.random.seed` aparece una sola vez en todo el proyecto, en el
notebook 01 (que no entrena). Los 9 notebooks que llaman a `.fit()` no fijan
`tf.random.set_seed` ni `PYTHONHASHSEED`. `random_state=42` en `train_test_split`
cubre la partición, pero no la inicialización de pesos, ni el dropout, ni el
barajado de lotes.

**Riesgo.** Volver a ejecutar `07b_compresion_Destillation` produce un modelo
estudiante distinto. Ese modelo es la entrada de la Etapa 5, así que las cifras de
la FPGA tampoco se reproducen. Si el tribunal pide reejecutar, los números de la
memoria no salen.

**Corrección.** Una celda al inicio de cada notebook que entrene:

```python
import os, random, numpy as np, tensorflow as tf
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
```

`TF_DETERMINISTIC_OPS` cuesta algo de rendimiento en GPU, pero para un MLP de
21 k parámetros es irrelevante frente a poder reproducir el resultado.

---

### D-02 · Cuatro raíces de rutas incompatibles · Pri 36

Los mismos tres archivos se referencian de cuatro formas distintas:

| Convención | Dónde | Ejemplo |
|---|---|---|
| Nombre suelto (relativo al CWD) | 02, 03, 04, 05, 05b, 05c, 05d, 06, 07a | `pd.read_feather('df_binary_balanced.feather')` |
| `./data/` | 01, celda 8 | `Path("./data/df_binary_balanced.feather")` |
| `Z:/TFG-Alvarez_Julian/IDS-IOT/` | 07a_corregido, 07b, 07c, 07d (7 apariciones) | `pd.read_feather('Z:/TFG-Alvarez_Julian/IDS-IOT/data/...')` |
| `/mnt/Datos/Julian_Alvarez-TFG/IDS-IOT/` | 05a | `DIR_MODELOS = f"/mnt/Datos/..."` |

Los notebooks viven en `notebooks/Etapa_N/`, así que un nombre suelto **no**
resuelve a `data/`: solo funciona si se copia el archivo al lado del notebook o se
lanza Jupyter desde la raíz. Además, `07c` usa una cadena sin prefijo `r`:

```python
'Z:\TFG-Alvarez_Julian\IDS-IOT\models\mlp_v2\mlp_binario_v2_exp2_focalloss.h5'
```

Aquí `\T`, `\I` y `\m` sobreviven por casualidad —Python los deja tal cual— pero
una ruta con `\t` o `\n` se rompería en silencio.

**Corrección.** Un único módulo `notebooks/comun/rutas.py` que localice la raíz
subiendo hasta encontrar un marcador, y que todos los notebooks importen:

```python
import os
from pathlib import Path

def raiz_proyecto() -> Path:
    p = Path.cwd().resolve()
    for cand in (p, *p.parents):
        if (cand / '.git').exists():
            return cand
    raise RuntimeError('No se encontró la raíz del proyecto')

RAIZ    = Path(os.environ.get('IDS_IOT_ROOT', raiz_proyecto()))
DATOS   = RAIZ / 'data'
MODELOS = RAIZ / 'models'
```

En el servidor basta con `export IDS_IOT_ROOT=/mnt/Datos/Julian_Alvarez-TFG/IDS-IOT`.
Un solo cambio cubre Windows, `Z:` y el servidor Debian.

---

### D-09 · El `scaler` de despliegue no se persiste · Pri 36

`07b_compresion_Destillation` construye el `StandardScaler` en la celda 3, entrena
el estudiante y guarda **solo el modelo**:

```python
student_model.save('modelo_estudiante_destilado_qat.h5')
```

`07d_eval_compressed_model` no carga ningún `.pkl`: repite las 25 líneas de split y
vuelve a ajustar el escalador desde cero, descartando `X_train`, `X_val` y `X_test`
inmediatamente. Solo necesita el objeto `scaler`, pero paga el coste completo y
—más importante— **cualquier cambio en el split cambia en silencio la normalización**.

**Riesgo directo sobre la placa.** El IP de la FPGA recibe valores ya normalizados.
Si el `scaler` que use el host PYNQ no es exactamente el que vio el estudiante
durante la destilación, la exactitud cae y el síntoma es indistinguible de un
problema de punto fijo — justo el tipo de fallo que costó depurar en la Etapa 5.

**Corrección.** Guardar el escalador junto al modelo en `07b`, y en `07d` y en el
driver PYNQ cargarlo, nunca reajustarlo:

```python
# 07b, junto al save del modelo
joblib.dump(scaler, DIR_MODELOS / 'scaler_estudiante_destilado.pkl')

# 07d y pynq/ids_driver.py
scaler = joblib.load(DIR_MODELOS / 'scaler_estudiante_destilado.pkl')
```

---

### E-03 · La latencia de CPU se mide con `time.time()` · Pri 32

`E3_06_simulador_tiempo_real.ipynb` mide la latencia de inferencia que la memoria
compara contra la FPGA:

```python
t0_inf = time.time()
y_prob = model(X_scaled, training=False).numpy()
t_fin  = time.time()
latencia_ms = (t_fin - t0_inf) * 1000
```

Tres problemas, en orden de gravedad:

1. **`time.time()` no es un reloj monótono de alta resolución.** Para latencias
   por debajo del milisegundo hay que usar `time.perf_counter()`. Con `time.time()`
   en Windows el ruido de cuantización del reloj es del mismo orden que la magnitud
   medida.
2. **La medida excluye el preprocesado.** `scaler.transform` queda fuera de la
   ventana, pero en la FPGA el equivalente sí forma parte del camino de datos.
   La comparación CPU-frente-a-FPGA no mide lo mismo a ambos lados.
3. **El bucle usa `iterrows()` y `clear_output()` por paquete.** El coste de
   `iterrows` (que materializa una `Series` por fila) y del refresco de consola
   domina el tiempo de pared, así que el *throughput* reportado
   (`1000 / latencia_avg`) no describe ningún sistema real.

**Corrección.**

```python
X_all = scaler.transform(df_real[FEATURES_SELECCIONADAS].to_numpy(dtype=np.float32))

# calentar el grafo antes de medir (la primera llamada compila)
_ = model(X_all[:1], training=False)

latencias = np.empty(len(X_all))
for i in range(len(X_all)):
    t0 = time.perf_counter()
    p  = model(X_all[i:i+1], training=False).numpy()
    latencias[i] = (time.perf_counter() - t0) * 1000

print(f'p50 {np.percentile(latencias, 50):.4f} ms | '
      f'p99 {np.percentile(latencias, 99):.4f} ms')
```

Conviene reportar mediana y p99 en lugar de media, máximo y mínimo: con colas
largas la media no es representativa y el máximo es un único evento del
planificador. Y refrescar el tablero cada N paquetes, no cada uno.

---

### D-08 · El holdout de `05c` no se limpia como en `05b` · Pri 30

`05b` y `05d` preparan el holdout así:

```python
df_holdout = pd.read_feather('test_real_traffic.feather')
df_holdout = df_holdout.replace([np.inf, -np.inf], np.nan) \
                       .dropna(subset=FEATURES_SELECCIONADAS + ['label'])
```

`05c_analisis_errores_holdout` lo lee sin limpiar nada. Evalúa por tanto un
conjunto distinto —con infinitos incluidos— del que produjo las métricas de `05b`.
El recuento de falsos negativos por clase de ataque que aparece en la memoria no
es comparable con la matriz de confusión de la que dice derivar.

**Corrección.** Extraer la carga del holdout a una función común (véase C-01) para
que sea imposible que dos notebooks lo preparen de forma distinta.

---

### C-02 · El tipo de dato se deduce de una cadena de texto · Pri 30

En `07a_corregido`, `07b` y `07c`:

```python
es_cuantizado = "QAT" in nombre or "Estudiante" in nombre
bytes_por_parametro = 1 if es_cuantizado else 4
huella_kb = (active_params * bytes_por_parametro) / 1024
```

`nombre` es la etiqueta que se imprime por pantalla. La huella en BRAM que aparece
en la memoria depende de que alguien escriba «QAT» en un título. Renombrar la
entrada del diccionario a algo como `"Modelo comprimido"` multiplica el resultado
por cuatro sin ningún aviso.

Además cuenta las escalas y desplazamientos de `BatchNormalization` como si fueran
enteros de 8 bits, cuando en el firmware generado son `ap_fixed<16,6>`.

**Corrección.** Derivar los bits del cuantizador real de cada capa:

```python
def bits_de_capa(capa, por_defecto=32):
    q = getattr(capa, 'kernel_quantizer_internal', None)
    return int(getattr(q, 'bits', por_defecto))
```

Es exactamente la misma introspección que ya usa `05a_HLS4ML` en
`_precision_desde_cuantizador`; conviene compartir esa función.

---

### D-06 · Dos `focal_loss` divergentes con el mismo nombre · Pri 28

Siete copias de `focal_loss`. Seis son idénticas. La de `05c` es **otra función**:

| | 05b / 05d / 07a / 07a_c / 07b / 07c | 05c |
|---|---|---|
| `alpha` por defecto | `0.75` | `0.25` |
| Reducción | `tf.reduce_mean(...)` global | `K.mean(..., axis=-1)` |
| Estructura | un término con `p_t` y `alpha_t` | dos términos separados |

`05c` la define pero después carga el modelo con `compile=False`, así que hoy no se
usa. El riesgo es futuro: si alguien recompila desde ese notebook, entrena con una
pérdida distinta bajo el mismo nombre.

**Corrección.** Una sola definición en `notebooks/comun/perdidas.py`; borrar la
variante de `05c`.

---

### D-07 · El umbral se ajusta sobre el conjunto de test · Pri 28

En `05b` y `05d`:

```python
probs_exp1 = evaluate_model(model_exp1, X_test, y_test, 'EXP1 — Test Interno')
thr1_f1, thr1_r99 = find_optimal_threshold(y_test, probs_exp1, 'Exp1')
```

El umbral que maximiza F1 y el de recall ≥ 99 % se eligen mirando `y_test`, y a
continuación se reporta el rendimiento sobre ese mismo `y_test`. El resultado del
test interno queda optimistamente sesgado.

El holdout (`test_real_traffic.feather`) sí es independiente, así que las cifras
finales de la memoria se salvan. Pero es el tipo de detalle que un tribunal
pregunta, y la corrección consiste en mover una variable.

**Corrección.** Ajustar sobre validación y reportar sobre test:

```python
probs_val = model_exp1.predict(X_val, batch_size=BATCH_SIZE)
thr1_f1, thr1_r99 = find_optimal_threshold(y_val, probs_val, 'Exp1 (validación)')
eval_with_threshold(y_test, probs_exp1, thr1_r99, 'Exp1 — Test con umbral de val.')
```

Hay además un detalle menor en la misma función: `precision_recall_curve` devuelve
`thr` con un elemento menos que `prec` y `rec`. El código lo parchea con
`thr[min(idx, len(thr)-1)]`, lo que en el punto final asocia un umbral que no
corresponde a la precisión y el recall que se imprimen al lado.

---

### E-02 · SHAP se calcula dos veces sobre 1.000.000 de filas · Pri 25

`E2_02_feature_importance_binario.ipynb`:

```python
# celda 18
X_shap = X_train.sample(n=1000000, random_state=42)
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_shap)      # primera pasada

# celda 22
explainer = shap.Explainer(clf, X_shap)          # X_shap como fondo: 1e6 filas
shap_values = explainer(X_shap)                  # segunda pasada, mismo resultado
```

La celda 22 recalcula lo mismo que la 18 y además pasa el millón de filas como
*background dataset*. La única diferencia real es el estilo del gráfico.

Para un ranking de importancia de 35 características, 1.000.000 de muestras no
aporta nada frente a 50.000: el error estándar de la media ya es despreciable.

**Corrección.** Una sola pasada, `X_shap` de 50.000 filas, fondo de ~2.000, y usar
`explainer(X_shap)` (objeto `Explanation`) para ambos gráficos.

Detalle relacionado: las llamadas `plt.figure(figsize=...)` que preceden a
`shap.summary_plot` no tienen efecto, porque SHAP crea su propia figura. Para
controlar el tamaño hay que pasar `show=False` y ajustar después con
`plt.gcf().set_size_inches(...)`.

---

### C-03 · `07a` roto convive con `07a_corregido` · Pri 25

`E4_07a_compresion_Pruning-QAT.ipynb` sigue en el repositorio con el defecto que
`07a_corregido` documenta explícitamente como FIX 2:

```python
qat_model.compile(optimizer='adam', loss=focal_loss, metrics=[...])
#                                        ^^^^^^^^^^ sin paréntesis
```

Además, ambos escriben `modelo_pruned_qat.h5` en el mismo sitio: el que sobreviva
depende de cuál se ejecutó al final.

**Corrección.** Borrar `E4_07a_compresion_Pruning-QAT.ipynb` (git conserva el
histórico) y renombrar `07a_corregido` a `E4_07a_compresion_Pruning-QAT.ipynb`. Si
interesa conservar el error como material de la memoria, moverlo a
`notebooks/descartados/` con una celda markdown al principio que explique por qué
está ahí.

---

### C-04 · El docstring de `05d` describe otra arquitectura · Pri 25

```python
def build_improved_model(input_dim, l2_reg=1e-3):
    """
    MLP v2: 64→128→64→32 con BatchNorm+L2 en todas las capas.
    """
    m = Sequential(name='mlp_binario_v3')
    m.add(Dense(64, ..., activation='relu'))   # sin BatchNormalization
```

El modelo de `05d` **no tiene ni una sola capa de BatchNormalization**; el
docstring se copió de `05b`. No es cosmético: la presencia de BatchNorm es justo lo
que determina los rangos de acumulador que la Etapa 5 tuvo que calibrar
(`PRECISION_ACCUM = ap_fixed<18,10>` porque las salidas de BN llegan a +34,7). Un
lector que confíe en el docstring saca la conclusión contraria.

`05d` tiene además 9 de 13 celdas sin ejecutar y una función rota (véase D-05):
está a medias. Conviene decidir si se termina o se descarta.

---

### I-01 · 7,4 MB de salidas versionadas en los `.ipynb` · Pri 25

| Notebook | Salidas | Imágenes |
|---|---:|---:|
| `E2_01_analisis_estadistico.ipynb` | 3.033 KB | 39 |
| `E2_02_feature_importance_binario.ipynb` | 2.548 KB | 5 |
| `E3_05b_entrenamiento_binario_mejorado.ipynb` | 745 KB | 12 |
| resto | ~1.100 KB | 20 |

Son los dos archivos más grandes del repositorio, por delante de cualquier código.
Cada reejecución produce un diff de megabytes de base64 ilegible, y los conflictos
de Syncthing sobre estos archivos son prácticamente irresolubles a mano.

**Corrección.**

```bash
pip install nbstripout && nbstripout --install --attributes .gitattributes
```

Las figuras que hagan falta para la memoria se guardan con `fig.savefig()` en
`models/<run>/`, que es donde ya están las demás.

---

### I-03 · Colisión de nombres de artefactos · Pri 25

`arquitectura_ganadora_binario.png` lo escriben `05`, `05b` y `05d` — tres
arquitecturas distintas, un solo archivo. Peor: `05b` y `05d` lo escriben desde el
bloque `try`, pero en el `except ImportError` **leen ese mismo archivo** para
mostrarlo, así que si `visualkeras` no está instalado el notebook muestra la figura
de otro modelo como si fuera el suyo.

Lo mismo con `modelo_pruned_qat.h5` (07a y 07a_corregido) y con
`best_model_exp1.h5` / `best_model_v3.h5`, que se escriben en el CWD y luego se
mueven a mano a `models/<run>/`.

**Corrección.** Derivar el nombre del identificador de la ejecución:

```python
RUN_ID  = f'{ARQUITECTURA}_{EPOCHS}ep_{BATCH_SIZE}b'
DIR_RUN = MODELOS / RUN_ID
DIR_RUN.mkdir(parents=True, exist_ok=True)
```

Es la convención que `05a_HLS4ML` ya aplica (`Destillation_15epochs_128batch`);
solo falta llevarla a las etapas 3 y 4.

---

### I-02 · Los artefactos no tienen trazabilidad · Pri 24

`.gitignore` excluye `*.h5`, `*.pkl` y `*.feather`. Es la decisión correcta para el
tamaño, pero deja el repositorio sin **ningún** registro de qué modelo produjo qué
número. `models/` contiene siete carpetas de ejecución y nada indica con qué
commit, con qué datos ni con qué hiperparámetros se generó cada una.

**Corrección barata, sin DVC.** Un `manifest.json` por ejecución:

```python
import hashlib, json, subprocess

def escribir_manifiesto(dir_run, modelo_path, metricas, hiperparams):
    sha = hashlib.sha256(Path(modelo_path).read_bytes()).hexdigest()[:16]
    commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                     text=True).strip()
    (Path(dir_run) / 'manifest.json').write_text(json.dumps({
        'modelo': Path(modelo_path).name, 'sha256': sha,
        'commit': commit, 'fecha': pd.Timestamp.now().isoformat(),
        'metricas': metricas, 'hiperparametros': hiperparams,
        'semilla': SEED,
    }, indent=2, ensure_ascii=False), encoding='utf-8')
```

El `manifest.json` sí se versiona (son 2 KB) y cierra el hueco entre «este número
está en la memoria» y «este archivo lo produjo».

---

### C-01 · ~600 líneas duplicadas entre notebooks · Pri 21

| Bloque | Copias | Notebooks |
|---|:-:|---|
| `FEATURES_SELECCIONADAS` | 11 | 04, 05, 05b, 05c, 05d, 06, 07a, 07a_c, 07b, 07c, 07d |
| Split + `StandardScaler` (25 líneas) | 9 | 04, 05, 05b, 05d, 07a, 07a_c, 07b, 07c, 07d |
| `focal_loss` | 7 | 05b, 05c, 05d, 07a, 07a_c, 07b, 07c |
| `graficar_curvas_entrenamiento` | 4 | 07a_c, 07b, 07c (+ variante rota en 05d) |
| `analizar_parametros_hardware` + `evaluar_y_comparar_modelos` | 3 | 07a_c, 07b, 07c |
| `eval_with_threshold` | 3 | 05b, 05d, 07d |
| `find_optimal_threshold` | 2 | 05b, 05d |

El coste no es el espacio: es que **una corrección hay que aplicarla once veces**, y
D-06 demuestra que ya ha fallado al menos una vez.

**Corrección.** Un paquete pequeño, sin sobreingeniería:

```
notebooks/comun/
├── __init__.py
├── config.py      # FEATURES_SELECCIONADAS, NOMBRE_CLASE_BENIGNA, SEED
├── rutas.py       # RAIZ, DATOS, MODELOS                  (resuelve D-02)
├── datos.py       # cargar_balanceado(), cargar_holdout(), split_estandar()
├── perdidas.py    # focal_loss                            (resuelve D-06)
└── evaluar.py     # eval_con_umbral(), buscar_umbral(), perfil_hardware()
```

Y en cada notebook:

```python
import sys; sys.path.insert(0, str(Path.cwd().parents[1]))
from notebooks.comun.config import FEATURES_SELECCIONADAS, SEED
from notebooks.comun.datos  import cargar_balanceado, split_estandar
```

Un módulo `.py` importado se puede además probar con `pytest`, cosa que una celda no.

Nota relacionada sobre `graficar_curvas_entrenamiento`: adivina el nombre de la
métrica con `history.history.get('auc') or history.history.get('auc_1')`. Eso hace
falta solo porque en algunos notebooks la métrica se declara como
`tf.keras.metrics.AUC()` sin `name=`. Poniendo `AUC(name='auc')` en todas partes
—como ya hace `07c`— la adivinanza sobra. Tal y como está, si `acc` sale `None` la
línea `range(1, len(acc) + 1)` lanza `TypeError`.

---

### D-05 · Tres celdas que fallan al ejecutarse · Pri 20

Verificado ejecutando el fragmento en el entorno `ids-iot-dev`:

**1 · `05d`, celda 7** — `plot_history`

```python
fig, axes = plt.subplots(1, 1, figsize=(16, 5))
for ax, hist, label in zip(axes, [hist1], ['Focal Loss']):
```

```
TypeError: 'Axes' object is not iterable
```

`plt.subplots(1, 1)` devuelve un `Axes` suelto, no un array. Se arregla con
`squeeze=False` o quitando el `zip`.

**2 · `02`, celda 16**

```python
clf = xgb.XGBModel.load_model('xgb_modelo_binario.json')
```

```
TypeError: XGBModel.load_model() missing 1 required positional argument: 'fname'
```

Se está llamando al método sobre la clase, no sobre una instancia. Correcto:

```python
clf = xgb.XGBClassifier()
clf.load_model('xgb_modelo_binario.json')
```

**3 · `07a` (versión no corregida), celda 7** — `loss=focal_loss` sin paréntesis.
Ya documentado en `07a_corregido` como FIX 2; desaparece al aplicar C-03.

Menores, del mismo tipo: en `07a_corregido`, `print('/nSparsity por capa...')` usa
`/n` en lugar de `\n`; en `05c`, `FEATURES_SELECCIONADAS = FEATURES_SELECCIONADAS = [...]`
es una asignación doble; en `03`, `import collections` no se usa y el par
`X = X_bal; y = y_final` mantiene vivas dos referencias al mismo array grande
justo donde el notebook intenta liberar RAM.

---

### E-01 · `append_to_feather` es O(n²) · Pri 20

`01_analisis_estadistico`, celda 6:

```python
def append_to_feather(batch_df, feather_path):
    if feather_path.exists():
        existing = feather.read_feather(feather_path)          # lee TODO
        combined = pd.concat([existing, batch_df], ignore_index=True)
    ...
    feather.write_feather(combined, feather_path)              # reescribe TODO
```

Feather no admite *append*, así que cada lote reescribe el acumulado completo. Con
169 CSV en lotes de 10 son 17 iteraciones: el volumen total escrito ronda
`17/2 × 4 GB ≈ 34 GB` para producir un archivo de 4 GB, más otro tanto de lectura.
El pico de RAM tampoco es el de un lote, sino el del acumulado más el lote — que es
justo lo que la estrategia documentada en el propio comentario intentaba evitar.

**Corrección.** Escribir en streaming con un único `ParquetWriter`:

```python
import pyarrow as pa, pyarrow.parquet as pq

escritor = None
for batch_idx in tqdm(range(n_batches), desc='Batches'):
    batch_df, _ = load_batch(...)
    if batch_df.empty:
        continue
    tabla = pa.Table.from_pandas(batch_df, preserve_index=False)
    if escritor is None:
        escritor = pq.ParquetWriter(RUTA_TEMP, tabla.schema, compression='zstd')
    escritor.write_table(tabla)
    del batch_df, tabla; gc.collect()
if escritor:
    escritor.close()
```

Una sola pasada de escritura, RAM acotada al lote, y con `zstd` el archivo baja
además de forma sustancial. Si se prefiere seguir en Feather, la alternativa es
`pyarrow.ipc.new_file`, que también admite escritura incremental.

---

### E-05 · Correlación de Pearson sobre 45 M de filas · Pri 20

`01`, celda 24: el comentario dice «Se usa una muestra representativa (50k filas)
para agilizar el cómputo» y la línea que lo haría está comentada:

```python
#.sample(n=min(CORR_SAMPLE_SIZE, len(df)), random_state=RANDOM_SEED)
```

Se calcula por tanto una matriz 47×47 sobre las 45 M de filas. Para estimar
coeficientes de correlación, 50.000 muestras dan tres decimales estables; el umbral
de decisión del notebook es 0,90, muy por encima de ese error.

Aparte: `df_corr_input` aplica `.dropna()` sobre **todas** las columnas, así que una
sola característica con muchos infinitos elimina filas del cálculo de todas las
demás. Conviene calcular con `.corr(min_periods=...)` sobre el frame sin `dropna`
global.

---

### E-04 · `BATCH_SIZE = 32` con un comentario que dice lo contrario · Pri 15

`05_entrenamiento_binario`, celda 6:

```python
BATCH_SIZE = 32 # Tamaño grande para aprovechar GPU RTX y paralelización
```

32 no es grande; es pequeño. Con ~2 M de muestras de entrenamiento y hasta 150
épocas son unos 9,7 M de pasos de optimizador, con la GPU infrautilizada casi todo
el tiempo. `05b` ya usa 128, que es el valor sensato. Es un residuo de una etapa
anterior, pero el comentario induce a error a quien lea el notebook.

Menor, en el mismo notebook: `EarlyStopping(restore_best_weights=True)` monitoriza
`val_loss`, mientras que `ModelCheckpoint` guarda el mejor por `val_auc`. El `.h5`
del checkpoint y el modelo que guarda la celda 8 son por tanto **modelos
distintos**, ambos presentados como «el modelo final». Conviene monitorizar la
misma métrica en los dos callbacks.

---

## 4 · Plan de remediación

Ordenado para no bloquear el trabajo de la placa. Ninguna fase exige detener la
Etapa 5.

### Fase 0 — Parar la sangría (una tarde)

Es lo que hay que hacer antes de volver a ejecutar cualquier notebook.

| Tarea | Hallazgo |
|---|---|
| Separar `FEATHER_TEMP_PATH` de `RUTA_ANALISIS` en el notebook 01 | D-01 |
| Añadir la celda de semillas a los 9 notebooks que entrenan | D-04 |
| Instalar `nbstripout` y limpiar las salidas versionadas | I-01 |
| Borrar `07a` sin corregir y renombrar `07a_corregido` | C-03 |
| Arreglar las tres celdas rotas y los typos | D-05 |

Riesgo cero para la Etapa 5: ninguna toca el firmware ni el modelo.

### Fase 1 — Blindar los números de la memoria (dos tardes)

Esta fase es la que protege la defensa.

| Tarea | Hallazgo |
|---|---|
| Celda de validación HLS frente a Keras, con `assert`, en `05a` | D-03 |
| Persistir el `scaler` en `07b`; cargarlo en `07d` y en el driver PYNQ | D-09 |
| Rehacer la medición de latencia de `06` con `perf_counter` y p50/p99 | E-03 |
| Mover el ajuste de umbral de test a validación en `05b` | D-07 |
| Limpiar el holdout en `05c` igual que en `05b` | D-08 |

### Fase 2 — Un solo origen de verdad (tres tardes, en paralelo a la placa)

| Tarea | Hallazgo |
|---|---|
| Crear `notebooks/comun/{config,rutas,datos,perdidas,evaluar}.py` | C-01, D-02, D-06 |
| Migrar los notebooks de la Etapa 4 primero (son los que tienen `Z:/`) | D-02 |
| Después la Etapa 3, y por último la 2 | C-01 |
| Sustituir la deducción por cadena de texto por introspección del cuantizador | C-02 |
| `manifest.json` por ejecución en `models/<run>/` | I-02, I-03 |

Migrar notebook a notebook: cada uno queda funcionando al terminarlo, sin big bang.

### Fase 3 — Eficiencia (cuando haya hueco)

| Tarea | Hallazgo |
|---|---|
| `ParquetWriter` en streaming en el notebook 01 | E-01 |
| Una sola pasada de SHAP con 50 k muestras | E-02 |
| Reactivar el muestreo de la matriz de correlación | E-05 |
| Corregir `BATCH_SIZE` y su comentario; unificar la métrica de los callbacks | E-04 |
| Decidir si `05d` se termina o se descarta | C-04 |

---

## 5 · Lo que está bien y conviene no tocar

Para que el informe no dé una impresión falsa: hay partes del proyecto
sensiblemente mejores que la media.

- **`E5_05a_HLS4ML.ipynb`** es el mejor notebook con diferencia. Parametrización
  global, cada constante numérica justificada con la medida que la motiva, y la
  lógica de reparación de `QActivation` escrita para tolerar diferencias entre
  versiones de hls4ml. Su único hueco real es D-03.
- **La estrategia de memoria del notebook 01** (lotes, `gc.collect()`, `ax.bxp()`
  sobre percentiles precalculados en lugar de `sns.boxplot`) está bien razonada, y
  el comentario que explica por qué se evita Seaborn con 45 M de filas es
  exactamente el tipo de nota que hay que dejar escrita. E-01 es un fallo de
  implementación dentro de una estrategia correcta.
- **`07a_corregido`** documenta sus tres correcciones explicando el porqué de cada
  una. Ese estilo es el que conviene extender al resto.
- **La separación del holdout del 10 % antes de cualquier balanceo** (notebook 05)
  es metodológicamente correcta, y es lo que salva a D-07 de ser un problema serio.
