# Guía de Configuración del Entorno de Desarrollo
## Proyecto: Módulo de Detección para IDS Pasivo

---

## Decisión de versiones

| Componente | Versión | Motivo |
|---|---|---|
| **Python** | 3.11 | Punto óptimo de estabilidad para TF + QKeras + keras-tcn + hls4ml |
| **TensorFlow** | 2.14.0 | Última versión con **Keras v2 integrado**. TF ≥2.15 separa Keras como paquete independiente (v3) e incompatibiliza con el conversor `keras_to_hls` de hls4ml (Hito 5) |
| **numpy** | <2.0.0 | Compatibilidad binaria con TF 2.14 |

---

## Prerequisitos

- [Anaconda](https://www.anaconda.com/download) o [Miniconda](https://docs.conda.io/en/latest/miniconda.html) instalado
- Git instalado
- Cuenta de Kaggle con `kaggle.json` configurado (para descarga del dataset)

---

## Estructura del repositorio

```
ids-iot/
│
├── environment.yml           # Entorno local Windows (CPU)
├── environment_hpc.yml       # Entorno servidor HPC Linux (GPU)
├── verificar_entorno.py      # Script de verificación de dependencias
├── README.md                 # Esta guía
│
├── notebooks/
│   └── IDS_IOT.ipynb         # Notebook principal de desarrollo
│
├── data/                     # (Ignorado por .gitignore — no subir al repo)
│   └── .gitkeep
│
└── artifacts/                # Modelos y encoders serializados
    └── .gitkeep
```

---

## Entorno local — Windows (CPU-only)

### 1. Crear el entorno

Abrí **Anaconda Prompt** (como Administrador) y ejecutá:

```bash
conda env create -f environment.yml
```

El proceso instala todas las dependencias en orden. Tarda ~5–10 minutos.

### 2. Activar el entorno

```bash
conda activate ids-iot-dev
```

Vas a ver `(ids-iot-dev)` al inicio del prompt. **Activá el entorno antes de cada sesión de trabajo.**

### 3. Registrar como kernel de Jupyter / VS Code

```bash
python -m ipykernel install --user --name ids-iot-dev --display-name "IDS-IoT Dev (Python 3.11)"
```

### 4. Verificar la instalación

```bash
python verificar_entorno.py
```

Salida esperada:

```
══════════════════════════════════════════════════════════
  Verificación de Entorno — ids-iot-dev
  Proyecto: Módulo de Detección para IDS Pasivo
══════════════════════════════════════════════════════════

  Python
  ──────────────────────────────────────────────────────
  [OK]   Python                 3.11.x

  Dependencias
  ──────────────────────────────────────────────────────
  [OK]   TensorFlow             2.14.0
  [OK]   Keras                  2.14.0
  [OK]   keras-tcn              3.5.0
  [OK]   QKeras                 0.9.0
  [OK]   scikit-learn           1.3.x
  [OK]   imbalanced-learn       0.11.0
  [OK]   pandas                 2.x.x
  [OK]   numpy                  1.x.x
  [OK]   pyarrow                12.x.x
  ...

  [✓] Entorno verificado. Listo para trabajar.
```

---

## Entorno servidor HPC — Linux (GPU)

### Prerequisitos en el servidor

TF 2.14 requiere **CUDA 11.8** y **cuDNN 8.6**. Verificar antes de crear el entorno:

```bash
nvidia-smi        # Verificar driver NVIDIA
nvcc --version    # Verificar CUDA Toolkit
```

### Crear el entorno en el servidor

```bash
conda env create -f environment_hpc.yml
conda activate ids-iot-train
python -m ipykernel install --user --name ids-iot-train --display-name "IDS-IoT Train (Python 3.11 GPU)"
```

### Verificar detección de GPU

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Debe retornar al menos un dispositivo GPU. Si retorna `[]`, revisar la instalación de CUDA.

---

## Flujo de trabajo entre entornos

```
Windows — ids-iot-dev (edición)    Linux HPC — ids-iot-train (entrenamiento)
────────────────────────────────   ─────────────────────────────────────────
Editar código y notebooks          git pull
Correr EDA liviano           →     Entrenar modelo con GPU
Validar pipelines                  git push artefactos (.keras, .joblib)
Analizar resultados          ←
```

Formas de sincronizar código con el servidor (definir según disponibilidad del lab):
- `git push` / `git pull` — recomendado para código y notebooks
- `scp` / `rsync` — para transferencia directa de artefactos pesados
- JupyterHub — si el laboratorio lo provee con acceso web

---

## Actualizar o reconstruir el entorno

```bash
# Actualizar dependencias manteniendo el entorno
conda activate ids-iot-dev
conda env update -f environment.yml --prune

# Reconstruir desde cero (más limpio ante conflictos)
conda deactivate
conda env remove -n ids-iot-dev
conda env create -f environment.yml
```

---

## Configurar KaggleHub

1. Ir a [kaggle.com](https://www.kaggle.com) → Account → API → **Create New Token**
2. Descargar `kaggle.json`
3. Colocarlo en:
   - **Windows:** `C:\Users\<TuUsuario>\.kaggle\kaggle.json`
   - **Linux:** `~/.kaggle/kaggle.json`

---

## Nota sobre hls4ml (Hito 5)

La librería `hls4ml` irá en un **tercer entorno separado** (`ids-iot-hls`) para evitar conflictos de dependencias. Se configurará al llegar a la fase de traducción a hardware. La razón por la que TF está acotado a 2.14 en los entornos actuales es precisamente garantizar que los modelos exportados sean compatibles con el conversor `keras_to_hls` de hls4ml sin conversiones intermedias.
