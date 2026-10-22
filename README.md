# Hardware-Accelerated Passive Intrusion Detection System for IoT (IDS-IoT)

[![Python Version](https://img.shields.io/badge/Python-3.11.x-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14.0-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![QKeras](https://img.shields.io/badge/QKeras-0.9.0-412991?style=flat-square)](https://github.com/google/qkeras)
[![hls4ml](https://img.shields.io/badge/hls4ml-0.8.1-E05A47?style=flat-square)](https://fastmachinelearning.org/hls4ml/)
[![Hardware Target](https://img.shields.io/badge/FPGA-Xilinx%20PYNQ--Z2-1F88C0?style=flat-square&logo=xilinx&logoColor=white)](http://www.pynq.io/)
[![Dataset](https://img.shields.io/badge/Dataset-CIC--IoT--2023-4CAF50?style=flat-square)](https://www.unb.ca/cic/datasets/iot-dataset-2023.html)
[![License: Academic](https://img.shields.io/badge/License-Academic%20Research-blue.svg?style=flat-square)](#)

---

## 📌 Abstract & Overview

The proliferation of Internet of Things (IoT) edge devices introduces critical cybersecurity vulnerabilities due to strict constraints on computational power, memory bandwidth, and real-time processing latency. Traditional signature-based or compute-heavy Deep Learning Intrusion Detection Systems (IDS) are ill-suited for on-device deployment in high-throughput IoT networks.

This repository presents an **end-to-end, hardware-accelerated, passive Intrusion Detection System (IDS)** specifically designed for resource-constrained edge architectures. Operating over the comprehensive **CIC-IoT-2023** benchmark dataset (~22 million network flows), the system implements a complete research and engineering pipeline:

1. **Statistical Characterization & Explainable AI (XAI)**: High-dimensional flow profiling and SHAP/XGBoost feature attribution for optimal feature reduction.
2. **Constrained Neural Architecture Search (NAS)**: Deep Neural Network (DNN/MLP) design optimized under strict parameter budgets and trained using **Binary Focal Loss** to counter extreme class imbalances.
3. **Advanced Model Compression**: Systematic evaluation of Weight Pruning ($\sim 60\%$ target sparsity), Quantization-Aware Training (QAT with QKeras), Knowledge Distillation (KD), and combined Pruned Quantization-Aware Training (PQAT).
4. **FPGA High-Level Synthesis (HLS)**: Hardware translation via `hls4ml`, performing fixed-point precision profiling (`ap_fixed<W,I>`), C-simulation, RTL synthesis (VHDL/Verilog), and AXI-DMA IP Core generation for the **Xilinx PYNQ-Z2 (Zynq-7000 SoC)** board.

---

## 🔄 End-to-End Pipeline Architecture

```mermaid
flowchart TD
    subgraph DataPrep ["Stage 2: Statistical Profiling & Feature Selection"]
        D1[CIC-IoT-2023 Dataset<br/>~22M Network Flows] --> D2[E2_01_analisis_estadistico.ipynb<br/>EDA, Outliers & Class Distribution]
        D2 --> D3[E2_02_feature_importance_binario.ipynb<br/>XGBoost & SHAP Attribution]
        D2 --> D4[E2_03_feature_importance_multiclase.ipynb<br/>Multiclass Attack Profiling]
        D3 --> FS[Optimal Feature Subset<br/>& Balanced Datasets]
    end

    subgraph DeepLearning ["Stage 3: Deep Learning & Architectural Optimization"]
        FS --> M1[E3_04_NAS_binary_model.ipynb<br/>Neural Architecture Search]
        M1 --> M2[E3_05_entrenamiento_binario.ipynb<br/>Baseline DNN Training]
        M2 --> M3[E3_05b_entrenamiento_binario_mejorado.ipynb<br/>Enhanced Architecture & LR Schedules]
        M3 --> M4[E3_05d_entrenamiento_binario_focal_loss.ipynb<br/>Focal Loss for Extreme Imbalance]
        M4 --> M5[E3_05c_analisis_errores_holdout.ipynb<br/>Holdout Generalization & Error Analysis]
        M4 --> M6[E3_06_simulador_tiempo_real.ipynb<br/>Throughput & Latency Simulation]
    end

    subgraph Compression ["Stage 4: Edge Model Compression (TinyML)"]
        M4 --> C1[E4_07a_compresion_Pruning-QAT.ipynb<br/>Magnitude-based Pruning]
        M4 --> C2[E4_07b_compresion_Destillation.ipynb<br/>Teacher-Student Knowledge Distillation]
        M4 --> C3[E4_07c_compresion_PQAT.ipynb<br/>Joint Pruning + QKeras QAT]
        C1 & C2 & C3 --> C4[E4_07d_eval_compressed_model.ipynb<br/>Comparative Benchmarking on Holdout]
    end

    subgraph HardwareHLS ["Stage 5: High-Level Synthesis & FPGA Deployment"]
        C4 --> H1[E5_05a_HLS4ML.ipynb<br/>hls4ml Profile & C-Simulation]
        H1 --> H2[Fixed-Point Optimization<br/>ap_fixed<16,6> / ap_fixed<8,2>]
        H2 --> H3[Vivado HLS Synthesis<br/>AXI-Stream DMA Engine]
        H3 --> H4[Xilinx PYNQ-Z2 Bitstream<br/>Hardware IP Core]
    end

    style DataPrep fill:#f8f9fa,stroke:#495057,stroke-width:1px
    style DeepLearning fill:#eef2ff,stroke:#4338ca,stroke-width:1px
    style Compression fill:#f0fdf4,stroke:#15803d,stroke-width:1px
    style HardwareHLS fill:#fff7ed,stroke:#c2410c,stroke-width:1px
```

---

## 📂 Repository Structure & File Catalog

```
IDS-IOT/
├── notebooks/                 # Jupyter notebooks, one folder per stage (E<N>_ prefix = stage)
│   ├── Etapa_2/               # EDA & Feature Selection        (E2_01 … E2_03)
│   ├── Etapa_3/               # Training & Evaluation          (E3_04 … E3_06)
│   ├── Etapa_4/               # Model Compression              (E4_07a … E4_07d)
│   ├── Etapa_5/               # High-Level Synthesis (hls4ml)  (E5_05a + plotting.py)
│   │   └── docs/              #   Stage-5 guides: Vitis IP + PYNQ overlay, Vitis on Debian 13
│   └── Etapa_6/               # PYNQ-Z2 bring-up               (E6_08a)
│       └── docs/              #   Stage-6 guide: AND gate in the PL
├── data/                      # Datasets (Feather): dataset_eda_temp, df_binary_balanced, test_real_traffic
├── models/                    # Trained weights (.h5), scalers (.pkl), architectures (.json), xgboost/
├── figuras/                   # All images & plots, by stage — README lists the notebook that made each one
├── vitis/                     # Vitis HLS 2022.2 implementations
│   ├── hls4ml/                #   hls4ml firmware: pynq_modelo_estudiante_destilado_qat_prj
│   └── ids_axi/               #   AXI wrapper IP: src/, scripts/, ids_axi_prj/, ip/
├── vivado/                    # Vivado 2022.2 implementations
│   ├── and_gate_pynqz2/       #   AND-gate test design (src, constraints, Tcl, and_gate_prk/)
│   └── ids_overlay/           #   IDS block design (constraints, build_overlay.tcl, project_ids/)
├── embebido/                  # Files deployed to the PYNQ-Z2 (.bit/.hwh, board drivers)
│   ├── and_gate/
│   └── ids_overlay/
├── docs/                      # General project documentation (+ referencias/ manuals)
├── enviroments/               # Cross-Platform Conda Environments
├── artifacts/                 # Serialized Artifacts & Encoders
└── README.md                  # Project Master Documentation
```

---

## 🔬 Granular File & Notebook Catalog

### 1. `notebooks/Etapa_2/` — Statistical Characterization & Feature Selection
| File | Scientific Objective & Methodology | Primary Outputs / Artifacts |
|---|---|---|
| [`E2_01_analisis_estadistico.ipynb`](notebooks/Etapa_2/E2_01_analisis_estadistico.ipynb) | Exploratory Data Analysis (EDA) of the raw CIC-IoT-2023 dataset. Evaluates 47+ features across ~22M flows, detecting nulls, infinite values, skewed distributions, and inter-feature Pearson correlations. | Distribution plots, percentile distributions, data sanitization specs. |
| [`E2_02_feature_importance_binario.ipynb`](notebooks/Etapa_2/E2_02_feature_importance_binario.ipynb) | Tree-based feature importance ranking for binary classification (Benign vs. Attack). Trains an optimized XGBoost model and applies **SHAP (SHapley Additive exPlanations)** to isolate key hardware-friendly features. | `xgb_modelo_binario.json`, `SHAP_BarPlot_Binary.png`, `SHAP_Beeswarm_binary.png`. |
| [`E2_03_feature_importance_multiclase.ipynb`](notebooks/Etapa_2/E2_03_feature_importance_multiclase.ipynb) | Multiclass feature ranking identifying discriminative features capable of differentiating specific attack taxonomies (DDoS, DoS, Recon, Web, BruteForce, Spoofing, Mirai). | Multiclass importance matrices and attack-specific feature signatures. |

### 2. `notebooks/Etapa_3/` — Deep Learning Modeling & Architectural Optimization
| File | Scientific Objective & Methodology | Primary Outputs / Artifacts |
|---|---|---|
| [`E3_04_NAS_binary_model.ipynb`](notebooks/Etapa_3/E3_04_NAS_binary_model.ipynb) | Constrained Neural Architecture Search (NAS) evaluating layer depths, neuron widths, activation functions, and parameter count trade-offs suitable for FPGA BRAM/DSP limits. | `arquitectura_ganadora_binario.png`, optimal DNN baseline topology. |
| [`E3_05_entrenamiento_binario.ipynb`](notebooks/Etapa_3/E3_05_entrenamiento_binario.ipynb) | Baseline training of the selected DNN model on balanced network traffic partitions (`df_binary_balanced.feather`). | Initial model checkpoints (`51epochs_128batch`, `71epochs_64batch`, etc.). |
| [`E3_05b_entrenamiento_binario_mejorado.ipynb`](notebooks/Etapa_3/E3_05b_entrenamiento_binario_mejorado.ipynb) | Hyperparameter tuning and architecture refinement utilizing batch normalization, dropout, Cosine Annealing/ReduceLROnPlateau learning schedules. | Improved baseline model (`mlp_v2`). |
| [`E3_05c_analisis_errores_holdout.ipynb`](notebooks/Etapa_3/E3_05c_analisis_errores_holdout.ipynb) | Holdout validation on unseen, natural (unbalanced) traffic (`test_real_traffic.feather`). Evaluates ROC-AUC, PR-AUC, False Positive Rate (FPR), and decision threshold sensitivity. | Confusion matrices, threshold calibration curves, per-class error reports. |
| [`E3_05d_entrenamiento_binario_focal_loss.ipynb`](notebooks/Etapa_3/E3_05d_entrenamiento_binario_focal_loss.ipynb) | Training with **Binary Focal Loss** ($\alpha=0.25, \gamma=2.0$) to mitigate class imbalance and focus gradients on hard-to-classify edge samples. | Robust teacher model with suppressed false alarms. |
| [`E3_06_simulador_tiempo_real.ipynb`](notebooks/Etapa_3/E3_06_simulador_tiempo_real.ipynb) | Real-time traffic stream simulation testing batch inference throughput, per-packet classification latency, and memory consumption. | Latency vs. batch size benchmarks in software execution. |

### 3. `notebooks/Etapa_4/` — Edge Compression (TinyML)
| File | Scientific Objective & Methodology | Primary Outputs / Artifacts |
|---|---|---|
| [`E4_07a_compresion_Pruning-QAT.ipynb`](notebooks/Etapa_4/E4_07a_compresion_Pruning-QAT.ipynb) / `_corregido` | Magnitude-based weight pruning enforcing 50%–70% sparsity combined with Quantization-Aware Training using TensorFlow Model Optimization (TF-MOT). | `QAT_15epochs_128batch/` compressed weights. |
| [`E4_07b_compresion_Destillation.ipynb`](notebooks/Etapa_4/E4_07b_compresion_Destillation.ipynb) | Knowledge Distillation (KD) transferring logits from the high-capacity teacher network to an ultra-compact student model ($< 5\text{k}$ parameters). | `Destillation_15epochs_128batch/` student model. |
| [`E4_07c_compresion_PQAT.ipynb`](notebooks/Etapa_4/E4_07c_compresion_PQAT.ipynb) | Simultaneous **Pruning + Quantization-Aware Training (PQAT)** using **QKeras** and custom Focal Loss, quantizing weights (`quantized_bits(8,2)`) and activations (`quantized_relu(8,2)`). | `PQAT_15epochs_128batch/` hardware-ready QModel. |
| [`E4_07d_eval_compressed_model.ipynb`](notebooks/Etapa_4/E4_07d_eval_compressed_model.ipynb) | Rigorous comparative benchmark comparing Float32 baseline vs. Pruned vs. QAT vs. Distilled vs. PQAT on natural test traffic (`test_real_traffic.feather`). | Comprehensive metrics table (Accuracy, Precision, Recall, F1, Sparsity, Compression Ratio). |

### 4. `notebooks/Etapa_5/` — High-Level Synthesis (HLS) & Hardware IP
| File | Scientific Objective & Methodology | Primary Outputs / Artifacts |
|---|---|---|
| [`E5_05a_HLS4ML.ipynb`](notebooks/Etapa_5/E5_05a_HLS4ML.ipynb) | Full HLS pipeline using `hls4ml`. Converts the QKeras PQAT model into C++ HLS firmware, configures AXI-Stream interfaces (`io_stream`), sets parallelization reuse factors (Reuse Factor = 64), runs C-simulation, and invokes Vivado HLS for RTL synthesis. | Synthesized VHDL/Verilog RTL, Vivado IP Core (`.zip`), C-Simulation validation reports. |
| [`plotting.py`](notebooks/Etapa_5/plotting.py) | Utility module for visualizing precision profiling and layer-by-layer quantization errors during HLS conversion. | Precision profiling figures. |
| [`docs/ip_core_vitis_y_overlay_pynq.md`](notebooks/Etapa_5/docs/ip_core_vitis_y_overlay_pynq.md) | Step-by-step guide: hls4ml firmware → `ids_axi` IP in Vitis HLS → Vivado block design → PYNQ overlay. | — |
| [`docs/entorno_vitis_debian13.md`](notebooks/Etapa_5/docs/entorno_vitis_debian13.md) | The four environment fixes Vitis HLS 2022.2 needs on Debian 13. | — |

The notebook writes the hls4ml firmware to [`vitis/hls4ml/`](vitis/hls4ml/) (wiped and regenerated on every run).

### 5. `notebooks/Etapa_6/` — PYNQ-Z2 Bring-up
| File | Scientific Objective & Methodology | Primary Outputs / Artifacts |
|---|---|---|
| [`E6_08a_pynq_z2_primeros_pasos.ipynb`](notebooks/Etapa_6/E6_08a_pynq_z2_primeros_pasos.ipynb) | First steps on the board: Jupyter, base overlay, loading custom bitstreams. | — |
| [`docs/and_gate_pynqz2.md`](notebooks/Etapa_6/docs/and_gate_pynqz2.md) | Minimal PL-only design (AND gate) validating the Vivado → `.bit` + `.hwh` → PYNQ flow. | `embebido/and_gate/` |

### 6. Hardware — `vitis/`, `vivado/`, `embebido/`
| Folder | Content |
|---|---|
| [`vitis/`](vitis/README.md) | Vitis HLS 2022.2 projects: `hls4ml/` (generated network firmware) and `ids_axi/` (AXI wrapper, `build_ip.tcl`, exported IP in `ip/`). |
| [`vivado/`](vivado/README.md) | Vivado 2022.2 projects: `and_gate_pynqz2/` (test design) and `ids_overlay/` (Zynq PS + AXI DMA + `ids_axi` block design). |
| [`embebido/`](embebido/README.md) | What goes to the board: `.bit`/`.hwh` pairs and the board-side Python (`ids_driver.py`, `ids_server.py`), plus the PC-side `host_sender.py`. |
| [`figuras/`](figuras/README.md) | Every image and plot, grouped by stage, with the notebook it comes from. |

### 7. `enviroments/` — Cross-Platform Environment Configurations
| File | Description & Purpose |
|---|---|
| [`environment.yml`](enviroments/environment.yml) | Conda specification for local workstation development on Windows (CPU-only, Python 3.11, TensorFlow 2.14.0, QKeras, pyarrow). |
| [`environment_hpc.yml`](enviroments/environment_hpc.yml) | Conda specification for High-Performance Computing (HPC) Linux servers with NVIDIA GPU acceleration (CUDA 11.8 / cuDNN 8.6). |
| [`environment_linux.yml`](enviroments/environment_linux.yml) | Standard Linux GPU environment definition. |
| [`verificar_entorno.py`](enviroments/verificar_entorno.py) | Automated sanity-check script validating package versions, Keras 2 compatibility, and GPU availability. |
| [`README.md`](enviroments/README.md) | Detailed documentation on environment setup, version rationale, and KaggleHub configuration. |

---

## ⚙️ Mathematical & Theoretical Formulations

### 1. Binary Focal Loss for Imbalanced Flow Detection
Standard Cross-Entropy struggles with extreme class imbalances where benign traffic overwhelms anomalous attack packets. We formulate the detection objective with Focal Loss:

$$\mathcal{L}_{\text{FL}}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

where:
- $p_t \in [0, 1]$ is the model's estimated probability for the ground-truth class.
- $\alpha_t \in [0, 1]$ is the weighting factor balancing positive and negative classes.
- $\gamma \ge 0$ is the tunable focusing parameter (set to $\gamma = 2.0$) that exponentially down-weights well-classified easy examples.

### 2. Quantization-Aware Training (QAT)
Continuous floating-point weights $w \in \mathbb{R}$ and activations $a \in \mathbb{R}$ are projected into quantized fixed-point representations $\text{ap\_fixed}\langle W, I \rangle$ (where $W$ is total bitwidth and $I$ is the integer bitwidth):

$$q(x) = \text{clamp}\left( \left\lfloor \frac{x}{\Delta} \right\rceil, -2^{W-1}, 2^{W-1}-1 \right) \cdot \Delta, \quad \Delta = 2^{-(W - I)}$$

The Straight-Through Estimator (STE) is utilized during the backward pass to propagate gradients through non-differentiable rounding operators.

### 3. Hardware Parallelism & Reuse Factor
In `hls4ml`, computational latency versus FPGA hardware resource utilization (DSP48E slices, LUTs, FFs, BRAM) is controlled by the **Reuse Factor ($R$)**:

$$\text{DSP Usage} \propto \frac{N_{\text{mult}}}{R}, \qquad \text{Latency (Clock Cycles)} \propto R$$

A balanced configuration ($R = 64$, Strategy = `Resource`, IO = `io_stream`) is adopted to meet real-time line-rate packet inspection constraints while fitting within the Xilinx Zynq XC7Z020 resource envelope.

---

## 🚀 Getting Started & Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/JulianAlvarez99/IDS-IOT.git
cd IDS-IOT
```

### 2. Set Up the Conda Environment

#### For Local Development (Windows CPU):
```bash
conda env create -f enviroments/environment.yml
conda activate ids-iot-dev
python -m ipykernel install --user --name ids-iot-dev --display-name "IDS-IoT Dev (Python 3.11)"
```

#### For Server Training (Linux GPU / HPC):
```bash
conda env create -f enviroments/environment_hpc.yml
conda activate ids-iot-train
python -m ipykernel install --user --name ids-iot-train --display-name "IDS-IoT Train (Python 3.11 GPU)"
```

### 3. Verify Environment Integrity
Run the built-in diagnostics script to verify all core dependencies:
```bash
python enviroments/verificar_entorno.py
```

### 4. Dataset Configuration (KaggleHub)
Ensure your Kaggle API token `kaggle.json` is located in `~/.kaggle/kaggle.json` (Linux) or `C:\Users\<User>\.kaggle\kaggle.json` (Windows) to allow automated streaming and caching of the CIC-IoT-2023 dataset.

---

## 📊 Summary of Model Architectures & Artifacts

| Model Type | Primary Method | Bitwidth / Sparsity | Target Platform | Checkpoint Directory |
|---|---|---|---|---|
| **Baseline DNN** | Full Precision (FP32) | 32-bit Float / 0% | CPU / GPU Server | `models/51epochs_128batch/`, `models/mlp_v2/` |
| **Distilled Student**| Knowledge Distillation | 32-bit Float / Compact | Edge CPU | `models/Destillation_15epochs_128batch/` |
| **Pruned QAT** | TF-MOT Pruning + QAT | 8-bit Int / ~60% Sparse | Edge TPU / MCU | `models/QAT_15epochs_128batch/` |
| **PQAT (QKeras)** | Joint Prune + QKeras | `ap_fixed<8,2>` / ~60% | **Xilinx PYNQ-Z2 FPGA** | `models/PQAT_15epochs_128batch/` |

---

## 🎓 Academic Attribution & Citation

If you use this codebase, models, or synthesis configurations in your research, please cite:

```bibtex
@misc{alvarez2026idsiot,
  author       = {Alvarez, Juli{\'a}n Manuel},
  title        = {{Hardware-Accelerated Passive Intrusion Detection System for IoT (IDS-IoT)}},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/JulianAlvarez99/IDS-IOT}},
  note         = {M{\'o}dulo de Detecci{\'o}n para IDS Pasivo implementado en FPGA mediante QKeras y HLS4ML}
}
```

---

## 📄 License & Acknowledgments
- Developed as part of academic research on edge network security and hardware acceleration.
- Dataset provided by the **Canadian Institute for Cybersecurity (CIC)**: [CIC-IoT-2023](https://www.unb.ca/cic/datasets/iot-dataset-2023.html).
