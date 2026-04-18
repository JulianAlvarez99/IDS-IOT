"""
verificar_entorno.py
====================
Script de verificación del entorno de desarrollo para el proyecto
"Módulo de Detección para IDS Pasivo".

Ejecutar con:
    python verificar_entorno.py

Valida que todas las dependencias críticas estén instaladas en las
versiones correctas y que el entorno esté listo para trabajar.

Decisión de versiones:
    - Python 3.11: punto óptimo de estabilidad para todo el stack.
    - TF 2.14:     última versión con Keras v2 integrado, requerido por
                   el conversor keras_to_hls de hls4ml (Hito 5).
"""

from __future__ import annotations

import sys
import importlib
from typing import NamedTuple


# ── Definición de dependencias esperadas ────────────────────────────────────

class DependencySpec(NamedTuple):
    """Especificación de una dependencia a verificar."""
    import_name: str       # Nombre para importar (puede diferir del paquete)
    display_name: str      # Nombre a mostrar en el reporte
    min_version: str       # Versión mínima aceptable
    max_version: str | None = None  # Versión máxima aceptable (inclusive)


DEPENDENCIES: list[DependencySpec] = [
    # TF acotado superiormente: >=2.15 rompe compatibilidad con hls4ml (Keras v3)
    DependencySpec("tensorflow",   "TensorFlow",        "2.14.0", "2.14.99"),
    DependencySpec("keras",        "Keras",             "2.14.0", "2.14.99"),
    DependencySpec("tcn",          "keras-tcn",         "3.5.0"),
    DependencySpec("qkeras",       "QKeras",            "0.9.0"),
    DependencySpec("sklearn",      "scikit-learn",      "1.3.0"),
    DependencySpec("imblearn",     "imbalanced-learn",  "0.11.0"),
    DependencySpec("pandas",       "pandas",            "2.0.0"),
    DependencySpec("numpy",        "numpy",             "1.23.5", "1.99.99"),
    DependencySpec("pyarrow",      "pyarrow",           "12.0.0"),
    DependencySpec("matplotlib",   "matplotlib",        "3.7.0"),
    DependencySpec("seaborn",      "seaborn",           "0.12.0"),
    DependencySpec("tqdm",         "tqdm",              "4.0.0"),
    DependencySpec("joblib",       "joblib",            "1.3.0"),
    DependencySpec("kagglehub",    "kagglehub",         "0.1.0"),
]


# ── Colores ANSI para la terminal ────────────────────────────────────────────

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def parse_version(version_str: str) -> tuple[int, ...]:
    """
    Convierte una cadena de versión en una tupla de enteros comparable.

    Maneja casos como '2.14.0', '2.14.0rc1', '1.3.2.post1'.

    Args:
        version_str: Cadena de versión a parsear.

    Returns:
        Tupla de enteros representando la versión (ej. (2, 14, 0)).
    """
    import re
    # Extraer solo la parte numérica inicial (ej. "2.14.0" de "2.14.0rc1")
    match = re.match(r"[\d.]+", version_str)
    clean = match.group(0) if match else "0.0.0"
    return tuple(int(x) for x in clean.split(".")[:3])


def check_python_version(required_major: int, required_minor: int) -> bool:
    """
    Verifica que la versión de Python del entorno sea la esperada.

    Args:
        required_major: Major version requerida (ej. 3).
        required_minor: Minor version requerida (ej. 11).

    Returns:
        True si la versión es correcta, False en caso contrario.
    """
    current = sys.version_info
    ok = (current.major == required_major and current.minor == required_minor)
    status = f"{GREEN}[OK]   {RESET}" if ok else f"{RED}[ERROR]{RESET}"
    version_str = f"{current.major}.{current.minor}.{current.micro}"

    print(f"  {status} {'Python':<22} {version_str}", end="")
    if not ok:
        print(
            f"  {YELLOW}→ Se requiere Python "
            f"{required_major}.{required_minor}.x{RESET}",
            end=""
        )
    print()
    return ok


def check_dependency(spec: DependencySpec) -> bool:
    """
    Importa un módulo y verifica que su versión esté dentro del rango requerido.

    Reporta [OK] si la versión es correcta, [WARN] si está fuera de rango
    pero instalada, y [FALTA] si no está instalada.

    Args:
        spec: Especificación de la dependencia a verificar.

    Returns:
        True si el módulo está instalado en versión correcta, False si falla.
    """
    try:
        module = importlib.import_module(spec.import_name)
        version_str: str = getattr(module, "__version__", "0.0.0")
        current = parse_version(version_str)
        minimum = parse_version(spec.min_version)

        above_min = current >= minimum
        below_max = True
        if spec.max_version is not None:
            below_max = current <= parse_version(spec.max_version)

        ok = above_min and below_max

        if ok:
            status = f"{GREEN}[OK]   {RESET}"
        else:
            status = f"{YELLOW}[WARN] {RESET}"

        print(f"  {status} {spec.display_name:<22} {version_str}", end="")

        if not above_min:
            print(f"  {YELLOW}→ Mínimo requerido: {spec.min_version}{RESET}", end="")
        elif not below_max:
            print(
                f"  {YELLOW}→ Máximo permitido: {spec.max_version} "
                f"(versiones superiores rompen compatibilidad con hls4ml){RESET}",
                end=""
            )
        print()
        return ok

    except ImportError:
        print(f"  {RED}[FALTA]{RESET} {spec.display_name:<22} No instalado")
        return False
    except Exception as e:
        print(f"  {RED}[ERROR]{RESET} {spec.display_name:<22} {e}")
        return False


def check_tensorflow_gpu() -> None:
    """
    Informa si TensorFlow detecta GPUs disponibles.

    En el entorno local Windows (CPU-only) es esperado que no haya GPUs.
    En el servidor HPC Linux debe detectar al menos una GPU NVIDIA.
    """
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            print(f"\n  {GREEN}[GPU]{RESET}  TensorFlow detectó {len(gpus)} GPU(s):")
            for gpu in gpus:
                print(f"         └── {gpu.name}")
        else:
            print(
                f"\n  {CYAN}[INFO]{RESET} No se detectaron GPUs."
            )
            print(
                f"         Esperado en entorno local Windows (CPU-only)."
            )
            print(
                f"         El entrenamiento con GPU se realiza en el servidor HPC Linux."
            )
    except Exception:
        pass


def main() -> None:
    """Punto de entrada principal del script de verificación."""

    print(f"\n{BOLD}{'═'*58}{RESET}")
    print(f"{BOLD}  Verificación de Entorno — ids-iot-dev{RESET}")
    print(f"{BOLD}  Proyecto: Módulo de Detección para IDS Pasivo{RESET}")
    print(f"{BOLD}{'═'*58}{RESET}\n")

    results: list[bool] = []

    # 1. Verificar versión de Python
    print(f"{BOLD}  Python{RESET}")
    print(f"  {'─'*54}")
    results.append(check_python_version(required_major=3, required_minor=11))

    # 2. Verificar todas las dependencias
    print(f"\n{BOLD}  Dependencias{RESET}")
    print(f"  {'─'*54}")
    for spec in DEPENDENCIES:
        results.append(check_dependency(spec))

    # 3. Verificar GPU (informativo, no penaliza el resultado)
    print(f"\n{BOLD}  Hardware{RESET}")
    print(f"  {'─'*54}")
    check_tensorflow_gpu()

    # 4. Resumen final
    total  = len(results)
    passed = sum(results)
    failed = total - passed

    print(f"\n{BOLD}  Resumen{RESET}")
    print(f"  {'─'*54}")
    print(f"  Total verificado : {total}")
    print(f"  {GREEN}Correctos{RESET}        : {passed}")

    if failed > 0:
        print(f"  {YELLOW}Con problemas{RESET}    : {failed}")
        print(
            f"\n  {YELLOW}[!] Ejecutá para resolver:{RESET}\n"
            f"      conda env update -f environment.yml --prune"
        )
        sys.exit(1)
    else:
        print(f"\n  {GREEN}{BOLD}[✓] Entorno verificado. Listo para trabajar.{RESET}\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
