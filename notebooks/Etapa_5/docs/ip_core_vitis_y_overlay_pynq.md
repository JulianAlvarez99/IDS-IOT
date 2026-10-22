# Etapa 5b — IP core en Vitis HLS y overlay para PYNQ-Z2

Convierte la red generada por hls4ml en un **IP core de Vivado** empaquetado, con
interfaces AXI y LEDs de veredicto gobernados por hardware, y lo despliega como
overlay de PYNQ.

El trabajo está repartido en el repositorio así (rutas desde la raíz):

| Qué | Dónde |
|---|---|
| Firmware generado por hls4ml (salida del notebook) | `vitis/hls4ml/` |
| Envoltorio `ids_axi` para Vitis HLS (fuentes, scripts, IP exportado) | `vitis/ids_axi/` |
| Block design de Vivado (constraints, `build_overlay.tcl`, proyecto) | `vivado/ids_overlay/` |
| Archivos que van a la placa (`.bit`/`.hwh`, driver y servidor) | `embebido/ids_overlay/` |

`vitis/ids_axi/` está **fuera** de `vitis/hls4ml/` a propósito: el notebook
`E5_05a_HLS4ML.ipynb` llama a `shutil.rmtree()` sobre su directorio de salida al
principio de cada ejecución, así que todo lo que se guarde allí se destruye en la
siguiente regeneración. Nada de `vitis/ids_axi/` lo sobrescribe el notebook.

> **Entorno del servidor.** Vitis HLS 2022.2 no funciona sobre Debian 13 sin
> cuatro correcciones previas (enlazador, rutas de bibliotecas y ruta de pesos).
> Están documentadas en [`entorno_vitis_debian13.md`](entorno_vitis_debian13.md).
> Aplícalas antes del paso 1 o la simulación C no llegará a enlazar.

---

## 1. Arquitectura

```
   PC (Windows)                    PYNQ-Z2  ┌─── PS (ARM / Linux) ───┐┌── PL (tejido FPGA) ──┐
┌──────────────────┐   Ethernet   │         │                        ││                      │
│ 18 características│  TCP :9000  │  eth0   │  ids_server.py         ││   ┌──────────────┐   │
│ StandardScaler   ├─────────────►│────────►│  cuantiza → uint32     │├──►│  AXI DMA     │   │
│ vector float32   │              │         │  pynq.allocate()       ││   └──────┬───────┘   │
│ host_sender.py   │              │         │                        ││          │ AXIS 32b  │
└──────────────────┘              │         │  AXI-Lite: N, umbral   ││   ┌──────▼───────┐   │
        ▲                         │         │  ───────────────────────┼──►│   ids_axi    │   │
        │   veredictos de vuelta  │         │                        ││   │  (envoltorio)│   │
        └─────────────────────────┤         │                        ││   │  ┌─────────┐ │   │
                                  │         │                        ││   │  │myproject│ │   │
                                  └─────────┘                        ││   │  │ hls4ml  │ │   │
                                                                     ││   │  └─────────┘ │   │
                                                                     ││   └──┬────────┬──┘   │
                                                                     │└──────┘        │led_o │
                                                                     │  AXIS          │[5:0] │
                                                                     └────────────────┼──────┘
                                                                                      ▼
                                                                          LEDs RGB LD4/LD5
                                                                    verde=benigno  rojo=ataque
```

Dos hechos estructurales condicionan todo el diseño:

**El Ethernet pertenece al PS, no al PL.** El MAC Ethernet del Zynq-7020 está
cableado al Processing System, así que los paquetes *tienen* que llegar primero a
Linux. En esta placa no existe ningún camino que alimente el tejido directamente
desde el cable. El PS es, por tanto, la capa de transporte, y el PL es el motor
de inferencia.

**El veredicto nunca vuelve por la CPU.** `ids_axi` expone un puerto crudo
`led_o` de 6 bits conectado directamente a los pines de los LEDs RGB. La cadena
clasificación → iluminación ocurre íntegramente en el tejido; el ARM solo mueve
datos. Ésa es la propiedad que conviene defender en la memoria, y es la razón de
que los LEDs no cuelguen de un AXI GPIO.

---

## 2. Leer esto antes de sintetizar: el firmware tal como se genera está roto

Simulando bit a bit la aritmética de punto fijo generada frente al modelo en
coma flotante, sobre 40.000 paquetes reservados y con umbral 0,5:

| Configuración | Exactitud | Precisión | Sensibilidad | F1 |
|---|---|---|---|---|
| Keras float32 (referencia) | 0,9913 | 0,9929 | 0,9896 | 0,9913 |
| **Firmware actual** — todo `ap_fixed<8,2>`, `AP_WRAP` | **0,5223** | 0,5102 | 0,9883 | 0,6730 |
| …igual pero con `AP_SAT` | 0,7328 | 0,6514 | 0,9954 | 0,7875 |
| …+ entrada ampliada a `<16,6>` únicamente | 0,5120 | 0,5048 | 0,9850 | 0,6675 |
| …+ activaciones `ufixed<4,0>` correctas únicamente | 0,4961 | 0,4967 | 0,9948 | 0,6626 |
| **Tras `regenerate_firmware.py`** | **0,9933** | 0,9983 | 0,9881 | 0,9932 |

La configuración actual apenas supera al azar: etiqueta casi todo como ataque
(sensibilidad 0,988, precisión 0,510). Fíjate en las filas intermedias:
**corregir una sola causa deja el sistema igual de roto.** Hay que corregir las
tres.

**Causa 1 — `granularity='model'` no emite ningún tipo por capa.** Verificado
directamente contra hls4ml: con `granularity='name'` deriva
`ufixed<4,0,RND_CONV,SAT,0>` para las dos capas QActivation, directamente de sus
cuantizadores `quantized_relu(4,0)`. Con `granularity='model'` la sección
`LayerName` queda vacía y todo colapsa sobre el valor global por defecto.

**Causa 2 — `AP_WRAP` es el modo por defecto de `ap_fixed`.** Un valor fuera de
rango *envuelve* hacia un número pequeño o negativo en lugar de saturar. Las
salidas de BatchNorm de tu modelo llegan a **+34,7**, así que activaciones que
`quantized_relu(4,0)` satura a 0,9375 durante el entrenamiento envuelven a casi
cero en hardware. Éste es el fallo dominante: incluso con anchos generosos,
`AP_WRAP` solo alcanza 0,6026.

**Causa 3 — `ap_fixed<8,2>` no se acerca a los rangos medidos:**

| Señal | Rango medido | Necesita | El notebook ponía |
|---|---|---|---|
| salida de `qdense_1` | [−325,6 · +275,2] | 10 bits enteros | `ap_fixed<8,2>` = ±2 |
| salida de `batch_normalization` | [−151,6 · +34,7] | 9 bits enteros | `ap_fixed<8,2>` = ±2 |
| acumulador de `qdense_2` | ±88,7 | 8 bits enteros | `ap_fixed<8,2>` = ±2 |
| acumulador de `output` | ±20,1 | 6 bits enteros | `ap_fixed<8,2>` = ±2 |

El valor por defecto *propio* de hls4ml ya es `fixed<16,6>`. El notebook
sustituyó un valor por defecto sensato por uno roto.

### El puerto de entrada es un problema aparte, y mucho menor

El 40,4 % de los paquetes tiene al menos una característica fuera de [−2, +2):
`flow_duration` llega a **+302 σ**, `fin_count` a +255, `rate` a +101. Pero
ampliar la entrada *por sí sola* mueve la exactitud extremo a extremo solo unos
0,06 puntos porcentuales, porque el modelo es robusto a esos extremos. Conviene
ampliarla igualmente —es casi gratis—, pero sin confundirla con el problema
principal.

> El envoltorio lee `input_t::value_type` e `input_t::size` del `defines.h`
> generado, así que recompila con *cualquier* precisión. Mantén los parámetros
> `--feature-w/--feature-i` del driver alineados con lo que acabe diciendo
> `defines.h`.

### Nota sobre versiones

Esto se validó con **hls4ml 1.3.0**, cuya propagación de precisión `auto`
calcula anchos intermedios sin pérdidas (por ejemplo `ap_fixed<47,22>` tras
BatchNorm — correcto, aunque generoso). Tu servidor ejecuta **0.7.1**, según
`enviroments/environment_linux.yml`, que resuelve las capas sin nombre al valor
por defecto del modelo. La fila simulada de 0,9933 modela ese comportamiento,
así que ambas sirven; 1.3.0 es simplemente más ajustada.

Las dos derivan el tipo de las QActivation a partir del cuantizador QKeras de
formas ligeramente distintas, así que `05a_HLS4ML.ipynb` no se fía: comprueba el
resultado y, si alguna capa se quedó sin tipo saturante, lo reconstruye leyendo
`bits`/`integer` del propio cuantizador. La celda 1 imprime `hls4ml.__version__`
para que quede registrado con qué versión se generó cada firmware.

---

## 3. Contenido

```
vitis/ids_axi/
├── src/
│   ├── ids_axi.h            declaración del top, códigos de LED, tipos derivados
│   ├── ids_axi.cpp          el envoltorio: interfaces, pragmas, lotes
│   └── ids_axi_tb.cpp       banco de pruebas de csim (chequeos estructurales + códec)
├── scripts/
│   ├── regenerate_firmware.py  reejecuta hls4ml con precisión sensata
│   └── build_ip.tcl            Vitis HLS: csim → csynth → export IP
├── ids_axi_prj/             proyecto Vitis HLS (generado, no versionado)
└── ip/                      IP core exportado (generado, no versionado)

vivado/ids_overlay/
├── constraints/
│   └── pynq_z2_ids_leds.xdc    asignación de pines de los LEDs RGB
├── scripts/
│   └── build_overlay.tcl       Vivado: block design → bitstream (ver aviso en el archivo)
└── project_ids/             proyecto Vivado (generado, no versionado)

embebido/ids_overlay/
├── ids_driver.py        driver de placa (DMA + códec de cuantización)
├── ids_server.py        front-end TCP en la placa
└── host_sender.py       reproducción de tráfico desde el PC + informe de exactitud
```

---

## 4. Etapa A — Vitis HLS 2022.2

### A.0 · Preparar el entorno del servidor

Solo la primera vez, y solo en Debian 13. Detalle completo en
[`entorno_vitis_debian13.md`](entorno_vitis_debian13.md).

```bash
# El enlazador empaquetado (binutils 2.37) no entiende las secciones .relr.dyn
# de la glibc moderna. Se sustituye por el del sistema.
cd /home/fitba/tools/Xilinx/Vivado/2022.2/tps/lnx64/binutils-2.37/bin && mv ld ld.xilinx-backup && ln -s /usr/bin/ld ld
```

```bash
# En cada sesión, antes de lanzar Vitis: rutas de búsqueda del enlazador.
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu
```

La GUI hereda el entorno del proceso que la lanza, así que **arranca Vitis desde
ese mismo terminal**; un icono de escritorio no recoge el `export`.

### A.1 · Regenerar el firmware

Dos caminos equivalentes; usa el que te resulte cómodo.

**Desde el notebook** (`notebooks/Etapa_5/E5_05a_HLS4ML.ipynb`), que ya lleva la
configuración correcta incorporada. Ejecuta las celdas 1 a 4b. La 4b es
obligatoria: aplica la ruta absoluta de los pesos y verifica los tipos emitidos.

**Desde la línea de órdenes**, en la máquina Linux que tiene hls4ml:

```bash
cd vitis/ids_axi
python scripts/regenerate_firmware.py
```

Ambos producen el mismo firmware. El script es útil para reproducir la
generación sin abrir Jupyter; el notebook, para documentar la decisión de
precisión junto al resto de la Etapa 5.

Sáltate este paso solo si has decidido conservar `ap_fixed<8,2>`; en ese caso el
`vitis/hls4ml/.../firmware/` existente sirve tal cual — con la exactitud de 0,5223 de
la tabla del §2.

Comprobación de que ha hecho lo que debe:

```bash
grep "typedef ap_" vitis/hls4ml/pynq_modelo_estudiante_destilado_qat_prj/firmware/defines.h
```

Debe aparecer `input_t` como `ap_fixed<16,6,AP_RND,AP_SAT>` y los dos tipos de
activación como `ap_ufixed<4,0,...,AP_SAT>`. Si sigues viendo `<8,2>`, el script
no se ejecutó sobre ese directorio.

> El script también inserta la ruta absoluta de los pesos en
> `firmware/nnet_utils/nnet_helpers.h`, necesaria para la simulación C. hls4ml
> vuelve a copiar ese árbol en cada regeneración, por eso se aplica de forma
> automática.

### A.2 · Crear el proyecto

**Por script (recomendado, reproducible):**

```bash
cd vitis/ids_axi
vitis_hls -f scripts/build_ip.tcl
```

Ejecuta simulación C, síntesis C y exporta el IP a `vitis/ids_axi/ip/`. Las etapas se
seleccionan por variables de entorno:

```bash
IDS_CSIM=1 IDS_SYNTH=0 IDS_EXPORT=0 vitis_hls -f scripts/build_ip.tcl   # solo csim
IDS_COSIM=1 vitis_hls -f scripts/build_ip.tcl                            # añade cosimulación
```

**Por GUI**, si prefieres explorar el planificador y los informes:

1. `vitis_hls` → **Create Project** → nombre `ids_axi_prj`, ubicación libre.
2. **Add design files** (clic derecho en *Source*):
   - `src/ids_axi.cpp`
   - `<proyecto hls4ml>/firmware/myproject.cpp`
3. **Top Function**: pulsa *Browse* solo después de añadir las fuentes, y elige
   `ids_axi`.
4. **Add testbench file**: `src/ids_axi_tb.cpp`.

> **No copies nada dentro del proyecto hls4ml.** Los archivos se añaden desde
> `src/`, donde están. Son dos árboles con dueños distintos: `src/` lo escribes
> tú y está en git; `firmware/` lo genera hls4ml y **se borra entero** en cada
> regeneración (`shutil.rmtree`). Una copia dentro de `firmware/` desaparece sin
> aviso la próxima vez que ejecutes la celda 4 del notebook, y el proyecto de
> Vitis se queda apuntando a archivos inexistentes.
>
> No hace falta copiar: `ids_axi.cpp` y `ids_axi_tb.cpp` incluyen `"ids_axi.h"`
> con comillas, que resuelven contra el directorio del propio archivo. El único
> include que cruza de un árbol al otro es `myproject.h`, y para eso está el
> `-I<firmware>` del paso 5.
>
> `ids_axi.h` tampoco se añade al proyecto: Vitis HLS solo lleva `.cpp` en
> *Source* y *Test Bench*; las cabeceras se localizan por rutas de inclusión.
5. En los **tres** archivos, clic derecho → *Edit CFLAGS…* → escribe exactamente
   esto, sustituyendo por tu ruta absoluta real:
   ```
   -std=c++14 -I/ruta/absoluta/al/pynq_..._prj/firmware
   ```
   Las rutas de `-I` deben ser **absolutas**: la GUI resuelve las relativas
   contra `solution1/csim/build`, cuatro niveles más abajo. **No** añadas
   `-DWEIGHTS_DIR` — ver la nota al final de esta etapa.
6. **Solution Configuration**: parte `xc7z020clg400-1`, periodo de reloj
   **10 ns**, *flow target* **Vivado**.

> El proyecto del PYNQ-Z2 lleva `myproject_test.cpp` en la carpeta del firmware,
> que es el banco de pruebas propio de hls4ml y tiene su propio `main()`. No lo
> añadas: dos `main()` producen un error de enlazado que no se parece en nada a
> su causa.

### A.3 · Simulación C

**Run C Simulation.** Compila el envoltorio y el núcleo con GCC y ejecuta el
banco de pruebas en el PC. Resultado esperado:

```
PASS - all structural checks succeeded
```

Qué comprueba realmente:

| Chequeo | Qué valida |
|---|---|
| un resultado por paquete | el lazo de lotes consume y produce en la proporción correcta |
| `TLAST` solo en la última palabra | el DMA sabrá cuándo termina la transferencia |
| `led_o` = reducción OR del lote | rojo si **algún** paquete del lote fue marcado |
| `n_packets == 0` no hace nada | el caso degenerado no bloquea el núcleo |

> **La simulación C no verifica la exactitud de la red.** Un `PASS` con el
> firmware `<8,2>` roto es perfectamente posible: los chequeos son estructurales.
> La exactitud se confirma en la cosimulación (A.5).

Si falla, el mensaje indica qué chequeo y el índice del paquete.

### A.4 · Síntesis C

**Run C Synthesis.** Traduce C++ a RTL. Tarda unos minutos. Al terminar se abre
el *Synthesis Summary*; hay que leer cuatro bloques.

**1 · Timing.** El *Estimated* debe quedar por debajo de los 10 ns menos la
incertidumbre (1,25 ns), es decir **≤ 8,75 ns**. Si no cumple:

- sube el `ReuseFactor` de las capas densas — más ciclos, menos lógica combinacional por ciclo;
- confirma que la estrategia es `Resource` y no `Latency`;
- como último recurso, relaja el reloj a 12 ns y ajusta después `FCLK_CLK0` en Vivado para que coincida.

**2 · Latency.** Interesa el intervalo de iniciación (II) del lazo de lotes. Con
`n_packets` en tiempo de ejecución, el informe mostrará un rango o `?`; es
normal, y por eso `ids_axi.cpp` lleva `LOOP_TRIPCOUNT`, que solo afecta al
informe, no al hardware.

**3 · Interface.** Es la tabla que hay que revisar con más cuidado, porque un
error aquí no se detecta hasta Vivado:

| Puerto | Modo esperado |
|---|---|
| `in_stream`, `out_stream` | `axis` |
| `s_axi_CTRL` | `s_axilite` |
| `led_o` | salida `ap_none` |
| `n_packets`, `threshold_raw` | dentro del bundle `CTRL` |

Si `led_o` aparece como `ap_vld` o `ap_hs`, falta el pragma correspondiente y en
Vivado saldrán pines de handshake que nadie conecta.

**4 · Utilization Estimates.** Presupuesto del **xc7z020**:

| Recurso | Disponible | Esperado |
|---|---|---|
| LUT | 53.200 | 15–25 % |
| FF | 106.400 | 10–20 % |
| DSP48E | 220 | ~30–40 |
| BRAM_18K | 280 | 10–20 |

Los DSP son el recurso crítico en esta pieza. Si te pasas de 220, sube el
`ReuseFactor` en `regenerate_firmware.py` (recuerda: debe ser divisor de
`n_in × n_out`; los válidos son 36 / 32 / 16) y vuelve a A.1.

### A.5 · Cosimulación (opcional pero recomendada)

**Run Cosimulation** → *Verilog*, *Dump trace: none*.

Ejecuta el mismo banco de pruebas contra el RTL generado en lugar del C++. Es
**la única etapa que confirma que el hardware calcula lo que el modelo dice**, y
por tanto la que respalda la cifra de 0,9933 del §2. Tarda bastante más que la
síntesis.

Necesita enlazar un ejecutable, así que exige las correcciones de A.0.

Si se queda colgada, casi siempre es `TLAST`: el banco de pruebas envía un número
de palabras distinto del que dice `n_packets`.

### A.6 · Exportar el IP

**Export RTL** con estos valores exactos:

| Campo | Valor |
|---|---|
| Format | *IP Catalog* |
| RTL | *Verilog* |
| Vendor | `utn` |
| Library | `ids` |
| Version | `1.0` |
| Display name | `IDS-IoT NN Accelerator` |
| Output location | `vitis/ids_axi/ip` |

El resultado es un `.zip` que Vivado consume como repositorio de IP. El VLNV
completo queda como `utn:ids:ids_axi:1.0`; anótalo, porque es lo que buscarás en
el catálogo de Vivado.

> **`WEIGHTS_DIR` y la GUI.** Este macro solo importa en la simulación C. Fuera
> de `__SYNTHESIS__`, hls4ml deja los arreglos de pesos sin inicializar y los
> carga de `firmware/weights/*.txt` en ejecución; bajo síntesis los valores están
> empotrados en las cabeceras y el macro se ignora. **No afecta al hardware.**
>
> Tiene que expandirse a un *literal de cadena*, comillas incluidas.
> `build_ip.tcl` pasa `-DWEIGHTS_DIR=\"...\"` y desde Tcl funciona, pero **la GUI
> de Vitis 2022.2 lo estropea**: su diálogo *Edit CFLAGS* vuelve a dividir el
> campo en tokens, elimina las comillas escapadas y reordena las opciones,
> dejando una ruta desnuda. El resultado es un error de compilación desconcertante
> dentro de `nnet_helpers.h`:
>
> ```
> error: expected primary-expression before '/' token
> note: in expansion of macro 'WEIGHTS_DIR'
> ```
>
> La solución no es escapar más. `regenerate_firmware.py` inserta la ruta
> absoluta directamente en el valor por defecto `#ifndef WEIGHTS_DIR` de
> `firmware/nnet_utils/nnet_helpers.h`, de modo que la GUI no necesita ningún
> `-D`. Si trabajas con un árbol de firmware generado de otra forma, aplícalo a
> mano:
>
> ```bash
> sed -i 's|#define WEIGHTS_DIR "weights"|#define WEIGHTS_DIR "/ruta/al/pynq_..._prj/firmware/weights"|' /ruta/al/pynq_..._prj/firmware/nnet_utils/nnet_helpers.h
> ```
>
> Sin una cosa u otra, csim termina con `ERROR: file w2.txt does not exist`.

---

## 5. Etapa B — Vivado 2022.2

**Por script** (desde `vivado/ids_overlay/`): `vivado -mode batch -source scripts/build_overlay.tcl`.
Deja `ids_overlay.bit` + `ids_overlay.hwh` en `embebido/ids_overlay/`.
(lee antes el aviso de la cabecera de ese archivo).

### B.1 · Crear el proyecto

*Create Project* → **RTL Project**, marcar *Do not specify sources at this time*
→ en la selección de destino, pestaña **Boards** → **PYNQ-Z2**.

Si el PYNQ-Z2 no aparece en la lista, los ficheros de placa no están instalados.
No es bloqueante: ve a la pestaña **Parts** y elige `xc7z020clg400-1`
directamente. Lo único que se pierde es el preajuste del PS, que habrá que
repasar a mano en B.4.

### B.2 · Añadir el repositorio de IP

*Settings* → **IP → Repository** → `+` → selecciona `vitis/ids_axi/ip`.

Debe aparecer «1 repository, 1 IP». Si no:

```tcl
update_ip_catalog -rebuild
get_ipdefs *ids*
```

### B.3 · Crear el block design

*IP Integrator* → **Create Block Design** → nombre `ids_bd`.

### B.4 · Processing System

1. `+` → **ZYNQ7 Processing System**.
2. Pulsa **Run Block Automation** → deja los valores por defecto → OK.
3. Doble clic en el bloque del PS:
   - **PS-PL Configuration → HP Slave AXI Interface** → activa **S AXI HP0**.
     Es el camino que el DMA necesita hacia la DDR; sin él, *Connection
     Automation* no tendrá dónde enganchar los maestros.
   - **Clock Configuration → PL Fabric Clocks** → confirma que **FCLK_CLK0** está
     a **100 MHz**. Es el reloj con el que se alimentará el acelerador, y tiene
     que corresponderse con los 10 ns con los que sintetizaste en A.4.

### B.5 · AXI DMA

1. `+` → **AXI Direct Memory Access**.
2. Doble clic:
   - desmarca **Enable Scatter Gather Engine** — el driver usa transferencias simples;
   - **Width of Buffer Length Register**: súbelo a **23** bits;
   - anchura de ambos streams: **32** bits;
   - deja activados los dos canales (MM2S y S2MM).

> **Por qué 23 bits.** Por defecto son 14, es decir 16.383 bytes por
> transferencia. Cada paquete son 18 palabras de 32 bits = 72 bytes, así que el
> valor por defecto topa en **227 paquetes por lote**. El registro `n_packets`
> admite hasta 65.535, que son 4,7 MB: con 23 bits (8 MB) entra holgado. Si lo
> dejas en 14 y envías un lote mayor, el DMA se cuelga sin mensaje de error.

### B.6 · Añadir el acelerador

`+` → busca `ids_axi` → añade **ids_axi_0**.

### B.7 · Conectar los streams a mano

*Connection Automation* no adivina la dirección de los streams. Traza estas dos
conexiones arrastrando pin a pin:

```
axi_dma_0/M_AXIS_MM2S   →   ids_axi_0/in_stream
ids_axi_0/out_stream    →   axi_dma_0/S_AXIS_S2MM
```

Si los nombres no coinciden, lístalos:

```tcl
get_bd_intf_pins ids_axi_0/*
```

### B.8 · Automatizar el resto

**Run Connection Automation** → *Select All* → OK.

Construye el interconexionado AXI-Lite hacia los dos esclavos (`ids_axi_0/s_axi_CTRL`
y `axi_dma_0/S_AXI_LITE`), enruta los maestros del DMA a `S_AXI_HP0` y reparte
relojes y resets. Al terminar no debe quedar ningún pin AXI suelto.

### B.9 · Sacar `led_o` al exterior

1. Clic derecho sobre el pin `led_o` de `ids_axi_0` → **Make External**.
2. Selecciona el puerto recién creado y, en *External Port Properties*,
   renómbralo a exactamente **`led_o`**.

Vivado lo llama `led_o_0` por defecto, y el XDC suministrado no lo encontraría.

### B.10 · Restricciones

*Add Sources* → **Add or create constraints** → `constraints/pynq_z2_ids_leds.xdc`.

Asigna los seis pines de LD4/LD5 y añade un `set_false_path` sobre `led_o[*]`:
son salidas visuales, no tiene sentido que participen en el análisis temporal.

### B.11 · Validar y envolver

1. **Validate Design** (`F6`). Debe terminar sin errores; los avisos sobre
   direcciones sin asignar se resuelven con *Assign Address*.
2. En la pestaña *Sources*, clic derecho sobre `ids_bd` → **Create HDL Wrapper**
   → *Let Vivado manage wrapper and auto-update*.

### B.12 · Síntesis, implementación y bitstream

**Generate Bitstream** encadena las tres etapas. Entre 10 y 25 minutos en el
Zynq-7020.

Al terminar, revisa:

- **Design Runs → impl_1 → WNS** (*worst negative slack*): tiene que ser **≥ 0**.
  Un WNS negativo significa que el diseño no cumple temporización, y aunque el
  bitstream se genere, el comportamiento en placa no está garantizado.
- **Report Utilization**: los porcentajes reales, ya con el DMA y el
  interconexionado incluidos, no solo el acelerador.

Si el WNS sale negativo, lo más eficaz es bajar `FCLK_CLK0` a 50 MHz en B.4 y
regenerar. Este sistema no está limitado por la latencia de inferencia sino por
el transporte Ethernet, así que la penalización real es despreciable.

---

## 6. Etapa C — Exportación a la FPGA

### C.1 · Localizar los dos archivos

PYNQ resuelve un overlay a partir de un `.bit` y un `.hwh` **que compartan el
nombre base**. El `.hwh` es la descripción del hardware: de ahí saca PYNQ el mapa
de registros y los nombres de los bloques.

```
<proyecto>.runs/impl_1/ids_bd_wrapper.bit
<proyecto>.gen/sources_1/bd/ids_bd/hw_handoff/ids_bd.hwh
```

Fíjate en que están en árboles distintos (`.runs` y `.gen`) y con nombres
distintos. Renombrar es obligatorio, no opcional:

```bash
cp <proyecto>.runs/impl_1/ids_bd_wrapper.bit                        ids_overlay.bit
cp <proyecto>.gen/sources_1/bd/ids_bd/hw_handoff/ids_bd.hwh         ids_overlay.hwh
```

### C.2 · Copiar a la placa

Usuario por defecto `xilinx` / `xilinx`, IP estática `192.168.2.99`:

```bash
cd embebido/ids_overlay
scp ids_overlay.bit ids_overlay.hwh ids_driver.py ids_server.py xilinx@192.168.2.99:~/ids/
```

### C.3 · Cargar el overlay y arrancar el servidor

Primero prueba sin `sudo`. En la imagen de la placa el usuario `xilinx` ya tiene
el virtualenv de PYNQ activo, así que es la invocación más simple y en muchas
imágenes basta:

```bash
ssh xilinx@192.168.2.99
cd ids && python3 ids_server.py --bitfile ids_overlay.bit
```

Eso falla con `Root permissions required.`: descargar el bitstream reprograma
el PL y necesita root. Pero **no vale `sudo python3` a secas, ni tampoco basta
con dar la ruta absoluta del intérprete**. La invocación correcta preserva el
entorno:

```bash
cd ids && sudo -E env PATH=$PATH python3 ids_server.py --bitfile ids_overlay.bit
```

> **Por qué así y no de otra forma.** `sudo` hace dos cosas que rompen PYNQ, y
> hay que deshacer las dos.
>
> 1. **Reescribe el `PATH`** desde su propio `secure_path`. Desde PYNQ 2.7 el
>    paquete `pynq` no está en los `dist-packages` del sistema, sino en un
>    virtualenv en `/usr/local/share/pynq-venv` que
>    `/etc/profile.d/pynq_venv.sh` activa para el usuario `xilinx`. Así que
>    `sudo python3` resuelve a `/usr/bin/python3`, que no tiene `pynq`. Síntoma:
>    `ModuleNotFoundError: No module named 'pynq'` **ejecutando ya en la placa**.
>
> 2. **Descarta el resto del entorno**, del que depende el sondeo de
>    dispositivos de PYNQ. Si solo arreglas lo primero —llamando al intérprete
>    del venv por ruta absoluta— el `import` pasa, pero la lista de dispositivos
>    sale vacía. Síntoma: `UserWarning: No devices found, is the XRT environment
>    sourced?` seguido de `No Devices Found`. El aviso engaña: es el mensaje
>    genérico de PYNQ para lista vacía, y en una Zynq-7020 XRT no interviene.
>
> `-E` conserva el entorno y `PATH=$PATH` reinyecta el `PATH` que `-E` no llega
> a salvar del `secure_path`. Comprobación de que el problema es el entorno y no
> el hardware:
>
> ```bash
> python3 -c 'import pynq; print(pynq.__file__)'                        # ruta del venv
> sudo python3 -c 'import pynq'                                         # ModuleNotFoundError
> sudo /usr/local/share/pynq-venv/bin/python3 -c 'from pynq import Device; print(Device.devices)'   # []
> sudo -E env PATH=$PATH python3 -c 'from pynq import Device; print(Device.devices)'                # [<EmbeddedDevice>]
> ```
>
> La alternativa limpia es tomar una shell de login de root, que ejecuta los
> scripts de `/etc/profile.d/` por sí misma:
>
> ```bash
> sudo -i
> cd '/home/xilinx/pynq/ids_test/Alvarez - TFG'
> python3 ids_server.py --bitfile ids_overlay.bit
> ```

La carga del overlay reprograma el PL. Si `Overlay()` lanza una excepción sobre
el `.hwh`, es que el par no comparte nombre base o el `.hwh` es de otra
compilación.

### C.4 · Lanzar tráfico desde el PC

```bash
cd embebido/ids_overlay        # las rutas por defecto a data/ son relativas a esta carpeta
python host_sender.py --host 192.168.2.99 --n 2000 --batch 64
```

Cada lote imprime su veredicto y los LEDs cambian de color al mismo ritmo. Si la
captura trae etiquetas, el script cierra con una matriz de confusión calculada
sobre inferencias **de la FPGA** — ése es el número que va al capítulo de
resultados, no el de la simulación.

Eso envía la mezcla natural de la captura. Para la **demostración en vivo** hay
un segundo modo que construye un flujo casi todo benigno e inyecta un ataque
cada N paquetes:

```bash
python host_sender.py --host 192.168.2.99 --n 500 --batch 1 --attack-every 50 --interval 0.02
```

| Opción | Para qué |
|---|---|
| `--attack-every N` | Un ataque cada N paquetes. `0` (defecto) desactiva el modo y vuelve al muestreo aleatorio. El periodo incluye la ráfaga, así que la densidad es exactamente `burst/N`. |
| `--attack-burst K` | K paquetes de ataque consecutivos por inyección. Debe ser menor que `--attack-every`. |
| `--attack-class SUBSTR` | Restringe los ataques inyectados a las etiquetas que contengan `SUBSTR`, p. ej. `SlowLoris`. Útil para reproducir en hardware las clases que `05c` identificó como las de más falsos negativos. |
| `--jitter F` | Aleatoriza el hueco con desviación `F × attack-every`. `0` = periodo exacto. |
| `--interval SEC` | Pausa entre lotes, para que el flujo se pueda seguir a ojo. |

> **`--batch` manda sobre lo que se ve en los LEDs.** El fabric los enclava por
> transferencia DMA, así que un lote está en rojo si **algún** paquete suyo fue
> marcado. Con `--batch 1` cada paquete tiene su propio estado y los LEDs
> alternan paquete a paquete — es lo que quieres para grabar la demo. Si
> `--batch` supera a `--attack-every`, todos los lotes contienen un ataque y los
> LEDs se quedan en rojo; el script lo avisa al arrancar.

El resumen del modo inyección puntúa la FPGA contra **lo que realmente se
envió**, no contra las etiquetas de la captura, y desglosa los fallos por clase:

```
 injected attacks : 30
   caught         : 27 (90.0% recall on what was injected)
   missed         : 3
     positions    : [79, 159, 239]
     Mirai-greeth_flood           1
 benign packets   : 270
   false alarms   : 0 (0.00%)
```

> Con `--interval` el throughput se calcula **solo sobre el tiempo en el cable**,
> excluyendo las pausas, y el script lo indica. Aun así, un `--batch 1` paga el
> coste de una transferencia DMA por paquete: las cifras de rendimiento del
> capítulo de resultados salen del modo sin pausas y con lotes grandes, no de la
> demo.

### C.5 · Comprobación de extremo a extremo

| Qué mirar | Resultado correcto |
|---|---|
| LEDs en reposo | apagados (`0x00`) tras una transferencia sin paquetes |
| lote solo benigno | verde (`0x12`) |
| lote con algún ataque | rojo (`0x24`) |
| exactitud del `host_sender` | cercana a 0,99 con el firmware regenerado |
| exactitud ≈ 0,52 | estás corriendo el firmware `<8,2>` — vuelve a A.1 |

---

## 7. Referencia de interfaces

### Mapa de registros AXI-Lite (`s_axi_CTRL`)

Los desplazamientos los asigna Vitis HLS; la lista autoritativa está en
`ids_axi_prj/solution1/impl/ip/drivers/ids_axi_v1_0/src/xids_axi_hw.h`.
`ids_driver.py` usa el `register_map` de PYNQ por nombre, así que sigue siendo
correcto pase lo que pase.

| Registro        | Significado                                             |
|-----------------|---------------------------------------------------------|
| `CTRL`          | `AP_START` / `AP_DONE` / `AP_IDLE` / `AP_READY`          |
| `n_packets`     | paquetes de esta transferencia, 1 … 65535                |
| `threshold_raw` | umbral de decisión, como bits crudos de `score_t`        |

El orden importa: los dos escalares se capturan en el flanco de `ap_start`, así
que hay que escribirlos **antes** de activar `AP_START`, y solo después lanzar
las transferencias del DMA.

### Formato del stream

| Sentido | Palabras | Contenido |
|---------|----------|-----------|
| entrada | 18 por paquete | `[feature_t::width-1 : 0]` = bits crudos de punto fijo |
| salida  | 1 por paquete  | `[15:0]` bits de puntuación, `[16]` veredicto (1 = ataque) |

`TLAST` se activa en la última palabra de salida de la transferencia.

### Códec de punto fijo

Del lado del host, no en hardware: `ap_fixed` usa `AP_WRAP` por defecto, así que
un valor atípico de +302 σ sin recortar envolvería a un negativo pequeño y
corrompería la inferencia en silencio, en lugar de quedarse pegado al techo del
rango.

```python
frac = W - I
q = round(x * 2**frac)
q = clip(q, -2**(W-1), 2**(W-1) - 1)     # saturar, nunca envolver
word = q & (2**W - 1)
```

### Codificación de los LEDs

`led_o[2:0]` → LD4, `led_o[5:3]` → LD5, cada uno `{R,G,B}` con el azul en el bit
menos significativo (el mismo convenio que la clase `RGBLED` de PYNQ).

| Código | Significado                                        |
|--------|----------------------------------------------------|
| `0x00` | reposo — ningún paquete en la última transferencia  |
| `0x12` | **verde** — todos los paquetes del lote benignos    |
| `0x24` | **rojo** — al menos un ataque en el lote            |

El LED se enclava por *transferencia*, no por paquete: rojo si **algún** paquete
del lote fue marcado. Con `n_packets = 1` degenera en un veredicto por paquete,
que es lo que usa la demostración en vivo.

---

## 8. Resolución de problemas

### Entorno (Debian 13)

| Síntoma | Causa |
|---|---|
| `extra characters after close-quote` | Comillas sin escapar en `-cflags` dentro de un script Tcl. |
| `cannot find crt1.o` / `cannot find -lpthread` | Falta `export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu`. |
| `unknown type [0x13] section '.relr.dyn'` | El `binutils 2.37` empaquetado es anterior a `SHT_RELR`. Sustituir `ld` por el del sistema (A.0). |

### Vitis HLS

| Síntoma | Causa |
|---|---|
| csim: `ERROR: file w2.txt does not exist` | `WEIGHTS_DIR` no apunta al directorio de pesos. Ejecuta `regenerate_firmware.py` o aplica el `sed` de A.6. |
| `expected primary-expression before '/' token` | `WEIGHTS_DIR` llegó sin comillas. Misma solución. |
| Error de enlazado por `main()` duplicado | Se añadió `myproject_test.cpp` además del banco de pruebas propio. |
| Tras regenerar, Vitis no encuentra `ids_axi.cpp` | Se copiaron las fuentes de `src/` dentro del proyecto hls4ml, y la regeneración borró el directorio entero. Añádelas desde `src/` (A.2). |
| `config_array_partition -maximum_size not supported` | Inocuo. Es una opción heredada de Vivado HLS que Vitis HLS eliminó; `build_ip.tcl` nunca la emite. |
| `led_o` aparece como `ap_vld` en el informe de interfaz | Falta el pragma `INTERFACE ap_none` sobre ese puerto. |

### Vivado

| Síntoma | Causa |
|---|---|
| `create_bd_cell` no encuentra el VLNV | Repositorio de IP no añadido, o falta `update_ip_catalog`. Comprueba con `get_ipdefs *ids*`. |
| Falla la conexión de `in_stream` | Vitis HLS pudo nombrar la interfaz de otra forma. Lista con `get_bd_intf_pins ids_axi_0/*`. |
| XDC: `no ports matched led_o[*]` | El puerto externo sigue llamándose `led_o_0`. Renómbralo en el BD (B.9). |
| WNS negativo tras implementación | Baja `FCLK_CLK0` a 50 MHz, o sube el `ReuseFactor` y repite desde A.1. |
| *Connection Automation* no ofrece destino para el DMA | `S_AXI_HP0` sin habilitar en el PS (B.4). |

### Placa

| Síntoma | Causa |
|---|---|
| `No module named 'pynq'` **ejecutando en la placa** | Lanzaste con `sudo python3`. `sudo` descarta el `PATH` del usuario y resuelve a `/usr/bin/python3`, que no ve el virtualenv de PYNQ. Usa `sudo -E env PATH=$PATH python3` (C.3). |
| `Root permissions required.` | Descargar el bitstream reprograma el PL. Hace falta root, con la invocación de C.3. |
| `No devices found, is the XRT environment sourced?` + `No Devices Found` | Engañoso: XRT no interviene en una Zynq-7020. `sudo` descartó el entorno del que depende el sondeo de dispositivos. Pasa si arreglaste solo el intérprete (ruta absoluta al venv) sin conservar el entorno. Usa `sudo -E env PATH=$PATH python3` (C.3). |
| `Overlay()` falla al leer el `.hwh` | El `.bit` y el `.hwh` no comparten nombre base, o son de compilaciones distintas. |
| `dma.recvchannel.wait()` se cuelga | `n_packets` no coincide con las palabras enviadas, así que `TLAST` nunca llega. Debe ser igual a `len(X)`, y el búfer de entrada debe contener exactamente `n_packets × 18` palabras. |
| Se cuelga solo con lotes grandes | *Width of Buffer Length Register* quedó en 14 bits (B.5): el tope son 227 paquetes. |
| Todos los paquetes puntúan igual | Overlay cargado pero con entradas recortadas: revisa el aviso de saturación de `predict()` y confirma que `feature_w`/`feature_i` coinciden con `defines.h`. |
| Los LEDs nunca se encienden | `led_o` sin conectar en el BD, o el XDC no se añadió a `constrs_1`. |

---

## 9. Qué está verificado y qué no

**Verificado por análisis.** Todas las construcciones `ap_fixed`/`ap_int`/
`nnet::array` que usa el envoltorio se contrastaron con las cabeceras `ap_types`
empaquetadas (`width`, `iwidth`, `value_type`, `size`, y los constructores y
operadores de asignación de `af_range_ref`/`ap_range_ref`). La tabla de exactitud
del §2 procede de una reimplementación en numpy de la aritmética de hls4ml —cada
valor intermedio convertido a su tipo `ap_[u]fixed` declarado, con el modo de
redondeo y desbordamiento correcto— ejecutada sobre 40.000 paquetes reservados.
Los rangos de activación y de acumulador se midieron sobre el modelo real, y
`regenerate_firmware.py` se ejecutó contra hls4ml 1.3.0 para confirmar los tipos
emitidos en `defines.h`.

**Verificado en el servidor.** Las cuatro correcciones de entorno de
[`entorno_vitis_debian13.md`](entorno_vitis_debian13.md):
ambos `.cpp` compilan y el enlazado de `csim.exe` se completa.

**No es sustituto del hardware.** La simulación modela rangos de valores y
desbordamiento, no las tablas de consulta de activación de hls4ml ni el RTL
exacto. Trata el 0,9933 como la cifra que hay que *confirmar* en cosimulación, no
como una medida de hardware.

**Sin verificar.** Estos archivos se escribieron en un entorno sin cadena de
herramientas de Xilinx y sin compilador de C++, de modo que la síntesis C,
`build_overlay.tcl` y el Python de placa no se han ejecutado. El resultado del
banco de pruebas de simulación C tampoco está confirmado todavía. Empieza por la
etapa A: es la comprobación más rápida y ejercita el envoltorio de extremo a
extremo.
