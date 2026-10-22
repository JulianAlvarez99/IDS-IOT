# Puesta en marcha de Vitis HLS 2022.2 sobre Debian 13

Registro de las correcciones aplicadas para que la simulación C del IP core
`ids_axi` compile y enlace en el servidor de laboratorio.

| | |
|---|---|
| **Servidor** | `seaman-fitba2` — Debian GNU/Linux 13 (trixie), kernel 6.12.95 |
| **Herramientas** | Vitis HLS 2022.2 y Vivado 2022.2 en `/home/fitba/tools/Xilinx` |
| **Proyecto** | `/mnt/Datos/Julian_Alvarez-TFG/IDS-IOT` |
| **Fecha** | Agosto 2026 |

---

## 1. Por qué aparecen estos fallos

Los cuatro problemas se encadenaron: cada corrección destapó el siguiente error.
Se dividen en dos familias con causas distintas.

**Fallos 1 y 2 — flujo de compilación.** El macro `WEIGHTS_DIR` que genera
hls4ml tiene que sobrevivir a dos capas de *parsing* (Tcl y el preprocesador de
C) antes de llegar al compilador. Ni el script Tcl original ni el diálogo de la
GUI lo transportaban intacto.

**Fallos 3 y 4 — incompatibilidad de distribución.** Vitis 2022.2 se publicó en
octubre de 2022; Debian 13 «trixie» es de 2025. Xilinx solo da soporte oficial a
RHEL, CentOS, Ubuntu LTS y SUSE. Las herramientas traen su propia copia de
`binutils 2.37`, que es anterior a cambios en el formato ELF que hoy usa glibc.

> **Ningún fallo está en el código HLS del proyecto.** Ninguna corrección
> modifica `ids_axi.cpp`, `ids_axi.h` ni la red neuronal. Todas son reversibles y
> ninguna necesita `root`.

### Cadena de fallos

| # | Síntoma | Capa | Estado |
|---|---|---|---|
| 1 | `extra characters after close-quote` | Tcl | Resuelto |
| 2 | `expected primary-expression before '/' token` | Preprocesador C | Resuelto |
| 3 | `cannot find crt1.o` / `cannot find -lpthread` | Enlazador (rutas) | Resuelto |
| 4 | `unknown type [0x13] section '.relr.dyn'` | Enlazador (formato ELF) | Resuelto |

---

## 2. Fallo 1 — Comillas sin escapar en `-cflags`

### Síntoma

```
extra characters after close-quote
    while executing
"add_files .../firmware/ids_axi.cpp -cflags "-std=c++14 -I../../src -I<abs path>/firmwar..."
    (file ".../ids_axi_prj/solution1/csim.tcl" line 8)
```

### Diagnóstico

El script abortó en la línea 8. **No se compiló ni un solo archivo.** El mensaje
final `Finished C simulation.` que imprime Vitis es incondicional: aparece haya
o no haya simulación, así que no sirve como confirmación de éxito.

### Causa

Tcl cierra una palabra entrecomillada en la primera `"` sin escapar. Al escribir:

```tcl
-cflags "-std=c++14 ... -DWEIGHTS_DIR="/ruta/weights""
```

Tcl interpreta `"-std=c++14 ... -DWEIGHTS_DIR="` como la cadena completa y se
encuentra con `/ruta/weights` suelto a continuación — de ahí *«caracteres extra
después de la comilla de cierre»*.

Además, la ruta contenía el literal `<abs path>`, un marcador de posición de la
documentación que nunca se sustituyó por la ruta real.

### Corrección

En `scripts/build_ip.tcl`, las comillas interiores se escapan para que
sobrevivan al *parsing* de Tcl y lleguen a GCC:

```tcl
set wdir   "\\\"${firmware}/weights\\\""
set cflags "-std=c++14 -I${src_dir} -I${firmware} -DWEIGHTS_DIR=${wdir}"
```

`\\` produce una barra invertida literal y `\"` una comilla literal, de modo que
el valor final que recibe el compilador es `\"/ruta/weights\"`.

**Esto funciona desde un script Tcl, pero no desde la GUI** — ver el fallo 2.

---

## 3. Fallo 2 — `WEIGHTS_DIR` no llega como literal de cadena

### Síntoma

```
nnet_helpers.h:24:40: error: expected primary-expression before '(' token
     std::string full_path = std::string(WEIGHTS_DIR) + "/" + std::string(fname);
<command-line>: error: expected primary-expression before '/' token
<command-line>: error: expected primary-expression before '.' token
note: in expansion of macro 'WEIGHTS_DIR'
```

### Diagnóstico

`<command-line>` en el mensaje indica que el problema está en el macro definido
en la línea de órdenes, no en el archivo de cabecera. Los tokens `.` y `/` que
GCC no sabe interpretar son los de la propia ruta: se está expandiendo a
`std::string(../../../../notebooks/...)`, es decir, **sin comillas**.

El registro de Vitis muestra exactamente qué emitió la GUI:

```
-cflags -DWEIGHTS_DIR=/ -I/mnt/Datos/.../firmware -std=c++14
```

El diálogo *Edit CFLAGS* de Vitis 2022.2 volvió a dividir el campo en tokens,
**reordenó las opciones, eliminó las comillas escapadas** y dejó el valor
reducido a `/`. Ninguna cantidad de escapado sobrevive a ese viaje de ida y
vuelta.

### Por qué existe el macro

En `firmware/weights/w2.h`, hls4ml genera los pesos con dos ramas:

```c
#ifndef __SYNTHESIS__
weight2_t w2[576];                      // sin inicializar → se cargan en ejecución
#else
weight2_t w2[576] = {0.28125, ...};     // empotrados en el binario
#endif
```

Durante la **simulación C** los arreglos se rellenan leyendo
`firmware/weights/*.txt`, y el directorio sale del macro `WEIGHTS_DIR`. Durante
la **síntesis** los valores ya están en las cabeceras y el macro se ignora.

Consecuencia importante: **este macro no afecta en absoluto al hardware
generado.** Solo interviene en `csim` y `cosim`.

El valor por defecto, en `firmware/nnet_utils/nnet_helpers.h`, es una ruta
relativa:

```c
#ifndef WEIGHTS_DIR
#define WEIGHTS_DIR "weights"
#endif
```

Relativa no sirve: el ejecutable de simulación se lanza desde
`ids_axi_prj/solution1/csim/build`, cuatro niveles por debajo de la raíz del
proyecto, donde no existe ninguna carpeta `weights`.

### Corrección

En lugar de pelear con el escapado, se reescribe el valor por defecto de la
propia cabecera. Así la GUI no necesita ninguna opción `-D`.

```bash
sed -i 's|#define WEIGHTS_DIR "weights"|#define WEIGHTS_DIR "/mnt/Datos/Julian_Alvarez-TFG/IDS-IOT/vitis/hls4ml/pynq_modelo_estudiante_destilado_qat_prj/firmware/weights"|' /mnt/Datos/Julian_Alvarez-TFG/IDS-IOT/vitis/hls4ml/pynq_modelo_estudiante_destilado_qat_prj/firmware/nnet_utils/nnet_helpers.h
```

Desglose de la orden:

| Parte | Función |
|---|---|
| `sed` | editor de flujo de texto |
| `-i` | edita el archivo **in situ**, sin copia de seguridad (`-i.bak` la crearía) |
| `s|VIEJO|NUEVO|` | sustituye la primera cadena por la segunda |
| `|` como separador | lo habitual es `s/.../.../`, pero las rutas están llenas de `/`; usar `|` evita escapar cada una |
| último argumento | archivo sobre el que se opera |

La orden es **idempotente**: al ejecutarla por segunda vez ya no encuentra el
texto `"weights"` y no cambia nada.

La modificación queda dentro del bloque `#ifndef __SYNTHESIS__`, por lo que la
síntesis, el *timing* y el IP exportado no se ven afectados.

### CFLAGS resultantes

Con la cabecera corregida, las opciones se reducen a esto — sin comillas, sin
nada que la GUI pueda estropear. Se aplican a los tres archivos
(`ids_axi.cpp`, `myproject.cpp` y el banco de pruebas):

```
-std=c++14 -I/mnt/Datos/Julian_Alvarez-TFG/IDS-IOT/vitis/hls4ml/pynq_modelo_estudiante_destilado_qat_prj/firmware
```

Las rutas de `-I` deben ser **absolutas**: la GUI resuelve las relativas contra
`solution1/csim/build`.

### Automatización

hls4ml vuelve a copiar el árbol `nnet_utils/` en cada regeneración, así que la
corrección manual se perdería. `scripts/regenerate_firmware.py` la aplica ahora
de forma automática mediante la función `patch_weights_dir()`, invocada
inmediatamente después de `hls_model.write()`.

### Verificación

```bash
grep -n WEIGHTS_DIR .../firmware/nnet_utils/nnet_helpers.h
```

---

## 4. Fallo 3 — `crt1.o`, `crti.o`, `-lpthread` y `-lm` no encontrados

### Síntoma

Los dos archivos fuente **compilaron correctamente**; el fallo apareció al
enlazar.

```
   Generating csim.exe
.../binutils-2.37/bin/ld: cannot find crt1.o: No such file or directory
.../binutils-2.37/bin/ld: cannot find crti.o: No such file or directory
.../binutils-2.37/bin/ld: cannot find -lpthread
.../binutils-2.37/bin/ld: cannot find -lm
collect2: error: ld returned 1 exit status
```

`crt1.o` y `crti.o` son los objetos de arranque del runtime de C de glibc.

### Diagnóstico

Primero hay que distinguir entre «el paquete de desarrollo no está instalado» y
«el enlazador no sabe dónde buscar»:

```bash
ls -l /usr/lib/x86_64-linux-gnu/crt1.o /usr/lib/x86_64-linux-gnu/crti.o \
      /usr/lib/x86_64-linux-gnu/libm.so /usr/lib/x86_64-linux-gnu/libpthread.so
```

Resultado obtenido:

```
ls: no se puede acceder a '.../libpthread.so': No existe el fichero o el directorio
-rw-r--r-- 1 root root 1776 abr 27 17:09 /usr/lib/x86_64-linux-gnu/crt1.o
-rw-r--r-- 1 root root 1072 abr 27 17:09 /usr/lib/x86_64-linux-gnu/crti.o
-rw-r--r-- 1 root root  140 abr 27 17:09 /usr/lib/x86_64-linux-gnu/libm.so
```

Tres de los cuatro **sí existen** → `libc6-dev` está instalado y el problema es
de rutas de búsqueda, no de paquetes.

`libpthread.so` falta por un motivo distinto y esperado: desde **glibc 2.34**
libpthread está fusionada dentro de libc, y Debian dejó de distribuir el enlace
simbólico de desarrollo. Comprobación:

```bash
ls -l /usr/lib/x86_64-linux-gnu/libpthread.*
```

```
-rw-r--r-- 1 root root     8 abr 27 17:09 libpthread.a
-rw-r--r-- 1 root root 14408 abr 27 17:09 libpthread.so.0
```

`libpthread.a` mide **8 bytes**, exactamente el tamaño de la cabecera mágica de
un archivo `ar` vacío (`!<arch>\n`). glibc lo distribuye a propósito para que
`-lpthread` siga resolviendo sin aportar código, ya que los símbolos de hilos
viven ahora en `libc.so.6`. GNU ld busca primero `libpthread.so`, no lo
encuentra, y recurre a `libpthread.a`.

### Corrección

```bash
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu
vitis_hls
```

`LIBRARY_PATH` es la variable que GCC consulta **en tiempo de enlazado** para
localizar `crt*.o` y las bibliotecas `-l`.

> **No confundir con `LD_LIBRARY_PATH`.** Esa otra variable la usa el cargador
> dinámico *en tiempo de ejecución* y no habría tenido ningún efecto aquí.

La GUI hereda el entorno del proceso que la lanza, de modo que Vitis debe
arrancarse **desde el mismo terminal** donde se hizo el `export`. Un lanzador de
escritorio no lo recoge.

### Verificación

Confirmado: los cuatro errores de «no encontrado» desaparecieron y el enlazado
avanzó hasta el fallo 4.

---

## 5. Fallo 4 — Sección `.relr.dyn` desconocida

### Síntoma

```
.../binutils-2.37/bin/ld: /lib/x86_64-linux-gnu/libm.so.6: unknown type [0x13] section `.relr.dyn'
.../binutils-2.37/bin/ld: skipping incompatible /lib/x86_64-linux-gnu/libm.so.6 when searching for /lib/x86_64-linux-gnu/libm.so.6
.../binutils-2.37/bin/ld: cannot find /lib/x86_64-linux-gnu/libm.so.6
```

### Diagnóstico

El mensaje se contradice a sí mismo, y esa es justamente la pista:
*«omitiendo el incompatible `libm.so.6` mientras buscaba `libm.so.6`»*.
Encontró el archivo exacto y acto seguido lo rechazó. **No es un problema de
rutas** — el `export LIBRARY_PATH` del paso anterior era correcto y debe
mantenerse — sino de capacidad del enlazador.

### Causa

`.relr.dyn` es una sección de tipo `SHT_RELR` (`0x13` = 19 decimal), un formato
compacto de reubicaciones relativas incorporado a ELF recientemente y adoptado
ya por glibc.

El `binutils 2.37` que empaqueta Vivado 2022.2 es de mediados de 2021 y es
anterior a ese soporte (llegó en la 2.38 y posteriores). Debian 13 distribuye
binutils 2.44. El enlazador de Xilinx, sencillamente, no sabe leer las
bibliotecas del sistema.

### Corrección

Comprobar la versión del enlazador del sistema (se necesita 2.38 o superior):

```bash
/usr/bin/ld --version | head -1
```

Sustituir el enlazador empaquetado por un enlace simbólico al del sistema:

```bash
cd /home/fitba/tools/Xilinx/Vivado/2022.2/tps/lnx64/binutils-2.37/bin && mv ld ld.xilinx-backup && ln -s /usr/bin/ld ld
```

Las herramientas están bajo `/home/fitba`, así que no hace falta `root`.

**Por qué un enlace simbólico y no simplemente renombrar:** GCC se invoca con
`-B` apuntando a ese directorio, por lo que busca un ejecutable llamado `ld`
*ahí concretamente*. Borrarlo funciona solo si la búsqueda de respaldo llega
hasta `/usr/bin`; el enlace simbólico lo garantiza.

Para revertir:

```bash
cd /home/fitba/tools/Xilinx/Vivado/2022.2/tps/lnx64/binutils-2.37/bin && rm ld && mv ld.xilinx-backup ld
```

### Verificación

Confirmado: tras sustituir el enlazador, el error `.relr.dyn` desaparece y el
enlazado de `csim.exe` se completa. Es la última de las cuatro correcciones de
la cadena.

### Alcance del cambio

- Afecta a **todos** los proyectos HLS de esa máquina que usen `csim` o `cosim`.
  No es una pérdida: el enlazador empaquetado ya era inservible en trixie.
- La síntesis, el emplazado y rutado y la generación del *bitstream* **no** lo
  utilizan.
- La fase de compilación ya funcionaba, luego `as`, `ar` y el resto de
  `binutils` empaquetado están bien. El único componente obsoleto es `ld`.

---

## 6. Configuración persistente

`export` desaparece al cerrar el terminal. Para no alterar el entorno de
enlazado de otras compilaciones del servidor, conviene un lanzador dedicado en
lugar de tocar `~/.bashrc`:

```bash
printf '#!/bin/bash\nexport LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu\nexec vitis_hls "$@"\n' > ~/bin/vitis_hls_fixed && chmod +x ~/bin/vitis_hls_fixed
```

La sustitución de `ld` es permanente por sí misma y no requiere nada más.

---

## 7. Estado

| Corrección | Estado | Evidencia |
|---|---|---|
| Escapado de comillas en `build_ip.tcl` | Verificado | El script llega a `csim_design` |
| `WEIGHTS_DIR` en `nnet_helpers.h` | Verificado | Ambos `.cpp` compilan sin errores |
| Automatización en `regenerate_firmware.py` | Verificado | Probado sobre copia, idempotente |
| `export LIBRARY_PATH` | Verificado | Desaparecen los errores de `crt1.o` y `-lpthread` |
| Sustitución de `ld` por el del sistema | Verificado | Desaparece `.relr.dyn`; el enlazado se completa |

### Qué queda por hacer

1. Confirmar el resultado del banco de pruebas: se espera
   `PASS - all structural checks succeeded`.
2. Ejecutar la síntesis C (no depende de ninguna de estas correcciones, porque
   no enlaza ningún ejecutable) y revisar la tabla de interfaces y el *timing*.
3. Ejecutar la co-simulación, que sí necesita el enlazador corregido y es la
   fase que confirmaría la exactitud numérica del modelo en RTL.

> La simulación C solo comprueba la estructura de la interfaz AXI: un resultado
> por paquete, `TLAST` únicamente en la última palabra, `led_o` coherente con la
> reducción OR del lote y el caso `n_packets == 0`. **No verifica la exactitud
> de la red.** Eso corresponde a la co-simulación.

---

## 8. Resumen de órdenes

```bash
# Diagnóstico
ls -l /usr/lib/x86_64-linux-gnu/crt1.o /usr/lib/x86_64-linux-gnu/crti.o \
      /usr/lib/x86_64-linux-gnu/libm.so /usr/lib/x86_64-linux-gnu/libpthread.so
ls -l /usr/lib/x86_64-linux-gnu/libpthread.*
cat /usr/lib/x86_64-linux-gnu/libm.so
/usr/bin/ld --version | head -1

# Fallo 2 — ruta de pesos absoluta en la cabecera
sed -i 's|#define WEIGHTS_DIR "weights"|#define WEIGHTS_DIR "/mnt/Datos/Julian_Alvarez-TFG/IDS-IOT/vitis/hls4ml/pynq_modelo_estudiante_destilado_qat_prj/firmware/weights"|' \
    /mnt/Datos/Julian_Alvarez-TFG/IDS-IOT/vitis/hls4ml/pynq_modelo_estudiante_destilado_qat_prj/firmware/nnet_utils/nnet_helpers.h

# Fallo 3 — rutas de búsqueda del enlazador
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu

# Fallo 4 — enlazador del sistema en lugar del empaquetado
cd /home/fitba/tools/Xilinx/Vivado/2022.2/tps/lnx64/binutils-2.37/bin && mv ld ld.xilinx-backup && ln -s /usr/bin/ld ld

# Reversión del fallo 4
cd /home/fitba/tools/Xilinx/Vivado/2022.2/tps/lnx64/binutils-2.37/bin && rm ld && mv ld.xilinx-backup ld
```
