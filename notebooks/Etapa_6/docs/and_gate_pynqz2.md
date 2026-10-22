# Compuerta AND en la PL de la PYNQ-Z2

Prueba de hardware **puramente FPGA**: los switches entran por pines de la Programmable
Logic, la AND se resuelve en compuertas sintetizadas y el resultado sale por un LED.
El ARM (PS) no participa del cálculo; solo descarga el bitstream desde Jupyter.

Es el mismo flujo de trabajo que se usa para embeber el acelerador de hls4ml
(Etapa 5), pero con un diseño trivial para validar la cadena completa
`Vivado -> .bit + .hwh -> PYNQ Overlay`.

| Elemento | Pin | Función |
|---|---|---|
| SW0 | M20 | entrada A |
| SW1 | M19 | entrada B |
| LD0 | R14 | **A AND B** |
| LD1 | P14 | eco de A |
| LD2 | N16 | eco de B |
| LD3 | M14 | sin usar |

## Contenido

El diseño vive en [`vivado/and_gate_pynqz2/`](../../../vivado/and_gate_pynqz2/); todas
las rutas y comandos de esta guía son relativos a esa carpeta. El `.bit` + `.hwh` ya
compilados están en [`embebido/and_gate/`](../../../embebido/and_gate/).

```
src/and_gate.v                    módulo RTL combinacional
constraints/pynq_z2_and_gate.xdc  asignación de pines (subset del master XDC de TUL)
build_and_gate.tcl                build no interactivo de punta a punta
build/deploy/                     salida: and_gate.bit + and_gate.hwh
and_gate_prk/                     proyecto Vivado creado desde la GUI (camino manual)
```

## Camino rápido (Tcl)

```
vivado -mode batch -source build_and_gate.tcl
```

Deja `build/deploy/and_gate.bit` y `build/deploy/and_gate.hwh`. Saltar a la sección
"Embebido en Jupyter".

## Camino manual (GUI de Vivado)

1. **Create Project** -> RTL Project, *Do not specify sources at this time* desmarcado.
   En *Default Part*, pestaña **Boards** elegir `PYNQ-Z2` si tenés los board files
   instalados; si no, pestaña **Parts** -> `xc7z020clg400-1`.
2. **Add Sources** -> `src/and_gate.v`.
   **Add Constraints** -> `constraints/pynq_z2_and_gate.xdc`.
3. **IP INTEGRATOR -> Create Block Design**, nombre `and_gate_bd`.
   El block design es obligatorio: el `.hwh` que necesita PYNQ solo se genera a
   partir de un BD, no de un proyecto RTL puro.
4. `+` -> **ZYNQ7 Processing System** -> **Run Block Automation** (dejar el preset de
   la placa). Doble clic en el PS -> *PS-PL Configuration* -> **AXI Non Secure
   Enablement -> M AXI GP0 interface: desmarcar**. No hay AXI en esta prueba y así
   no queda una interfaz colgando.
5. Clic derecho en el canvas -> **Add Module...** -> `and_gate`. Aparece una caja con
   los puertos `sw[1:0]` y `led[3:0]`.
6. Clic derecho en cada puerto -> **Make External** (`Ctrl+T`). Seleccionar cada puerto
   externo creado y en **External Port Properties** renombrarlo a `sw` y `led`
   (Vivado los llama `sw_0` / `led_0` por defecto y entonces no coinciden con el XDC).
7. **Validate Design** (`F6`) -> **Save**.
8. En *Sources*, clic derecho sobre `and_gate_bd.bd` -> **Create HDL Wrapper** ->
   *Let Vivado manage wrapper*. Queda `and_gate_bd_wrapper` como top.
9. **Generate Bitstream**. Acepta lanzar síntesis e implementación.

### Dónde quedan los archivos

Vivado 2020.2 y posteriores:

```
<proyecto>.runs/impl_1/and_gate_bd_wrapper.bit
<proyecto>.gen/sources_1/bd/and_gate_bd/hw_handoff/and_gate_bd.hwh
```

En versiones anteriores el `.hwh` está bajo `<proyecto>.srcs/` en vez de `<proyecto>.gen/`.
También se llega por **File -> Export -> Export Block Design** / **Export Hardware**.

## Empaquetado

PYNQ exige que el bitstream y el hardware handoff **compartan el mismo nombre base**
y vivan en la misma carpeta:

```
and_gate.bit
and_gate.hwh
```

Renombrá ambos a `and_gate.*` (el script Tcl ya lo hace en `build/deploy/`).

## Copiar a la placa

Desde la PC, con la placa en red (por USB-Ethernet suele ser `192.168.2.99`):

```
ssh xilinx@192.168.2.99 "mkdir -p /home/xilinx/pynq/overlays/and_gate"
scp build/deploy/and_gate.bit build/deploy/and_gate.hwh xilinx@192.168.2.99:/home/xilinx/pynq/overlays/and_gate/
```

Usuario y contraseña por defecto: `xilinx` / `xilinx`. Alternativa sin consola: el
botón **Upload** de Jupyter, arrastrando los dos archivos.

## Embebido en Jupyter

```python
from pynq import Overlay

ol = Overlay("/home/xilinx/pynq/overlays/and_gate/and_gate.bit")  # descarga a la PL
print("bitstream:", ol.bitfile_name)
print("IPs en el overlay:", list(ol.ip_dict))   # {} -> no hay AXI, es PL pura
```

Al construir el `Overlay` el bitstream ya baja a la PL. A partir de ese momento
LD0 responde a los switches en hardware: mové SW0 y SW1 y mirá la tabla de verdad,
sin que Python intervenga. Podés incluso cerrar el notebook y la AND sigue andando.

Para volver al overlay del notebook de Etapa 6:

```python
from pynq.overlays.base import BaseOverlay
base = BaseOverlay("base.bit")
```

> Ojo: mientras esté cargado `and_gate.bit` no existen los AXI GPIO de `base`, así que
> `base.leds` / `base.switches` del notebook anterior no funcionan. Son overlays
> mutuamente excluyentes.

## Variante: leer el resultado desde Python

Si además querés inspeccionar la AND desde el notebook (y ese es el puente hacia el
flujo de hls4ml), agregá al block design un **AXI GPIO** en modo dual:
`GPIO` de 2 bits como entrada conectado a `sw`, `GPIO2` de 1 bit como entrada conectado
a `led[0]`. Eso requiere conectar `FCLK_CLK0` / `FCLK_RESET0_N` del PS y correr
*Run Connection Automation* para el `M_AXI_GP0` (volver a habilitarlo en el paso 4).
Después:

```python
gpio = ol.axi_gpio_0
print("sw  =", gpio.read(0))   # canal 1
print("and =", gpio.read(8))   # canal 2, offset 0x8
```

## Problemas frecuentes

- **`Overlay` levanta un error de parsing** -> falta el `.hwh`, o no tiene el mismo
  nombre base que el `.bit`, o el block design no contiene el `ZYNQ7 Processing System`.
- **`[Place 30-574] Poor placement` / DRCs de banco** -> falta `CFGBVS`/`CONFIG_VOLTAGE`;
  ya están en el XDC de este repo.
- **Los puertos del XDC no existen** -> los puertos externos del BD quedaron como
  `sw_0` / `led_0`. Renombralos (paso 6) o ajustá el XDC.
- **La placa no arranca después de cargar el overlay** -> no puede pasar con un
  bitstream solo-PL; el PS ya está configurado por el boot y no se toca.
