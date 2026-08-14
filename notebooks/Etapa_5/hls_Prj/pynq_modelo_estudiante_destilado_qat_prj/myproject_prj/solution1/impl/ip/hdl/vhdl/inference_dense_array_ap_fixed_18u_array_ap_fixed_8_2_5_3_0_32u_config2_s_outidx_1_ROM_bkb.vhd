-- ==============================================================
-- Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.2 (64-bit)
-- Version: 2022.2
-- Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
-- ==============================================================
library ieee; 
use ieee.std_logic_1164.all; 
use ieee.std_logic_unsigned.all;

entity inference_dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_s_outidx_1_ROM_bkb is 
    generic(
             DataWidth     : integer := 2; 
             AddressWidth     : integer := 7; 
             AddressRange    : integer := 72
    ); 
    port (
 
          address0        : in std_logic_vector(AddressWidth-1 downto 0); 
          ce0             : in std_logic; 
          q0              : out std_logic_vector(DataWidth-1 downto 0);

          reset               : in std_logic;
          clk                 : in std_logic
    ); 
end entity; 


architecture rtl of inference_dense_array_ap_fixed_18u_array_ap_fixed_8_2_5_3_0_32u_config2_s_outidx_1_ROM_bkb is 
 
signal address0_tmp : std_logic_vector(AddressWidth-1 downto 0); 

type mem_array is array (0 to AddressRange-1) of std_logic_vector (DataWidth-1 downto 0); 

signal mem0 : mem_array := (
    0 => "00", 1 => "00", 2 => "00", 3 => "00", 
    4 => "00", 5 => "00", 6 => "00", 7 => "00", 
    8 => "00", 9 => "00", 10 => "00", 11 => "00", 
    12 => "00", 13 => "00", 14 => "00", 15 => "00", 
    16 => "00", 17 => "00", 18 => "01", 19 => "01", 
    20 => "01", 21 => "01", 22 => "01", 23 => "01", 
    24 => "01", 25 => "01", 26 => "01", 27 => "01", 
    28 => "01", 29 => "01", 30 => "01", 31 => "01", 
    32 => "01", 33 => "01", 34 => "01", 35 => "01", 
    36 => "10", 37 => "10", 38 => "10", 39 => "10", 
    40 => "10", 41 => "10", 42 => "10", 43 => "10", 
    44 => "10", 45 => "10", 46 => "10", 47 => "10", 
    48 => "10", 49 => "10", 50 => "10", 51 => "10", 
    52 => "10", 53 => "10", 54 => "11", 55 => "11", 
    56 => "11", 57 => "11", 58 => "11", 59 => "11", 
    60 => "11", 61 => "11", 62 => "11", 63 => "11", 
    64 => "11", 65 => "11", 66 => "11", 67 => "11", 
    68 => "11", 69 => "11", 70 => "11", 71 => "11");



begin 

 
memory_access_guard_0: process (address0) 
begin
      address0_tmp <= address0;
--synthesis translate_off
      if (CONV_INTEGER(address0) > AddressRange-1) then
           address0_tmp <= (others => '0');
      else 
           address0_tmp <= address0;
      end if;
--synthesis translate_on
end process;

p_rom_access: process (clk)  
begin 
    if (clk'event and clk = '1') then
 
        if (ce0 = '1') then  
            q0 <= mem0(CONV_INTEGER(address0_tmp)); 
        end if;

end if;
end process;

end rtl;

