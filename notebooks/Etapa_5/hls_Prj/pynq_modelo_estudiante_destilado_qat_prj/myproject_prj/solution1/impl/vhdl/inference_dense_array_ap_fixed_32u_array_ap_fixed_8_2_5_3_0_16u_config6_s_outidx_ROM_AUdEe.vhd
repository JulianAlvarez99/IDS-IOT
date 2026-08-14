-- ==============================================================
-- Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.2 (64-bit)
-- Version: 2022.2
-- Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
-- ==============================================================
library ieee; 
use ieee.std_logic_1164.all; 
use ieee.std_logic_unsigned.all;

entity inference_dense_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_16u_config6_s_outidx_ROM_AUdEe is 
    generic(
             DataWidth     : integer := 1; 
             AddressWidth     : integer := 6; 
             AddressRange    : integer := 64
    ); 
    port (
 
          address0        : in std_logic_vector(AddressWidth-1 downto 0); 
          ce0             : in std_logic; 
          q0              : out std_logic_vector(DataWidth-1 downto 0);

          reset               : in std_logic;
          clk                 : in std_logic
    ); 
end entity; 


architecture rtl of inference_dense_array_ap_fixed_32u_array_ap_fixed_8_2_5_3_0_16u_config6_s_outidx_ROM_AUdEe is 
 
signal address0_tmp : std_logic_vector(AddressWidth-1 downto 0); 

type mem_array is array (0 to AddressRange-1) of std_logic_vector (DataWidth-1 downto 0); 

signal mem0 : mem_array := (
    0 => "0", 1 => "0", 2 => "0", 3 => "0", 
    4 => "0", 5 => "0", 6 => "0", 7 => "0", 
    8 => "0", 9 => "0", 10 => "0", 11 => "0", 
    12 => "0", 13 => "0", 14 => "0", 15 => "0", 
    16 => "0", 17 => "0", 18 => "0", 19 => "0", 
    20 => "0", 21 => "0", 22 => "0", 23 => "0", 
    24 => "0", 25 => "0", 26 => "0", 27 => "0", 
    28 => "0", 29 => "0", 30 => "0", 31 => "0", 
    32 => "1", 33 => "1", 34 => "1", 35 => "1", 
    36 => "1", 37 => "1", 38 => "1", 39 => "1", 
    40 => "1", 41 => "1", 42 => "1", 43 => "1", 
    44 => "1", 45 => "1", 46 => "1", 47 => "1", 
    48 => "1", 49 => "1", 50 => "1", 51 => "1", 
    52 => "1", 53 => "1", 54 => "1", 55 => "1", 
    56 => "1", 57 => "1", 58 => "1", 59 => "1", 
    60 => "1", 61 => "1", 62 => "1", 63 => "1");



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

