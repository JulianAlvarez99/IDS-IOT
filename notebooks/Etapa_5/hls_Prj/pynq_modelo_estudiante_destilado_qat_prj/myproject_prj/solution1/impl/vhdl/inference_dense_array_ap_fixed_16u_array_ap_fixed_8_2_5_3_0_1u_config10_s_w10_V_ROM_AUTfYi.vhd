-- ==============================================================
-- Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.2 (64-bit)
-- Version: 2022.2
-- Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
-- ==============================================================
library ieee; 
use ieee.std_logic_1164.all; 
use ieee.std_logic_unsigned.all;

entity inference_dense_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_1u_config10_s_w10_V_ROM_AUTfYi is 
    generic(
             DataWidth     : integer := 6; 
             AddressWidth     : integer := 4; 
             AddressRange    : integer := 16
    ); 
    port (
 
          address0        : in std_logic_vector(AddressWidth-1 downto 0); 
          ce0             : in std_logic; 
          q0              : out std_logic_vector(DataWidth-1 downto 0);

          reset               : in std_logic;
          clk                 : in std_logic
    ); 
end entity; 


architecture rtl of inference_dense_array_ap_fixed_16u_array_ap_fixed_8_2_5_3_0_1u_config10_s_w10_V_ROM_AUTfYi is 
 
signal address0_tmp : std_logic_vector(AddressWidth-1 downto 0); 

type mem_array is array (0 to AddressRange-1) of std_logic_vector (DataWidth-1 downto 0); 

signal mem0 : mem_array := (
    0 => "001011", 1 => "000100", 2 => "110001", 3 => "000100", 
    4 => "000011", 5 => "000110", 6 => "000111", 7 => "111010", 
    8 => "110110", 9 => "010001", 10 => "010010", 11 => "111101", 
    12 => "110010", 13 => "111100", 14 => "101101", 15 => "111010");



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

