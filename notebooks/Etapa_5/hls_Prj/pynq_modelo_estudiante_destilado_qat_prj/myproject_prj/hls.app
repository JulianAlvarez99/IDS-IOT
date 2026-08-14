<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="com.autoesl.autopilot.project" name="myproject_prj" top="inference">
  <files>
    <file name="../../tb_data" sc="0" tb="1" cflags=" -Wno-unknown-pragmas -Wno-unknown-pragmas -Wno-unknown-pragmas" blackbox="false" csimflags=" -Wno-unknown-pragmas"/>
    <file name="../../firmware/weights" sc="0" tb="1" cflags=" -Wno-unknown-pragmas -Wno-unknown-pragmas -Wno-unknown-pragmas" blackbox="false" csimflags=" -Wno-unknown-pragmas"/>
    <file name="../../myproject_test.cpp" sc="0" tb="1" cflags=" -std=c++14 -Wl,-z,norelro -Wno-unknown-pragmas -Wno-unknown-pragmas" blackbox="false" csimflags=" -Wno-unknown-pragmas"/>
    <file name="firmware/myproject.cpp" sc="0" tb="false" cflags="-std=c++14" blackbox="false" csimflags=""/>
  </files>
  <solutions>
    <solution name="solution1" status="active"/>
  </solutions>
  <includePaths/>
  <libraryPaths/>
  <Simulation>
    <SimFlow name="csim" csimMode="0" lastCsimMode="0"/>
  </Simulation>
</project>
