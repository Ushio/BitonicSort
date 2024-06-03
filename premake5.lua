workspace "BitonicSort"
    location "build"
    configurations { "Debug", "Release" }
    startproject "main"

architecture "x86_64"

project "bitonic_cpu"
    kind "ConsoleApp"
    language "C++"
    targetdir "bin/"
    systemversion "latest"
    flags { "MultiProcessorCompile", "NoPCH" }

    -- Src
    files { "main_cpu.cpp" }

    -- Orochi
    -- defines {"OROCHI_ENABLE_CUEW"}
    -- includedirs {"$(CUDA_PATH)/include"}

    -- includedirs { "libs/orochi" }
    -- files { "libs/orochi/Orochi/Orochi.h" }
    -- files { "libs/orochi/Orochi/Orochi.cpp" }
    -- includedirs { "libs/orochi/contrib/hipew/include" }
    -- files { "libs/orochi/contrib/hipew/src/hipew.cpp" }
    -- includedirs { "libs/orochi/contrib/cuew/include" }
    -- files { "libs/orochi/contrib/cuew/src/cuew.cpp" }
    -- links { "version" }

    -- postbuildcommands { 
    --     "{COPYFILE} ../libs/orochi/contrib/bin/win64/*.dll ../bin"
    -- }

    symbols "On"

    filter {"Debug"}
        runtime "Debug"
        targetname ("bitonic_cpu_Debug")
        optimize "Off"
    filter {"Release"}
        runtime "Release"
        targetname ("bitonic_cpu")
        optimize "Full"
    filter{}

project "bitonic_gpu"
    kind "ConsoleApp"
    language "C++"
    targetdir "bin/"
    systemversion "latest"
    flags { "MultiProcessorCompile", "NoPCH" }

    -- Src
    files { "main_gpu.cpp", "shader.hpp", "typedbuffer.hpp", "typedbuffer.natvis", "kernel.cu" }
    
    -- Orochi
    defines {"OROCHI_ENABLE_CUEW"}
    includedirs {"$(CUDA_PATH)/include"}

    includedirs { "libs/orochi" }
    files { "libs/orochi/Orochi/Orochi.h" }
    files { "libs/orochi/Orochi/Orochi.cpp" }
    files { "libs/orochi/Orochi/OrochiUtils.h" }
    files { "libs/orochi/Orochi/OrochiUtils.cpp" }
    includedirs { "libs/orochi/contrib/hipew/include" }
    files { "libs/orochi/contrib/hipew/src/hipew.cpp" }
    includedirs { "libs/orochi/contrib/cuew/include" }
    files { "libs/orochi/contrib/cuew/src/cuew.cpp" }
    links { "version" }

    postbuildcommands { 
        "{COPYFILE} ../libs/orochi/contrib/bin/win64/*.dll ../bin"
    }

    symbols "On"

    filter {"Debug"}
        runtime "Debug"
        targetname ("bitonic_gpu_Debug")
        optimize "Off"
    filter {"Release"}
        runtime "Release"
        targetname ("bitonic_gpu")
        optimize "Full"
    filter{}