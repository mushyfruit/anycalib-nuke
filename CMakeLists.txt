cmake_minimum_required(VERSION 3.20)
project(AnyCalibNuke VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(ANYCALIB_BUILD_NUKE "Build the Nuke plugin and libtorch inference CLI" ON)

# Builds and tests without Nuke, Torch or downloads.
add_library(anycalib_core STATIC src/LensModel.cpp src/CameraFit.cpp src/LineDetect.cpp
  src/ShotAnalysis.cpp)
target_include_directories(anycalib_core PUBLIC src)
set_target_properties(anycalib_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
enable_testing()
add_executable(test_fit tests/test_fit.cpp)
target_link_libraries(test_fit PRIVATE anycalib_core)
add_test(NAME fit COMMAND test_fit)
add_executable(test_shot tests/test_shot.cpp)
target_link_libraries(test_shot PRIVATE anycalib_core)
add_test(NAME shot COMMAND test_shot)

if(NOT ANYCALIB_BUILD_NUKE)
  return()
endif()

set(NUKE_ROOT "$ENV{NUKE_ROOT}" CACHE PATH "Nuke install directory")
if(NOT NUKE_ROOT)
  message(FATAL_ERROR "Set NUKE_ROOT (e.g. -DNUKE_ROOT=$HOME/nuke/Nuke17.0v3)")
endif()
find_package(Nuke CONFIG REQUIRED PATHS "${NUKE_ROOT}/cmake" NO_DEFAULT_PATH)
find_package(OpenGL REQUIRED)

# The network runs on the libtorch Nuke ships (PyTorch 2.7.1, with its CUDA libraries).
# Only the headers come from the official release; everything links Nuke's libraries.
# -DLIBTORCH_ROOT=<extracted libtorch 2.7.1> skips the download.
set(LIBTORCH_ROOT "" CACHE PATH "Extracted libtorch 2.7.1 release (optional)")
if(NOT LIBTORCH_ROOT)
  include(FetchContent)
  FetchContent_Declare(libtorch
    URL "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcpu.zip"
    URL_HASH SHA256=63d572598c8d532128a335018913e795c1bbb32602ce378896dc8cfbb5590976
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  FetchContent_MakeAvailable(libtorch)
  set(LIBTORCH_ROOT "${libtorch_SOURCE_DIR}")
endif()
add_library(nuke_torch INTERFACE)
target_include_directories(nuke_torch SYSTEM INTERFACE
  "${LIBTORCH_ROOT}/include" "${LIBTORCH_ROOT}/include/torch/csrc/api/include")
# libtorch.so only pulls in libtorch_cuda.so, so keep it even though nothing calls into it.
target_link_libraries(nuke_torch INTERFACE
  "-Wl,--push-state,--no-as-needed"
  "${NUKE_ROOT}/libtorch.so" "${NUKE_ROOT}/libtorch_cpu.so" "${NUKE_ROOT}/libc10.so"
  "-Wl,--pop-state")

add_library(anycalib_engine STATIC src/AnyCalibEngine.cpp)
target_link_libraries(anycalib_engine PUBLIC anycalib_core nuke_torch)
set_target_properties(anycalib_engine PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_nuke_plugin(AnyCalib src/AnyCalibNuke.cpp)
target_link_libraries(AnyCalib PRIVATE anycalib_engine OpenGL::GL)
target_compile_definitions(AnyCalib PRIVATE ANYCALIB_NUKE_VERSION="${PROJECT_VERSION}")
# No RPATH: inside Nuke, its own RPATH ($ORIGIN) finds libtorch and the CUDA libraries.
set_target_properties(AnyCalib PROPERTIES SKIP_BUILD_RPATH ON)

# Outside Nuke the CLI finds its libraries through the build RPATH: DT_RPATH, not RUNPATH,
# so libtorch's own dependencies see it too.
add_executable(anycalib_cli tests/anycalib_cli.cpp)
target_link_libraries(anycalib_cli PRIVATE anycalib_engine)
target_link_options(anycalib_cli PRIVATE "LINKER:--disable-new-dtags")

# <prefix>/anycalib/{AnyCalib.so, menu.py, anycalib_nuke.py, models/}: add that folder to NUKE_PATH.
set(ANYCALIB_MODEL_FILE "" CACHE FILEPATH "TorchScript network to bundle (optional)")
install(TARGETS AnyCalib LIBRARY DESTINATION anycalib)
install(FILES python/menu.py python/anycalib_nuke.py DESTINATION anycalib)
if(ANYCALIB_MODEL_FILE)
  install(FILES "${ANYCALIB_MODEL_FILE}" DESTINATION anycalib/models RENAME anycalib_dist.pt)
endif()
