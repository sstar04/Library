cmake_minimum_required(VERSION 2.8.5 FATAL_ERROR)
find_package(CUDA QUIET REQUIRED)
option(DEBUG "Print debug information" ON)
add_subdirectory(TMEInspector)
include_directories(TMEInspector) 

cuda_add_executable(TestUM TestUM.cpp )
set(LIBS ${LIBS} TMEInspector)
target_link_libraries(TestUM ${LIBS})

