cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)
set(CMAKE_BUILD_TYPE Release)
set(CUDA_NVCC_FLAGS 
    ${CUDA_NVCC_FLAGS};
    -gencode arch=compute_35,code=sm_35
    -gencode arch=compute_30,code=sm_30)

if(DEBUG)
	add_definitions(-DDEBUG)
endif(DEBUG)

cuda_add_library(TMEInspector SHARED  TGSimulator.cu  TMEInspector.cu Filter.cu)
target_include_directories(TMEInspector PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

#install library
install (TARGETS TMEInspector DESTINATION bin)
install (FILES 
    TGSimulator.h
    TMEInspector.h 
    Filter.h
    DataStructure.h DESTINATION include)

