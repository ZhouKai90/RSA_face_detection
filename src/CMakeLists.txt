cmake_minimum_required(VERSION 2.8)

SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)           #设置库文件的输出目录

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61;-std=c++11;)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

set(SRC_SRC
    rsa_face_detection.cpp
    nms_kernel.cu
    inf_rsa_face_detection_api.cpp
)

set(CAFFE_ROOT ${PROJECT_SOURCE_DIR}/caffe)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/eigen3
    ${PROJECT_SOURCE_DIR}/src
    ${CAFFE_ROOT}/include
    ${OpneCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
)

# message(STATUS ${CAFFE_ROOT})
# message(STATUS ${CUDA_INCLUDE_DIRS})

link_directories(
    ${CAFFE_ROOT}
    ${Boost_LIBRARY_DIRS}
    "/usr/local/lib"
)

set(LINKER_EXT_LIBS
    ${OpenCV_LIBRARIES}
    ${Boost_LIBRARIES}
    ${CUDA_LIBRARIES}
    libcaffe.so
    libglog.so.0
)

cuda_add_library(faceDetection SHARED ${SRC_SRC})
target_link_libraries(faceDetection ${LINKER_EXT_LIBS})