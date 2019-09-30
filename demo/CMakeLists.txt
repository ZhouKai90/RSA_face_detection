cmake_minimum_required(VERSION 2.8)

set(RSA_UNIT_TEST
    main.cpp
    image_file_opt.cpp
    )

include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    ${OpneCV_INCLUDE_DIRS}
    )
    
set(LINKER_EXT_LIBS
    ${OpenCV_LIBRARIES}
    ${CUDA_LIBRARIES}
    libfaceDetection.so
    libcaffe.so
    )

link_directories(${PROJECT_SOURCE_DIR}/lib
                ${PROJECT_SOURCE_DIR}/caffe)

add_executable(target ${RSA_UNIT_TEST})
target_link_libraries(target ${LINKER_EXT_LIBS})