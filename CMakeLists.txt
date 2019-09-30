cmake_minimum_required(VERSION 2.8)
project(RSA_EX)

include(CheckCXXCompilerFlag)

CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} not suppoet C++11.")
endif()

find_package(OpenCV 3.4.1)
find_package(CUDA)
find_package(Boost REQUIRED COMPONENTS system filesystem)

add_subdirectory(src)
add_subdirectory(demo)


install(FILES 
        ${PROJECT_SOURCE_DIR}/lib/libfaceDetection.so
	${PROJECT_SOURCE_DIR}/caffe/libcaffe.so
        ${PROJECT_SOURCE_DIR}/caffe/libcaffe.so.1.0.0
		DESTINATION ${PROJECT_SOURCE_DIR}/bin
        )

install(FILES 
        ${PROJECT_SOURCE_DIR}/src/inf_rsa_face_detection_api.hh
		DESTINATION ${PROJECT_SOURCE_DIR}/bin/include
		)

install(DIRECTORY 
        ${PROJECT_SOURCE_DIR}/model/
        DESTINATION ${PROJECT_SOURCE_DIR}/bin/model)