cmake_minimum_required(VERSION 3.10)
project(opticalFlow)

set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_CXX_FLAGS "")

set(CMAKE_CXX_FLAGS_RELEASE "-O3")
#set(CMAKE_CXX_FLAGS_DEBUG "-O3")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(opticalFlow
        opticalFlow.cpp
        ORBextractor.cpp
        ORBextractor.hpp
        robustMatcher.cpp
        robustMatcher.hpp
        DenseORB.cpp
        DenseORB.hpp
        )

target_link_libraries(opticalFlow ${OpenCV_LIBS})