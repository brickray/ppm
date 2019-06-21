cmake_minimum_required(VERSION 3.2)

PROJECT(sample_ppm)

SET(CMAKE_BUILD_TYPE RELEASE)

INCLUDE_DIRECTORIES(ext)
LINK_DIRECTORIES(ext)
LINK_LIBRARIES(freeglut.lib glew32.lib)

ADD_EXECUTABLE(sample_ppm src/main.cpp)
