
project(REINFORCE)

cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 11)

if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Threads REQUIRED)
find_package(Matlab REQUIRED MAT_LIBRARY MX_LIBRARY)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -flto")

include_directories(${Matlab_INCLUDE_DIRS})

add_executable(reinforce_small main_small.cpp)
add_executable(reinforce_large main_large.cpp)
add_executable(reinforce_structured main_structured.cpp)

target_link_libraries(reinforce_small
    Eigen3::Eigen
    ${CMAKE_THREAD_LIBS_INIT}
    ${Matlab_LIBRARIES}
)

target_link_libraries(reinforce_large
    Eigen3::Eigen
    ${CMAKE_THREAD_LIBS_INIT}
    ${Matlab_LIBRARIES}
)

target_link_libraries(reinforce_structured
    Eigen3::Eigen
    ${CMAKE_THREAD_LIBS_INIT}
    ${Matlab_LIBRARIES}
)
