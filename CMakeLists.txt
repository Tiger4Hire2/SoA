cmake_minimum_required(VERSION 3.12)

# set the project name
project(SoATest)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
find_package(ComputeCpp REQUIRED)

set(CMAKE_CXX_STANDARD 17)

enable_testing()
# add the executable
add_executable(SoATest src/soa.cpp)
target_include_directories(SoATest PUBLIC include "${ComputeCpp_DIR}/include")
target_link_libraries(SoATest PUBLIC pthread gtest gtest_main ${ComputeCpp_DIR}/lib/libComputeCpp.so)
add_sycl_to_target(TARGET SoATest SOURCES src/soa.cpp)