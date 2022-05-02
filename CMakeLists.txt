cmake_minimum_required(VERSION 3.12)

# set the project name
project(SoATest)

set(CMAKE_CXX_STANDARD 17)

# add the executable
add_executable(SoATest src/soa.cpp)
target_include_directories(SoATest PUBLIC include)
target_link_libraries(SoATest PUBLIC pthread gtest gtest_main)