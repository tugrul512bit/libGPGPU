cmake_minimum_required(VERSION 3.20)

project(MultiplyAddBench)

FILE(GLOB CppSources *.cpp)
add_executable(${PROJECT_NAME} ${CppSources})
find_package(OpenCL REQUIRED)
target_link_libraries(MultiplyAddBench PRIVATE OpenCL::OpenCL)

