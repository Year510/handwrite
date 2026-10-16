cmake_minimum_required(VERSION 3.16)
project(handwrite LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(handwrite src/main.cpp)
target_include_directories(handwrite PRIVATE ${CMAKE_SOURCE_DIR}/src)

find_package(Threads REQUIRED)
target_link_libraries(handwrite PRIVATE Threads::Threads)

if(MSVC)
    target_compile_options(handwrite PRIVATE /utf-8)
else()
    target_compile_options(handwrite PRIVATE -finput-charset=UTF-8 -fexec-charset=UTF-8)
endif()
