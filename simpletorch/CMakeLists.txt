CMAKE_MINIMUM_REQUIRED(VERSION 3.12)

PROJECT(simpletorch)

FIND_PACKAGE(Torch REQUIRED)
FIND_PACKAGE(PythonLibs REQUIRED)

INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_DIRS})

ADD_EXECUTABLE(simpletorch main.cpp)

TARGET_LINK_LIBRARIES(simpletorch ${PYTHON_LIBRARIES} ${TORCH_LIBRARIES})