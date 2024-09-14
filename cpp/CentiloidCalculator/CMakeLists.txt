# CMakeList.txt: CentiloidCalculator 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required (VERSION 3.8)

# Find ITK.
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
find_package(SimpleITK REQUIRED)
include(${SimpleITK_USE_FILE})



# ONNX
set(ONNXRUNTIME_DIR "E:/projects/sitk/SITKExample/onnxruntime-win-x64-1.18.0")
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)


# 将源代码添加到此项目的可执行文件。
add_executable (CentiloidCalculator "CentiloidCalculator.cpp" "CentiloidCalculator.h" "Rigid.h" "Rigid.cpp")

message(STATUS "ITK_LIBRARIES: ${ITK_LIBRARIES}")
target_link_libraries(CentiloidCalculator ${ITK_LIBRARIES} ${SimpleITK_LIBRARIES} onnxruntime)