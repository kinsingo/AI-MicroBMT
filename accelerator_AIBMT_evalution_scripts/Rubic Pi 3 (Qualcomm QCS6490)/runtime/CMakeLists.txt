cmake_minimum_required(VERSION 3.14)

project(AI_BMT_GUI_Submitter_App VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PROJECT_SOURCES main.cpp)

# Create the executable
add_executable(AI_BMT_GUI_Submitter ${PROJECT_SOURCES})

# OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(AI_BMT_GUI_Submitter PUBLIC OpenMP::OpenMP_CXX)
endif()

# OpenCV 
find_package(OpenCV REQUIRED)
target_include_directories(AI_BMT_GUI_Submitter PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(AI_BMT_GUI_Submitter PUBLIC ${OpenCV_LIBS})

# ONNX Runtime 
# ORT headers (기존 그대로 사용)
set(ONNXRUNTIME_DIR "/home/ubuntu/onnxruntime-linux-aarch64-1.23.0")
target_include_directories(AI_BMT_GUI_Submitter PUBLIC ${ONNXRUNTIME_DIR}/include)

# ✅ QNN 포함 ORT 라이브러리 (실제 존재하는 파일로 링크)
set(ORT_CAPI_DIR "/home/ubuntu/miniconda3/envs/ai-bmt/lib/python3.12/site-packages/onnxruntime/capi")
set(ONNXRUNTIME_QNN_SO "${ORT_CAPI_DIR}/libonnxruntime.so.1.23.0")

target_link_libraries(AI_BMT_GUI_Submitter PUBLIC ${ONNXRUNTIME_QNN_SO})

# 런타임 로딩을 위해 RPATH에 capi dir 추가 (강추)
set_target_properties(AI_BMT_GUI_Submitter PROPERTIES
  BUILD_RPATH "${ORT_CAPI_DIR};${CMAKE_BINARY_DIR}/lib"
  INSTALL_RPATH "${ORT_CAPI_DIR};${CMAKE_INSTALL_PREFIX}/lib"
  INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Add the directory where the headers are located
target_include_directories(AI_BMT_GUI_Submitter PUBLIC 
${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link the libraries to the executable
target_link_libraries(AI_BMT_GUI_Submitter PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/build/libAI_BMT_GUI_Library.so)



# Ensure RPATH is always used (not stripped out)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
