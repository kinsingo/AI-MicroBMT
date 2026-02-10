#!/bin/bash

# INT8 ONNX 모델들의 Dynamic Batch Size를 1로 고정하는 스크립트
# Rubik Pi 3 NPU requirement: Fixed batch size = 1

set -e  # Exit on error

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Conda 환경 설정
CONDA_ENV="ai-bmt"

# 디렉토리 설정
INPUT_DIR="output_onnx_torchvision_int8"
OUTPUT_DIR="output_onnx_torchvision_int8_batch_fixed"
INPUT_SHAPE="1,3,224,224"  # batch_size=1, channels=3, height=224, width=224

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Fix Dynamic Batch Size to 1${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Input directory:  ${INPUT_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Input shape:      ${INPUT_SHAPE}"
echo ""

# 출력 디렉토리 생성
mkdir -p "${OUTPUT_DIR}"

# ONNX 파일 찾기
echo -e "${BLUE}Searching for ONNX files...${NC}"
ONNX_FILES=($(find "${INPUT_DIR}" -type f -name "*.onnx"))
TOTAL_FILES=${#ONNX_FILES[@]}

if [ ${TOTAL_FILES} -eq 0 ]; then
    echo -e "${RED}No ONNX files found in ${INPUT_DIR}${NC}"
    exit 1
fi

echo -e "${GREEN}Found ${TOTAL_FILES} ONNX files${NC}"
echo ""

# 카운터 초기화
SUCCESS_COUNT=0
SKIP_COUNT=0
FAIL_COUNT=0
FAILED_MODELS=()

# 각 ONNX 파일 처리
for i in "${!ONNX_FILES[@]}"; do
    ONNX_FILE="${ONNX_FILES[$i]}"
    INDEX=$((i+1))
    
    # 상대 경로 계산
    REL_PATH="${ONNX_FILE#${INPUT_DIR}/}"
    
    # 출력 파일 경로 (서브폴더 구조 유지)
    OUTPUT_FILE="${OUTPUT_DIR}/${REL_PATH}"
    OUTPUT_SUBDIR=$(dirname "${OUTPUT_FILE}")
    
    echo -e "${BLUE}[${INDEX}/${TOTAL_FILES}] Processing: ${REL_PATH}${NC}"
    
    # 이미 존재하는 경우 Skip
    if [ -f "${OUTPUT_FILE}" ]; then
        echo -e "  ${YELLOW}✓ Skipped (already exists)${NC}"
        SKIP_COUNT=$((SKIP_COUNT+1))
        echo ""
        continue
    fi
    
    # 출력 서브디렉토리 생성
    mkdir -p "${OUTPUT_SUBDIR}"
    
    # 입력 이름 추출 (netron으로 확인하거나 기본값 사용)
    # 대부분의 PyTorch 모델은 "input" 또는 "x"를 사용
    # 입력 이름을 자동으로 찾기 위해 python 스크립트 사용
    INPUT_NAME=$(conda run -n ${CONDA_ENV} python3 -c "
import onnx
import sys
try:
    model = onnx.load('${ONNX_FILE}')
    print(model.graph.input[0].name)
except Exception as e:
    print('input', file=sys.stderr)
    sys.exit(0)
" 2>/dev/null || echo "input")
    
    # Dynamic shape 고정 실행
    if conda run -n ${CONDA_ENV} python3 -m onnxruntime.tools.make_dynamic_shape_fixed \
        "${ONNX_FILE}" \
        "${OUTPUT_FILE}" \
        --input_name "${INPUT_NAME}" \
        --input_shape "${INPUT_SHAPE}" > /dev/null 2>&1; then
        
        # 파일 크기 비교
        INPUT_SIZE=$(stat -f%z "${ONNX_FILE}" 2>/dev/null || stat -c%s "${ONNX_FILE}" 2>/dev/null)
        OUTPUT_SIZE=$(stat -f%z "${OUTPUT_FILE}" 2>/dev/null || stat -c%s "${OUTPUT_FILE}" 2>/dev/null)
        INPUT_SIZE_MB=$(echo "scale=2; ${INPUT_SIZE}/1024/1024" | bc)
        OUTPUT_SIZE_MB=$(echo "scale=2; ${OUTPUT_SIZE}/1024/1024" | bc)
        
        echo -e "  ${GREEN}✓ Success${NC}"
        echo "    Input size:  ${INPUT_SIZE_MB} MB"
        echo "    Output size: ${OUTPUT_SIZE_MB} MB"
        SUCCESS_COUNT=$((SUCCESS_COUNT+1))
    else
        echo -e "  ${RED}✗ Failed${NC}"
        FAILED_MODELS+=("${REL_PATH}")
        FAIL_COUNT=$((FAIL_COUNT+1))
    fi
    
    echo ""
done

# 최종 요약
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Batch Size Fixing Completed${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Total models:         ${TOTAL_FILES}"
echo -e "${GREEN}Successfully fixed:   ${SUCCESS_COUNT}${NC}"
echo -e "${YELLOW}Skipped (exists):     ${SKIP_COUNT}${NC}"
echo -e "${RED}Failed:               ${FAIL_COUNT}${NC}"

if [ ${FAIL_COUNT} -gt 0 ]; then
    echo ""
    echo -e "${RED}Failed models:${NC}"
    for model in "${FAILED_MODELS[@]}"; do
        echo "  - ${model}"
    done
fi

echo ""
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# 결과를 파일로 저장
RESULT_FILE="${OUTPUT_DIR}/fix_batch_size_result.txt"
{
    echo "========================================"
    echo "Fix Dynamic Batch Size Results"
    echo "========================================"
    echo ""
    echo "Date: $(date)"
    echo "Input directory:  ${INPUT_DIR}"
    echo "Output directory: ${OUTPUT_DIR}"
    echo "Input shape:      ${INPUT_SHAPE}"
    echo ""
    echo "Total models:         ${TOTAL_FILES}"
    echo "Successfully fixed:   ${SUCCESS_COUNT}"
    echo "Skipped (exists):     ${SKIP_COUNT}"
    echo "Failed:               ${FAIL_COUNT}"
    echo ""
    
    if [ ${FAIL_COUNT} -gt 0 ]; then
        echo "========================================"
        echo "Failed Models"
        echo "========================================"
        echo ""
        for model in "${FAILED_MODELS[@]}"; do
            echo "  - ${model}"
        done
        echo ""
    fi
} > "${RESULT_FILE}"

echo "Results saved to: ${RESULT_FILE}"

# Exit code
if [ ${FAIL_COUNT} -gt 0 ]; then
    exit 1
else
    exit 0
fi
