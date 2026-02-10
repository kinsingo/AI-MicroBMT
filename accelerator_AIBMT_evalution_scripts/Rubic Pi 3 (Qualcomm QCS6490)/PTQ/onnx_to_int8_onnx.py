#!/usr/bin/env python3
"""
ONNX Post-Training Quantization (INT8) for Rubik Pi 3 NPU

Rubik Pi 3 NPU 요구사항:
- uint8/int8 quantized models only
- Fixed input shape (no dynamic shapes)
- ONNX format with QNN ExecutionProvider support

참고: https://www.thundercomm.com/rubik-pi-3/en/docs/rubik-pi-3-user-manual/1.0.0-u/Application%20Development%20and%20Execution%20Guide/Framework-Driven%20AI%20Sample%20Execution/onnx
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import onnx
from onnx import shape_inference
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType

class ImageDataReader(CalibrationDataReader):
    """
    Calibration 이미지를 읽어서 ONNX Quantization에 제공하는 DataReader
    """
    def __init__(self, calibration_image_folder, input_name, input_shape, num_samples=1024):
        """
        Args:
            calibration_image_folder: Calibration 이미지 폴더 경로
            input_name: ONNX 모델의 입력 이름 (예: "input")
            input_shape: 입력 shape, 예: (1, 3, 224, 224) for NCHW
            num_samples: 사용할 calibration 샘플 수 (기본 1024, 전체 데이터)
        """
        self.image_folder = Path(calibration_image_folder)
        self.input_name = input_name
        self.input_shape = input_shape
        self.num_samples = num_samples
        
        # 이미지 파일 수집
        self.image_files = self._collect_images()
        
        # Iterator 초기화
        self.enum_data = None
        
    def _collect_images(self):
        """이미지 파일 수집"""
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in extensions:
            image_files.extend(self.image_folder.glob(f'*{ext}'))
            image_files.extend(self.image_folder.glob(f'*{ext.upper()}'))
        
        total_images = len(image_files)
        image_files = sorted(image_files)[:self.num_samples]
        print(f"  Using {len(image_files)}/{total_images} calibration images")
        return image_files
    
    def get_next(self):
        """다음 calibration 샘플 반환"""
        if self.enum_data is None:
            self.enum_data = iter(self.image_files)
        
        try:
            image_path = next(self.enum_data)
            return self._preprocess_image(image_path)
        except StopIteration:
            return None
    
    def _preprocess_image(self, image_path):
        """
        이미지 전처리 및 ONNX 입력 형식으로 변환
        
        Note: Inference 시와 동일한 전처리 적용 필수!
              ImageNet pretrained 모델의 표준 전처리:
              1. [0-255] → [0-1] 정규화
              2. ImageNet mean/std normalization
              3. HWC → CHW transpose
        """
        # Input shape 파싱
        batch, channels, height, width = self.input_shape
        
        # 이미지 로드 및 리사이즈 (BGR이 아닌 RGB로 로드)
        img = Image.open(image_path).convert("RGB")
        img = img.resize((width, height))
        
        # NumPy 배열로 변환
        img_array = np.array(img, dtype=np.float32)
        
        # [0-255] → [0-1] 정규화
        img_array = img_array / 255.0
        
        # ImageNet mean/std normalization (inference 코드와 동일)
        # PyTorch ImageNet 표준값
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_array = (img_array - mean) / std
        
        # HWC → CHW 변환
        img_array = np.transpose(img_array, (2, 0, 1))
        
        # 배치 차원 추가
        img_array = np.expand_dims(img_array, axis=0)
        
        return {self.input_name: img_array}


def get_input_info(onnx_model_path):
    """
    ONNX 모델의 입력 정보 추출
    
    Returns:
        (input_name, input_shape) 튜플
    """
    model = onnx.load(onnx_model_path)
    
    # 첫 번째 입력 정보 가져오기
    input_tensor = model.graph.input[0]
    input_name = input_tensor.name
    
    # Shape 추출
    input_shape = []
    for dim in input_tensor.type.tensor_type.shape.dim:
        if dim.dim_param:
            # Dynamic dimension (예: "batch_size")
            input_shape.append(1)  # Fixed batch size로 설정
        else:
            input_shape.append(dim.dim_value)
    
    return input_name, tuple(input_shape)


def quantize_onnx_model(
    input_model_path,
    output_model_path,
    calibration_data_reader,
    quant_format='QDQ'
):
    """
    ONNX 모델을 INT8로 양자화
    
    Args:
        input_model_path: 입력 ONNX 모델 경로
        output_model_path: 출력 INT8 ONNX 모델 경로
        calibration_data_reader: CalibrationDataReader 인스턴스
        quant_format: 'QDQ' (Quantize-Dequantize) 또는 'QOperator'
                     QDQ는 NPU 호환성이 더 좋습니다.
    
    Returns:
        (success: bool, error_message: str or None)
    """
    
    print(f"  Input model: {Path(input_model_path).name}")
    print(f"  Output model: {Path(output_model_path).name}")
    print(f"  Quantization format: {quant_format}")
    
    try:
        quantize_static(
            model_input=str(input_model_path),
            model_output=str(output_model_path),
            calibration_data_reader=calibration_data_reader,
            quant_format=quant_format,
            weight_type=QuantType.QInt8,      # INT8 weights
            activation_type=QuantType.QUInt8,  # UINT8 activations (NPU 요구사항)
            per_channel=True,  # Per-channel quantization (정확도 향상)
        )
        
        # 파일 크기 확인
        input_size = Path(input_model_path).stat().st_size / (1024 * 1024)
        output_size = Path(output_model_path).stat().st_size / (1024 * 1024)
        compression_ratio = (1 - output_size / input_size) * 100
        
        print(f"  ✓ Quantization successful")
        print(f"    Original size: {input_size:.2f} MB")
        print(f"    Quantized size: {output_size:.2f} MB")
        print(f"    Compression: {compression_ratio:.1f}%")
        
        return (True, None)
        
    except Exception as e:
        error_msg = str(e)[:500]  # 에러 메시지 캡처
        print(f"  ✗ Quantization failed: {error_msg}")
        return (False, error_msg)


def convert_onnx_to_int8(
    onnx_model_path,
    output_dir,
    calibration_dir,
    num_samples=1024
):
    """
    단일 ONNX 모델을 INT8로 변환
    
    Args:
        onnx_model_path: 입력 ONNX 모델 경로
        output_dir: 출력 디렉토리
        calibration_dir: Calibration 이미지 디렉토리
        num_samples: 사용할 calibration 샘플 수 (기본 1024, 전체 데이터)
    """
    onnx_path = Path(onnx_model_path)
    output_path = Path(output_dir) / onnx_path.name.replace('.onnx', '_int8.onnx')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 이미 생성된 모델은 Skip
    if output_path.exists():
        print(f"\n{'='*80}")
        print(f"Skipping (already exists): {onnx_path.name}")
        print(f"  Output: {output_path.name}")
        print(f"{'='*80}")
        return "skipped"
    
    print(f"\n{'='*80}")
    print(f"Quantizing: {onnx_path.name}")
    print(f"{'='*80}")
    
    # 1. 입력 정보 추출
    try:
        input_name, input_shape = get_input_info(onnx_path)
        print(f"  Input name: {input_name}")
        print(f"  Input shape: {input_shape}")
    except Exception as e:
        print(f"  ✗ Failed to read model info: {e}")
        return False
    
    # 2. Calibration DataReader 생성
    calibration_reader = ImageDataReader(
        calibration_image_folder=calibration_dir,
        input_name=input_name,
        input_shape=input_shape,
        num_samples=num_samples
    )
    
    # 3. INT8 양자화 수행
    success, error_msg = quantize_onnx_model(
        input_model_path=onnx_path,
        output_model_path=output_path,
        calibration_data_reader=calibration_reader,
        quant_format='QDQ',  # QDQ는 NPU 호환성 우수
    )
    
    if success:
        return True
    else:
        return (False, error_msg)


def convert_folder_to_int8(
    onnx_folder,
    output_root,
    calibration_dir,
    num_samples=1024
):
    """
    폴더 내 모든 ONNX 모델을 INT8로 변환
    
    Args:
        onnx_folder: 입력 ONNX 모델 폴더 (서브폴더 포함)
        output_root: 출력 루트 디렉토리
        calibration_dir: Calibration 이미지 디렉토리
        num_samples: 사용할 calibration 샘플 수 (기본 1024, 전체 데이터)
    """
    onnx_folder = Path(onnx_folder)
    output_root = Path(output_root)
    calibration_path = Path(calibration_dir)
    
    # Calibration 이미지 개수 미리 확인
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']
    calib_images = []
    for ext in extensions:
        calib_images.extend(calibration_path.glob(f'*{ext}'))
    total_calib_images = len(calib_images)
    actual_samples = min(num_samples, total_calib_images)
    
    print(f"\n{'='*80}")
    print(f"Batch ONNX INT8 Quantization")
    print(f"Input folder: {onnx_folder}")
    print(f"Output folder: {output_root}")
    print(f"Calibration folder: {calibration_dir}")
    print(f"Calibration images: {actual_samples}/{total_calib_images} images (using {actual_samples} per model)")
    print(f"{'='*80}\n")
    
    # ONNX 파일 수집 (재귀적으로)
    onnx_files = sorted(onnx_folder.rglob('*.onnx'))
    
    if not onnx_files:
        print("No ONNX files found!")
        return
    
    print(f"Found {len(onnx_files)} ONNX models to quantize\n")
    
    success_count = 0
    skipped_count = 0
    failed_models = {}  # {model_path: error_reason}
    
    for idx, onnx_path in enumerate(onnx_files, 1):
        # 상대 경로 계산
        relative_path = onnx_path.relative_to(onnx_folder)
        
        # 출력 경로 (서브폴더 구조 유지)
        output_dir = output_root / relative_path.parent
        
        print(f"[{idx}/{len(onnx_files)}] Processing: {relative_path}")
        
        try:
            result = convert_onnx_to_int8(
                onnx_model_path=onnx_path,
                output_dir=output_dir,
                calibration_dir=calibration_dir,
                num_samples=num_samples
            )
            
            if result == "skipped":
                skipped_count += 1
            elif isinstance(result, tuple) and not result[0]:
                # Failure with error message
                failed_models[str(relative_path)] = result[1]
            elif result:
                success_count += 1
            else:
                failed_models[str(relative_path)] = "Quantization returned False (unknown error)"
        
        except Exception as e:
            error_msg = str(e)[:500]  # Limit error message length
            print(f"  ✗ Unexpected error: {error_msg}")
            failed_models[str(relative_path)] = error_msg
    
    # 최종 요약
    total_models = len(onnx_files)
    failed_count = len(failed_models)
    
    print(f"\n{'='*80}")
    print(f"Batch Quantization Completed")
    print(f"  Total models: {total_models}")
    print(f"  Successfully quantized: {success_count}")
    print(f"  Skipped (already exists): {skipped_count}")
    print(f"  Failed: {failed_count}")
    
    if failed_models:
        print(f"\nFailed models:")
        for model_path, error_reason in failed_models.items():
            print(f"  - {model_path}")
            print(f"    Reason: {error_reason}")
    
    print(f"\nOutput directory: {output_root}")
    print(f"{'='*80}\n")
    
    # 결과를 파일로 저장
    result_file = output_root / "quantization_result.txt"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("ONNX INT8 Quantization Results\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Input folder: {onnx_folder}\n")
        f.write(f"Output folder: {output_root}\n")
        f.write(f"Calibration folder: {calibration_dir}\n")
        f.write(f"Calibration samples: {actual_samples} images\n\n")
        
        f.write(f"Total models: {total_models}\n")
        f.write(f"Successfully quantized: {success_count}\n")
        f.write(f"Skipped (already exists): {skipped_count}\n")
        f.write(f"Failed: {failed_count}\n\n")
        
        if failed_models:
            f.write("="*80 + "\n")
            f.write("Failed Models (with reasons)\n")
            f.write("="*80 + "\n\n")
            for model_path, error_reason in failed_models.items():
                f.write(f"Model: {model_path}\n")
                f.write(f"Reason: {error_reason}\n")
                f.write("-"*80 + "\n\n")
    
    print(f"Results saved to: {result_file}")


if __name__ == "__main__":
    import argparse
    
    # /tmp 가 가득 찼을 경우를 대비해 임시 디렉토리를 SSD1로 변경
    os.environ['TMPDIR'] = '/SSD1/tmp'
    os.environ['TEMP'] = '/SSD1/tmp'
    os.environ['TMP'] = '/SSD1/tmp'
    Path('/SSD1/tmp').mkdir(exist_ok=True)
    
    parser = argparse.ArgumentParser(
        description='Quantize ONNX models to INT8 for Rubik Pi 3 NPU'
    )
    
    # 단일 모델 변환
    parser.add_argument('--model', type=str, default=None,
                       help='Single ONNX model path to quantize')
    
    # 배치 변환
    parser.add_argument('--model_folder', type=str,
                       default='/SSD1/jonghyun/RubicPi3-Models/output_onnx_torchvision',
                       help='Folder containing ONNX models (recursive)')
    
    # Calibration 설정
    parser.add_argument('--calibration_dir', type=str,
                       default='/SSD1/jonghyun/RubicPi3-Models/Calibration_Images_Classification',
                       help='Calibration images directory')
    parser.add_argument('--num_samples', type=int, default=1024,
                       help='Number of calibration samples (default: 1024, use all images)')
    
    # 출력 설정
    parser.add_argument('--output_dir', type=str, default='output_onnx_torchvision_int8',
                       help='Output directory for quantized models')
    
    args = parser.parse_args()
    
    # Calibration 디렉토리 확인
    if not Path(args.calibration_dir).exists():
        print(f"Error: Calibration directory not found: {args.calibration_dir}")
        sys.exit(1)
    
    # 단일 모델 변환
    if args.model:
        if not Path(args.model).exists():
            print(f"Error: Model file not found: {args.model}")
            sys.exit(1)
        
        convert_onnx_to_int8(
            onnx_model_path=args.model,
            output_dir=args.output_dir,
            calibration_dir=args.calibration_dir,
            num_samples=args.num_samples
        )
    
    # 배치 변환 (기본 동작)
    else:
        if not Path(args.model_folder).exists():
            print(f"Error: Model folder not found: {args.model_folder}")
            sys.exit(1)
        
        convert_folder_to_int8(
            onnx_folder=args.model_folder,
            output_root=args.output_dir,
            calibration_dir=args.calibration_dir,
            num_samples=args.num_samples
        )
