#!/usr/bin/env python3
"""
Input Resolution Sweep용 ONNX INT8 Quantization with Fixed Batch Size

각 input resolution별로 개별 INT8 quantization 수행:
- Calibration 이미지를 해당 resolution에 맞게 resize
- Batch size = 1로 고정 (Rubik Pi 3 NPU 요구사항)
- Resolution sweep: 48x48 ~ 448x448 (step: 16) - 총 26개
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import onnx
from onnx import shape_inference
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType


class ResolutionAwareImageDataReader(CalibrationDataReader):
    """
    특정 resolution에 맞게 이미지를 resize하여 calibration 데이터 제공
    """
    def __init__(self, calibration_image_folder, input_name, target_resolution, num_samples=1024):
        """
        Args:
            calibration_image_folder: Calibration 이미지 폴더 경로
            input_name: ONNX 모델의 입력 이름
            target_resolution: 목표 해상도 (예: 224)
            num_samples: 사용할 calibration 샘플 수 (기본 1024)
        """
        self.image_folder = Path(calibration_image_folder)
        self.input_name = input_name
        self.target_resolution = target_resolution
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
        actual_samples = len(image_files)
        
        if actual_samples == 0:
            raise ValueError(f"No calibration images found in {self.image_folder}")
        
        print(f"    Using {actual_samples}/{total_images} calibration images")
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
    
    def rewind(self):
        """DataReader를 처음부터 다시 순회 가능하도록 리셋"""
        self.enum_data = None
    
    def _preprocess_image(self, image_path):
        """
        이미지 전처리: target_resolution에 맞게 resize
        
        ImageNet pretrained 모델의 표준 전처리:
        1. [0-255] → [0-1] 정규화
        2. ImageNet mean/std normalization
        3. HWC → CHW transpose
        """
        # 이미지 로드 및 리사이즈
        img = Image.open(image_path).convert("RGB")
        img = img.resize((self.target_resolution, self.target_resolution))
        
        # NumPy 배열로 변환
        img_array = np.array(img, dtype=np.float32)
        
        # [0-255] → [0-1] 정규화
        img_array = img_array / 255.0
        
        # ImageNet mean/std normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_array = (img_array - mean) / std
        
        # HWC → CHW 변환
        img_array = np.transpose(img_array, (2, 0, 1))
        
        # 배치 차원 추가 (batch_size=1)
        img_array = np.expand_dims(img_array, axis=0)
        
        return {self.input_name: img_array}


def fix_input_shape_to_resolution(fp32_model_path, output_model_path, resolution):
    """
    FP32 ONNX 모델의 input shape를 특정 resolution으로 고정 (batch_size=1)
    
    Args:
        fp32_model_path: 입력 FP32 ONNX 모델 경로
        output_model_path: 출력 ONNX 모델 경로 (fixed shape)
        resolution: 목표 해상도 (예: 224)
    
    Returns:
        (success: bool, input_name: str)
    """
    try:
        # 모델 로드
        model = onnx.load(fp32_model_path)
        
        # 입력 텐서 정보
        input_tensor = model.graph.input[0]
        input_name = input_tensor.name
        elem_type = input_tensor.type.tensor_type.elem_type
        
        # 새로운 shape 설정 (batch_size=1, channels=3, height=resolution, width=resolution)
        new_shape = [1, 3, resolution, resolution]
        
        # 새로운 입력 정의 생성
        new_input = onnx.helper.make_tensor_value_info(
            input_name,
            elem_type,
            new_shape
        )
        
        # 입력 교체
        model.graph.input.remove(input_tensor)
        model.graph.input.insert(0, new_input)
        
        # Shape inference (출력 shape도 자동 업데이트)
        try:
            model = shape_inference.infer_shapes(model)
        except:
            pass  # Shape inference 실패해도 진행
        
        # 모델 저장
        onnx.save(model, output_model_path)
        
        return (True, input_name)
        
    except Exception as e:
        print(f"    ✗ Failed to fix input shape: {e}")
        return (False, None)


def quantize_onnx_model(
    input_model_path,
    output_model_path,
    calibration_data_reader,
):
    """
    ONNX 모델을 INT8로 양자화
    
    Args:
        input_model_path: 입력 ONNX 모델 경로
        output_model_path: 출력 INT8 ONNX 모델 경로
        calibration_data_reader: CalibrationDataReader 인스턴스
    
    Returns:
        success: bool
    """
    try:
        quantize_static(
            model_input=str(input_model_path),
            model_output=str(output_model_path),
            calibration_data_reader=calibration_data_reader,
            quant_format='QDQ',  # QDQ는 NPU 호환성 우수
            weight_type=QuantType.QInt8,      # INT8 weights
            activation_type=QuantType.QUInt8,  # UINT8 activations (NPU 요구사항)
            per_channel=True,  # Per-channel quantization (정확도 향상)
        )
        
        # 파일 크기 확인
        input_size = Path(input_model_path).stat().st_size / (1024 * 1024)
        output_size = Path(output_model_path).stat().st_size / (1024 * 1024)
        compression_ratio = (1 - output_size / input_size) * 100
        
        print(f"    ✓ Quantization successful")
        print(f"      Original: {input_size:.2f} MB → Quantized: {output_size:.2f} MB ({compression_ratio:.1f}% compressed)")
        
        return True
        
    except Exception as e:
        print(f"    ✗ Quantization failed: {str(e)[:200]}")
        return False


def process_resolution_sweep(
    base_model_path,
    output_dir,
    calibration_dir,
    resolutions,
    num_samples=1024
):
    """
    단일 베이스 모델에 대해 여러 resolution의 INT8 모델 생성
    
    Args:
        base_model_path: 베이스 FP32 ONNX 모델 경로
        output_dir: 출력 디렉토리
        calibration_dir: Calibration 이미지 디렉토리
        resolutions: Resolution 리스트 (예: [48, 64, 80, ..., 448])
        num_samples: Calibration 샘플 수 (기본 1024)
    
    Returns:
        (success_count, fail_count, skip_count)
    """
    base_model_path = Path(base_model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 베이스 모델명 추출
    base_name = base_model_path.stem  # 확장자 제거
    base_name = base_name.replace('_opset13', '')  # _opset13 제거
    
    print(f"\n{'='*80}")
    print(f"Base Model: {base_model_path.name}")
    print(f"{'='*80}\n")
    
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    # 임시 디렉토리 (fixed shape FP32 모델 저장용)
    temp_dir = output_dir / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    for idx, resolution in enumerate(resolutions, 1):
        print(f"[{idx}/{len(resolutions)}] Resolution: {resolution}x{resolution}")
        
        # 출력 파일명: 모델명_int8_inputres{resolution}.onnx
        output_filename = f"{base_name}_int8_inputres{resolution}.onnx"
        output_path = output_dir / output_filename
        
        # 이미 존재하는 경우 Skip
        if output_path.exists():
            print(f"  ✓ Skipped (already exists): {output_filename}")
            skip_count += 1
            continue
        
        print(f"  Output: {output_filename}")
        
        # 1. FP32 모델의 input shape를 해당 resolution으로 고정
        temp_fp32_path = temp_dir / f"temp_{base_name}_res{resolution}.onnx"
        success, input_name = fix_input_shape_to_resolution(
            base_model_path,
            temp_fp32_path,
            resolution
        )
        
        if not success:
            fail_count += 1
            continue
        
        print(f"    Input name: {input_name}")
        print(f"    Input shape: (1, 3, {resolution}, {resolution})")
        
        # 2. Calibration DataReader 생성 (해당 resolution에 맞게)
        try:
            calibration_reader = ResolutionAwareImageDataReader(
                calibration_image_folder=calibration_dir,
                input_name=input_name,
                target_resolution=resolution,
                num_samples=num_samples
            )
        except Exception as e:
            print(f"    ✗ Failed to create calibration reader: {e}")
            fail_count += 1
            temp_fp32_path.unlink(missing_ok=True)
            continue
        
        # 3. INT8 양자화 수행
        success = quantize_onnx_model(
            input_model_path=temp_fp32_path,
            output_model_path=output_path,
            calibration_data_reader=calibration_reader,
        )
        
        # 4. 임시 FP32 파일 삭제
        temp_fp32_path.unlink(missing_ok=True)
        
        if success:
            success_count += 1
        else:
            fail_count += 1
        
        print()
    
    # 임시 디렉토리 정리
    try:
        temp_dir.rmdir()
    except:
        pass
    
    return (success_count, fail_count, skip_count)


def main():
    import argparse
    
    # /tmp가 가득 찼을 경우를 대비해 임시 디렉토리를 SSD1로 변경
    os.environ['TMPDIR'] = '/SSD1/tmp'
    os.environ['TEMP'] = '/SSD1/tmp'
    os.environ['TMP'] = '/SSD1/tmp'
    Path('/SSD1/tmp').mkdir(exist_ok=True)
    
    parser = argparse.ArgumentParser(
        description='Input Resolution Sweep: Create INT8 ONNX models with fixed batch size'
    )
    
    parser.add_argument('--base_models_dir', type=str,
                       default='input_resolution_sweep_eval_onnxs',
                       help='Directory containing base FP32 ONNX models')
    
    parser.add_argument('--calibration_dir', type=str,
                       default='Calibration_Images_Classification',
                       help='Calibration images directory')
    
    parser.add_argument('--num_samples', type=int, default=1024,
                       help='Number of calibration samples (default: 1024)')
    
    parser.add_argument('--output_dir', type=str,
                       default='input_resolution_sweep_eval_onnxs_int8_batch_fixed',
                       help='Output directory for quantized models')
    
    parser.add_argument('--start_res', type=int, default=48,
                       help='Start resolution (default: 48)')
    
    parser.add_argument('--end_res', type=int, default=448,
                       help='End resolution (default: 448)')
    
    parser.add_argument('--step_res', type=int, default=16,
                       help='Resolution step (default: 16)')
    
    args = parser.parse_args()
    
    # 경로 설정
    base_models_dir = Path(args.base_models_dir)
    calibration_dir = Path(args.calibration_dir)
    output_dir = Path(args.output_dir)
    
    # Calibration 디렉토리 확인
    if not calibration_dir.exists():
        print(f"Error: Calibration directory not found: {calibration_dir}")
        sys.exit(1)
    
    # 베이스 모델 디렉토리 확인
    if not base_models_dir.exists():
        print(f"Error: Base models directory not found: {base_models_dir}")
        sys.exit(1)
    
    # 베이스 모델 수집
    base_models = sorted(base_models_dir.glob('*.onnx'))
    if not base_models:
        print(f"Error: No ONNX models found in {base_models_dir}")
        sys.exit(1)
    
    # Resolution 리스트 생성
    resolutions = list(range(args.start_res, args.end_res + 1, args.step_res))
    
    print("="*80)
    print("Input Resolution Sweep INT8 Quantization")
    print("="*80)
    print(f"Base models directory: {base_models_dir}")
    print(f"Output directory:      {output_dir}")
    print(f"Calibration directory: {calibration_dir}")
    print(f"Calibration samples:   {args.num_samples}")
    print(f"Resolutions:           {len(resolutions)} ({args.start_res}x{args.start_res} ~ {args.end_res}x{args.end_res}, step: {args.step_res})")
    print(f"Base models:           {len(base_models)}")
    for model in base_models:
        print(f"  - {model.name}")
    print(f"Total models to create: {len(base_models) * len(resolutions)}")
    print("="*80)
    
    # 전체 통계
    total_success = 0
    total_fail = 0
    total_skip = 0
    
    # 각 베이스 모델 처리
    for base_model in base_models:
        success, fail, skip = process_resolution_sweep(
            base_model_path=base_model,
            output_dir=output_dir,
            calibration_dir=calibration_dir,
            resolutions=resolutions,
            num_samples=args.num_samples
        )
        
        total_success += success
        total_fail += fail
        total_skip += skip
    
    # 최종 요약
    total_models = len(base_models) * len(resolutions)
    
    print("\n" + "="*80)
    print("Resolution Sweep Quantization Completed")
    print("="*80)
    print(f"Total models:         {total_models}")
    print(f"Successfully created: {total_success}")
    print(f"Skipped (exists):     {total_skip}")
    print(f"Failed:               {total_fail}")
    print(f"\nOutput directory: {output_dir}")
    print("="*80 + "\n")
    
    # 결과 파일 저장
    result_file = output_dir / "quantization_result.txt"
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("Input Resolution Sweep INT8 Quantization Results\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Base models directory: {base_models_dir}\n")
        f.write(f"Output directory:      {output_dir}\n")
        f.write(f"Calibration directory: {calibration_dir}\n")
        f.write(f"Calibration samples:   {args.num_samples}\n")
        f.write(f"Resolutions:           {len(resolutions)} ({args.start_res} ~ {args.end_res}, step: {args.step_res})\n")
        f.write(f"Base models:           {len(base_models)}\n\n")
        
        f.write(f"Total models:         {total_models}\n")
        f.write(f"Successfully created: {total_success}\n")
        f.write(f"Skipped (exists):     {total_skip}\n")
        f.write(f"Failed:               {total_fail}\n\n")
        
        f.write("="*80 + "\n")
        f.write("Generated Models by Resolution\n")
        f.write("="*80 + "\n\n")
        
        for base_model in base_models:
            base_name = base_model.stem.replace('_opset13', '')
            f.write(f"{base_model.name}:\n")
            for resolution in resolutions:
                output_filename = f"{base_name}_int8_inputres{resolution}.onnx"
                output_path = output_dir / output_filename
                if output_path.exists():
                    size_mb = output_path.stat().st_size / (1024 * 1024)
                    f.write(f"  ✓ {output_filename} ({size_mb:.2f} MB)\n")
                else:
                    f.write(f"  ✗ {output_filename} (not created)\n")
            f.write("\n")
    
    print(f"Results saved to: {result_file}\n")


if __name__ == "__main__":
    main()
