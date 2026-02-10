import os
# from qubee import mxq_compile
from qbcompiler import mxq_compile

#[⣾] Parsing deep learning model....[2026-01-27 00:07:13] qubee.model_dict.parser.parser : [INFO] hf_vit_base_patch8_224_pretrained_opset14 model's number of operations 156.8552 GOPS, 78.4276 MACs
# ============================================================
# ❌ Failed Models:
# ============================================================
#   • convnext_large_pretrained_opset13.onnx (from output_onnx_torchvision)
#     Error: [EXCEPTION] Try to put dummy dma to size != 0 bundle (c45b115)

if __name__ == "__main__":
    # Directories
    onnx_base_dir = "./microBMT_67_onnx_models"
    mxq_output_dir = "./microBMT_67_mxq_models_compiler_1.0.1_modified"
    calib_dir = "./Calibaration_Images_npy"
    calib_data_path = os.path.join(calib_dir, "classification_calibrationDataset")
    
    # Create output directory if it doesn't exist
    os.makedirs(mxq_output_dir, exist_ok=True)
    
    # Check if calibration dataset exists
    if not os.path.exists(calib_data_path):
        print(f"❌ Error: Calibration dataset not found at {calib_data_path}")
        print(f"   Please ensure the calibration dataset is available")
        exit(1)
    
    # Collect all ONNX models
    model_list = []
    for filename in os.listdir(onnx_base_dir):
        if filename.endswith(".onnx"):
            model_list.append(filename)
    
    print(f"Found {len(model_list)} ONNX models to compile")
    print(f"Calibration dataset: {calib_data_path}")
    print(f"Output directory: {mxq_output_dir}")
    print(f"\n{'='*60}")
    
    # Compile each model
    compiled_count = 0
    skipped_count = 0
    failed_count = 0
    failed_models = []
    
    for model_name in model_list:
        onnx_path = os.path.join(onnx_base_dir, model_name)
        mxq_name = model_name.replace(".onnx", ".mxq")
        mxq_path = os.path.join(mxq_output_dir, mxq_name)
        
        # Skip if already exists
        if os.path.exists(mxq_path):
            print(f"⏭️  Skipping {model_name} - MXQ already exists")
            skipped_count += 1
            continue
        
        print(f"\n{'='*60}")
        print(f"Compiling: {model_name}")
        print(f"ONNX path: {onnx_path}")
        print(f"Output path: {mxq_path}")
        print(f"{'='*60}")
        
        try:
            # mxq_compile(
            #     model=onnx_path,
            #     calib_data_path=calib_data_path,
            #     quantization_mode=2,#quantize_method="maxpercentile",  # quantization method to use
            #     is_quant_ch=True,  # whether to use channel-wise quantization
            #     percentile=0.9999,#quantize_percentile=0.9999,
            #     topk_ratio=0.01,
            #     quantization_output = 0, #quant_output="layer", 
            #     save_path=mxq_path,
            #     backend="onnx",
            #     inference_scheme="single",
            #     optimize_option=2,
            # )

            #qb compiler (v1.0.1, Released on 26.02.04)
            # quantization_mode (int): Quantization mode index
            # 0: max
            # 1: maxPercentile
            # 2: histogram
            #  By default, this option is set to 1 (maxPercentile).
            mxq_compile(
                model=onnx_path,
                calib_data_path=calib_data_path,
                quantization_mode=1, # now the default 1 is "maxpercentile"
                is_quant_ch=True,  
                percentile=0.9999,
                topk_ratio=0.01,
                quantization_output = 0, #for classification
                save_path=mxq_path,
                backend="onnx",
                inference_scheme="single",
                optimize_option=2,
                act_mixed_precision=True, #for high accuracy
                device="gpu", # "cpu" or "gpu"
            )
            print(f"✅ Successfully compiled: {mxq_name}")
            compiled_count += 1
        except Exception as e:
            print(f"❌ Error compiling {model_name}: {str(e)}")
            failed_count += 1
            failed_models.append((model_name, str(e)))
    
    print(f"\n{'='*60}")
    print("✅ Compilation process completed!")
    print(f"{'='*60}")
    print(f"Total models found: {len(model_list)}")
    print(f"Successfully compiled: {compiled_count}")
    print(f"Skipped (already exists): {skipped_count}")
    print(f"Failed: {failed_count}")
    
    if failed_models:
        print(f"\n{'='*60}")
        print("❌ Failed Models:")
        print(f"{'='*60}")
        for model_name, error in failed_models:
            print(f"  • {model_name}")
            print(f"    Error: {error}")
    
    print(f"{'='*60}")
