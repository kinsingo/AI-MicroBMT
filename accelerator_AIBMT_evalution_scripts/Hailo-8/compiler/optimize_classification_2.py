import os
import traceback
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.python.eager.context import eager_mode

# Model: hf_vit_base_patch8_224_pretrained_opset14
# [ERROR] Failed to optimize hf_vit_base_patch8_224_pretrained_opset14: 
# failed to initialize intent(inout) array -- expected elsize=8 but got 4

# Model: convnext_large_pretrained_opset13
# HAR: classification_hars/output_onnx_torchvision/convnext_large_pretrained_opset13_hailo_model.har
# [ERROR] Failed to optimize convnext_large_pretrained_opset13: Subprocess failed with traceback

# import the hailo sdk client relevant classes
from hailo_sdk_client import ClientRunner, InferenceContext
print("[DEBUG] optimize_classification_2.py START")

# Initialize error log file
# with open("optimize_error_log.txt", "w") as f:
#     f.write(f"Optimization Error Log - {os.popen('date').read()}")
#     f.write(f"{'='*80}\n\n")

images_path = "classification_calibration_images"
images_list = [img_name for img_name in os.listdir(images_path) if os.path.splitext(img_name)[1] in [".jpg", ".jpeg"]]
calib_dataset = np.zeros((len(images_list), 224, 224, 3))
for idx, img_name in enumerate(sorted(images_list)):
    img = np.array(Image.open(os.path.join(images_path, img_name)))
    assert img.shape == (224, 224, 3), f"{img_name} has unexpected shape {img.shape}"
    calib_dataset[idx] = img

# Create ObjectDetection_quantized_hars folder if it doesn't exist
os.makedirs("classification_quantized_hars", exist_ok=True)

target_folders = [
    # "output_onnx_huggingface",
    "output_onnx_torchvision",
    # "output_onnx_torchvision_regnet",
]

for folder in target_folders:
    input_folder = os.path.join("classification_hars", folder)
    if not os.path.isdir(input_folder):
        print(f"[SKIP] folder not found: {input_folder}")
        continue

    har_files = [f for f in os.listdir(input_folder) if f.endswith(".har")]

    for har_file in har_files:
        model_name = har_file.replace("_hailo_model.har", "")
        hailo_model_har_name = os.path.join(input_folder, har_file)        
        print(f"[INFO] Processing {model_name}...")
        
        try:
            output_folder = os.path.join("classification_quantized_hars", folder)
            os.makedirs(output_folder, exist_ok=True)
            quantized_model_har_path = os.path.join(
                output_folder,
                f"{model_name}_bgr2rgb_normalized_quantized_model.har"
            )

            if os.path.exists(quantized_model_har_path):
                print(f"[INFO]{quantized_model_har_path} is existing.. continue..")
                continue

            # Default For GPU : (optimization_level=2,compression_level=1).
            # Default For CPU : (optimization_level=0,compression_level=0).
            runner = ClientRunner(har=hailo_model_har_name)

            
            # Load the model script to ClientRunner so it will be considered on optimization
            # alls = (
            #     "model_optimization_flavor(optimization_level=2,compression_level=2,batch_size=2)\n"
            #     #"performance_param(compiler_optimization_level=max)\n" --> Compile 시간이 엄청 늘어남, 일반적 사용 X
            #     "model_optimization_config(calibration, calibset_size=1024)\n"
            #     "normalization1=normalization([123.675,116.28,103.53],[58.395,57.12,57.375])\n"
            # )

            safe_model_name = "".join(c if c.isalnum() else "_" for c in model_name)
            norm_name = f"normalization_{safe_model_name}"
            alls = (
                "model_optimization_flavor(optimization_level=2,compression_level=2,batch_size=2)\n"
                f"model_optimization_config(calibration, calibset_size={len(images_list)})\n"
                f"{norm_name}=normalization([123.675,116.28,103.53],[58.395,57.12,57.375])\n"
            )


            runner.load_model_script(alls)
            # Call Optimize to perform the optimization process
            runner.optimize(calib_dataset) 
            # Save the result state to a Quantized HAR file
            runner.save_har(quantized_model_har_path)
            print(f"[SUCCESS] {model_name} optimized successfully")
        except Exception as e:
            error_msg = f"[ERROR] Failed to optimize {model_name}: {str(e)}\n"
            error_msg += f"Traceback:\n{traceback.format_exc()}\n"
            print(error_msg)
            
            # Log to file
            with open("optimize_error_log.txt", "a") as f:
                f.write(f"{'='*80}\n")
                f.write(f"Model: {model_name}\n")
                f.write(f"HAR: {hailo_model_har_name}\n")
                f.write(error_msg)
                f.write(f"{'='*80}\n\n")
            
            continue