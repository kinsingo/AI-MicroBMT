# from hailo_sdk_client import ClientRunner
# import os

# os.makedirs("classification_compiled_hefs", exist_ok=True)

# for quantized_model_har_name in os.listdir("classification_quantized_hars"):
#     if not quantized_model_har_name.endswith(".har"):
#         continue

#     har_path = os.path.join("classification_quantized_hars", quantized_model_har_name)
#     print(f"[INFO] Compiling {har_path} ...")

#     try:
#         runner = ClientRunner(har=har_path)
#         hef = runner.compile()
#     except Exception as e:
#         print(f"[ERROR] Failed to compile {quantized_model_har_name}: {e}")
#         # 여기서 로그만 남기고 다음 모델로 넘어감
#         continue

#     base_name = os.path.splitext(quantized_model_har_name)[0]
#     out_path = f"classification_compiled_hefs/{base_name}_compiled.hef"
#     with open(out_path, "wb") as f:
#         f.write(hef)

#     print(f"[SUCCESS] Saved {out_path}")

from hailo_sdk_client import ClientRunner
import os
import traceback

# Initialize error log file
with open("compile_error_log.txt", "w") as f:
    f.write(f"Compilation Error Log - {os.popen('date').read()}")
    f.write(f"{'='*80}\n\n")

input_root = "classification_quantized_hars"
output_root = "classification_compiled_hefs"

os.makedirs(output_root, exist_ok=True)

for root, _, files in os.walk(input_root):
    for file in files:
        if not file.endswith(".har"):
            continue

        har_path = os.path.join(root, file)
        rel_dir = os.path.relpath(root, input_root)  # subfolder 경로
        out_dir = os.path.join(output_root, rel_dir)
        os.makedirs(out_dir, exist_ok=True)

        print(f"[INFO] Compiling {har_path} ...")

        base_name = os.path.splitext(file)[0]
        out_path = os.path.join(out_dir, f"{base_name}_compiled.hef")

        if os.path.exists(out_path):
            print(f"[SKIP] already compiled: {out_path}")
            continue

        try:
            runner = ClientRunner(har=har_path)
            hef = runner.compile()
        except Exception as e:
            error_msg = f"[ERROR] Failed to compile {har_path}: {str(e)}\n"
            error_msg += f"Traceback:\n{traceback.format_exc()}\n"
            print(error_msg)
            
            # Log to file
            with open("compile_error_log.txt", "a") as f:
                f.write(f"{'='*80}\n")
                f.write(f"HAR: {har_path}\n")
                f.write(f"Output: {out_path}\n")
                f.write(error_msg)
                f.write(f"{'='*80}\n\n")
            
            continue


        with open(out_path, "wb") as f:
            f.write(hef)

        print(f"[SUCCESS] Saved {out_path}")
