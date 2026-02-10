from pathlib import Path
from hailo_sdk_client import ClientRunner

input_dir = Path("classification_output_onnx")
output_dir = Path("classification_hars")

# “Although Hailo’s Model Zoo lists certain Vision Transformer (ViT) variants as supported for Hailo-8 devices, 
# in practice the general ONNX ViT export contains graph patterns (e.g., Gather on batch dimension from attention blocks) 
# that the current Hailo SDK parser does not support. Therefore, off-the-shelf ViT ONNX models 
# typically fail to compile on Hailo-8 without special modifications or Model Zoo configuration.”

# (hailo-compiler) mobilint@mobilint:~/jonghyun/hailo_compiler$ python parse_clasification_1.py 
# [UNSUPPORTED] model skipped: mobilenetv2_selu_w1_0_trained_opset13.onnx
# [UNSUPPORTED] model skipped: resnet18_selu_w1_0_trained_opset13.onnx
# [UNSUPPORTED] model skipped: vit_small_d24_trained_opset13.onnx

UNSUPPORTED_KEYWORDS = [
    #"mobilenetv2_selu_w1_0_trained_opset13", #UnsupportedOperationError: Selu operation is unsupported
    #"resnet18_selu_w1_0_trained_opset13", #UnsupportedOperationError: Selu operation is unsupported
    #"vit_small_d24_trained_opset13" #it looks that the compilatrion failed maybe it's a too big model for compilation
    'hf_vit_large_patch16_224_pretrained_opset14' #hailo_sdk_common.hailo_nn.exceptions.UnsupportedModelError: ew mult layer ew_mult1 (translated from /blocks/blocks.0/attn/Mul) expects 2 inputs but found 1
    # [RUN] classification_output_onnx/vit/vit_small_d24_trained_opset13.onnx
    # [info] Translation started on ONNX model vit_small_d24_trained_opset13
    # [warning] Large model detected. The graph may contain either a large number of operators, or weight variables with a very large capacity.
    # [warning] Translation time may be a bit long, and some features may be disabled (e.g. model augmentation, retry simplified model, onnx runtime hailo model extraction, etc.).
    # UnsupportedSliceLayerError in op /blocks.0/attn/Gather_3: Gather operation on the batch dimension is not supported.....
    # UnsupportedSliceLayerError in op /blocks.0/attn/Gather_3: Gather operation on the batch dimension is not supported....
]

target_folders = [
    # "output_onnx_huggingface",
    "output_onnx_torchvision",
    # "output_onnx_torchvision_regnet",
]

for folder in target_folders:
    folder_path = input_dir / folder
    if not folder_path.exists():
        print(f"[SKIP] folder not found: {folder_path}")
        continue

    har_dir = output_dir / folder
    har_dir.mkdir(parents=True, exist_ok=True)

    for onnx_path in folder_path.rglob("*.onnx"):
        onnx_model_name = onnx_path.stem
        har_path = har_dir / f"{onnx_model_name}_hailo_model.har"

        # 🚫 unsupported model skip
        if any(k in onnx_model_name for k in UNSUPPORTED_KEYWORDS):
            print(f"[UNSUPPORTED] model skipped: {onnx_path.name}")
            continue

        # ✅ 이미 생성된 HAR가 있으면 skip
        if har_path.exists():
            print(f"[SKIP] already exists: {har_path}")
            continue

        print(f"[RUN] {onnx_path}")

        runner = ClientRunner(hw_arch="hailo8")
        hn, npz = runner.translate_onnx_model(
            str(onnx_path),
            onnx_model_name,
            start_node_names=["input"],
            end_node_names=["output"],
            net_input_shapes={"input": [1, 3, 224, 224]},
        )

        runner.save_har(str(har_path))
        print(f"[DONE] saved: {har_path}")
