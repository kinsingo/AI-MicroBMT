#!/bin/bash

#./dx_com_M1_v2.1.0-rc.4/dx_com -m microbmt_onnx_models_2.1.0_ver2/output_onnx_torchvision/densenet201_pretrained_opset13.onnx -c classification.json -o microbmt_dnnx_models_2.1.0_ver2/output_onnx_torchvision/densenet201_pretrained_opset13.dnnx

# Base directories
INPUT_DIR="microbmt_onnx_models_2.1.0_ver2"
OUTPUT_DIR="microbmt_dnnx_models_2.1.0_ver2"
DX_COM="./dx_com_M1_v2.1.0-rc.4/dx_com"
CONFIG="classification.json"

# # Find all subdirectories in the input directory
for subdir in "$INPUT_DIR"/*/; do
    # Get the subdirectory name (e.g., "vit", "resnet18_activation")
    subdir_name=$(basename "$subdir")
    
    # Create corresponding output subdirectory if it doesn't exist
    mkdir -p "$OUTPUT_DIR/$subdir_name"
    
    echo "Processing subdirectory: $subdir_name"
    
    # Find all .onnx files in the current subdirectory
    find "$subdir" -maxdepth 1 -type f -name "*.onnx" | while read -r onnx_file; do
        # Get the filename without path
        filename=$(basename "$onnx_file")
        
        # Create output filename (.onnx -> .dnnx)
        output_file="$OUTPUT_DIR/$subdir_name/${filename%.onnx}.dnnx"
        
        # Skip if output file already exists
        if [ -f "$output_file" ]; then
            echo "  ⊘ Skipping (already exists): $filename"
            continue
        fi
        
        echo "  Converting: $filename"
        
        # Run the dx_com command
        "$DX_COM" \
            -m "$onnx_file" \
            -c "$CONFIG" \
            -o "$output_file"
        
        if [ $? -eq 0 ]; then
            echo "  ✓ Successfully converted: $filename"
        else
            echo "  ✗ Failed to convert: $filename"
        fi
    done
    
    echo ""
done

echo "All conversions completed!"