"""
Utility functions for MLPerf benchmark data analysis.
Common data loading and preprocessing routines.

All user-configurable settings (colours, folder paths, device names, etc.)
live in ``analysis_config.py``.  Edit that file — not this one — when
adapting the pipeline to a new benchmark run.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import ALL settings from the central config
from analysis_config import (
    BASELINE_DEVICE,
    ALL_ACCELERATORS,
    NPU_ACCELERATORS,
    ACCELERATOR_COLORS,
    DATA_FOLDERS,
    ACTIVATION_VARIANT_FOLDERS,
    MODEL_NAME_STRIP_SUFFIXES,
    WIDTH_MULTIPLIER_MAP,
    BASE_OUTPUT_DIR,
    MPL_STYLE,
    MPL_STYLE_FALLBACK,
    SNS_PALETTE,
)

# Set style for publication-quality figures
try:
    plt.style.use(MPL_STYLE)
except OSError:
    plt.style.use(MPL_STYLE_FALLBACK)
sns.set_palette(SNS_PALETTE)

# Base output directory (subfolders created by each script)
base_output_dir = BASE_OUTPUT_DIR
base_output_dir.mkdir(exist_ok=True)

# ============================================================================
# Model Name Processing Functions
# ============================================================================
def normalize_model_name(model_name):
    """Normalize model name to match across different accelerators.

    Strips the suffixes listed in ``analysis_config.MODEL_NAME_STRIP_SUFFIXES``.
    """
    normalized = model_name
    for suffix in MODEL_NAME_STRIP_SUFFIXES:
        normalized = normalized.replace(suffix, '')
    return normalized

def extract_width_multiplier(model_name):
    """Extract width multiplier (e.g., w0_25 → 0.25) from *model_name*.

    The recognised tokens are defined in ``analysis_config.WIDTH_MULTIPLIER_MAP``.
    """
    for token, value in WIDTH_MULTIPLIER_MAP.items():
        if token in model_name:
            return value
    return None

def extract_model_family(model_name):
    """Extract base model family (e.g., mobilenetv2, resnet50)"""
    if 'mobilenet' in model_name:
        return 'MobileNetV2'
    elif 'convnext' in model_name:
        if 'tiny' in model_name:
            return 'ConvNeXt-Tiny'
        elif 'small' in model_name:
            return 'ConvNeXt-Small'
        elif 'base' in model_name:
            return 'ConvNeXt-Base'
        return 'ConvNeXt'
    elif 'resnext50' in model_name:
        if 'c64' in model_name:
            return 'ResNeXt50-C64'
        elif 'c32' in model_name:
            return 'ResNeXt50-C32'
        elif 'c16' in model_name:
            return 'ResNeXt50-C16'
        elif 'c8' in model_name:
            return 'ResNeXt50-C8'
        return 'ResNeXt50'
    elif 'regnet' in model_name:
        if 'regnet_x_400mf' in model_name:
            return 'RegNetX-400MF'
        elif 'regnet_x_800mf' in model_name:
            return 'RegNetX-800MF'
        elif 'regnet_x_1_6gf' in model_name:
            return 'RegNetX-1.6GF'
        elif 'regnet_x_3_2gf' in model_name:
            return 'RegNetX-3.2GF'
        elif 'regnet_x_8gf' in model_name:
            return 'RegNetX-8GF'
        elif 'regnet_y_400mf' in model_name:
            return 'RegNetY-400MF'
        elif 'regnet_y_800mf' in model_name:
            return 'RegNetY-800MF'
        elif 'regnet_y_1_6gf' in model_name:
            return 'RegNetY-1.6GF'
        elif 'regnet_y_3_2gf' in model_name:
            return 'RegNetY-3.2GF'
        elif 'regnet_y_8gf' in model_name:
            return 'RegNetY-8GF'
        return 'RegNet'
    elif 'shufflenet' in model_name:
        if 'x1_0' in model_name:
            return 'ShuffleNetV2-x1.0'
        elif 'x1_5' in model_name:
            return 'ShuffleNetV2-x1.5'
        elif 'x2_0' in model_name:
            return 'ShuffleNetV2-x2.0'
        return 'ShuffleNetV2'
    elif 'densenet' in model_name:
        if 'densenet121' in model_name:
            return 'DenseNet-121'
        elif 'densenet169' in model_name:
            return 'DenseNet-169'
        elif 'densenet201' in model_name:
            return 'DenseNet-201'
        return 'DenseNet'
    elif 'vit' in model_name:
        if 'tiny' in model_name:
            return 'ViT-Tiny'
        elif 'small' in model_name and 'p32' in model_name:
            return 'ViT-Small-P32'
        elif 'small' in model_name and 'p8' in model_name:
            return 'ViT-Small-P8'
        elif 'small' in model_name and 'd6' in model_name:
            return 'ViT-Small-D6'
        elif 'small' in model_name and 'd24' in model_name:
            return 'ViT-Small-D24'
        elif 'small' in model_name:
            return 'ViT-Small'
        elif 'base' in model_name:
            return 'ViT-Base'
        return 'ViT'
    elif 'resnet101' in model_name:
        return 'ResNet-101'
    elif 'resnet50' in model_name:
        return 'ResNet-50'
    elif 'resnet34' in model_name:
        return 'ResNet-34'
    elif 'resnet18' in model_name:
        return 'ResNet-18'
    elif 'resnet10' in model_name:
        return 'ResNet-10'
    return 'Unknown'

# ============================================================================
# Data Loading Functions
# ============================================================================
def load_single_stream_data():
    """Load and preprocess single-stream latency data for all model variants.

    Folder paths come from ``analysis_config.DATA_FOLDERS`` and
    ``analysis_config.ACTIVATION_VARIANT_FOLDERS``.
    """
    dfs = []
    for folder in DATA_FOLDERS.values():
        dfs.append(pd.read_csv(f'{folder}/{folder} single-stream results.csv'))
    for folder in ACTIVATION_VARIANT_FOLDERS.values():
        dfs.append(pd.read_csv(f'{folder}/{folder} single-stream results.csv'))
    df = pd.concat(dfs, ignore_index=True)
    
    # Add preprocessing
    df['normalized_model'] = df['benchmark_model'].apply(normalize_model_name)
    df['width_multiplier'] = df['benchmark_model'].apply(extract_width_multiplier)
    df['model_family'] = df['benchmark_model'].apply(extract_model_family)
    
    # Get baseline accuracy from the CPU baseline device
    baseline_df = df[df['accelerator_type'] == BASELINE_DEVICE][['normalized_model', 'accuracy']].copy()
    baseline_df.columns = ['normalized_model', 'baseline_accuracy']
    
    # Merge baseline accuracy
    df = df.merge(baseline_df, on='normalized_model', how='left')
    
    # Calculate accuracy drop
    df['accuracy_drop'] = df['baseline_accuracy'] - df['accuracy']
    df['accuracy_drop_percent'] = (df['accuracy_drop'] / df['baseline_accuracy']) * 100
    
    # Convert latency
    df['latency_ms'] = df['sample_latency_average']
    
    print("[OK] Single-stream data loaded and preprocessed")
    print(f"Accelerator types: {df['accelerator_type'].unique()}")
    print(f"Model families: {df['model_family'].unique()}")
    print(f"Total data points: {len(df)}\n")
    
    return df

def load_offline_data():
    """Load and preprocess offline throughput data for all model variants.

    Folder paths come from ``analysis_config.DATA_FOLDERS`` and
    ``analysis_config.ACTIVATION_VARIANT_FOLDERS``.
    """
    dfs = []
    for folder in DATA_FOLDERS.values():
        dfs.append(pd.read_csv(f'{folder}/{folder} offline results.csv'))
    for folder in ACTIVATION_VARIANT_FOLDERS.values():
        dfs.append(pd.read_csv(f'{folder}/{folder} offline results.csv'))
    df = pd.concat(dfs, ignore_index=True)
    
    # Add preprocessing
    df['normalized_model'] = df['benchmark_model'].apply(normalize_model_name)
    df['width_multiplier'] = df['benchmark_model'].apply(extract_width_multiplier)
    df['model_family'] = df['benchmark_model'].apply(extract_model_family)
    
    print("[OK] Offline data loaded and preprocessed")
    print(f"Accelerator types: {df['accelerator_type'].unique()}")
    print(f"Model families: {df['model_family'].unique()}")
    print(f"Total data points: {len(df)}\n")
    
    return df
