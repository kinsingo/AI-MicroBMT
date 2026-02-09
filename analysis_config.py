"""
=============================================================================
  Analysis Configuration  (analysis_config.py)
  ---------------------------------------------------------------------------
  All user-configurable settings for the analysis / visualization scripts.
  Edit ONLY this file when adapting the pipeline to a new benchmark run.
=============================================================================
"""

from pathlib import Path

# ============================================================================
# 1. HARDWARE / ACCELERATOR SETTINGS
# ============================================================================

# CPU baseline device name  (used as the reference for speedup & accuracy-drop)
BASELINE_DEVICE = 'Apple M4 CPU'

# Ordered list of ALL accelerators (CPU + NPUs)
ALL_ACCELERATORS = [
    'Apple M4 CPU',
    'Apple M4 ANE',
    'Mobilint-ARIES',
    'DeepX M1',
    'Hailo-8',
    'Qualcomm QCS6490',
    'RTX PRO 6000 Max-Q',
]

# NPU-only accelerators (automatically excludes BASELINE_DEVICE)
NPU_ACCELERATORS = [acc for acc in ALL_ACCELERATORS if acc != BASELINE_DEVICE]

# Colour palette – consistent across every chart
ACCELERATOR_COLORS = {
    'Hailo-8':              '#1f77b4',   # Blue
    'DeepX M1':             '#ff7f0e',   # Orange
    'Mobilint-ARIES':       '#2ca02c',   # Green
    'Apple M4 ANE':         '#d62728',   # Red
    'Apple M4 CPU':         '#9467bd',   # Purple
    'Qualcomm QCS6490':     '#8c564b',   # Brown
    'RTX PRO 6000 Max-Q':  '#e377c2',   # Pink
}

# ============================================================================
# 2. DATA FOLDER PATHS  (relative to the project root)
# ============================================================================

# Base model variant folders  (single-stream & offline CSVs)
# key → folder name;  CSV filenames are derived as
#   "<folder>/<folder> single-stream results.csv"
#   "<folder>/<folder> offline results.csv"
DATA_FOLDERS = {
    'mobilenet':    'mobilenet variant',
    'resnet':       'resnet variant',
    'convnext':     'convnext variant',
    'resnext50':    'resnext50 variant',
    'vit':          'vit variant',
    'regnet':       'regnet variant',
    'shufflenet':   'shufflenet variant',
    'densenet':     'densenet variant',
}

# Activation variant folders (same CSV naming convention)
ACTIVATION_VARIANT_FOLDERS = {
    'mobilenet_activation':  'mobilenet activation variant',
    'resnet_activation':     'resnet activation variant',
}

# Input resolution variant folder
INPUT_RESOLUTION_FOLDER = 'input resolution variant'

# ============================================================================
# 3. OUTPUT DIRECTORIES
# ============================================================================

BASE_OUTPUT_DIR = Path('analysis_charts')

# Sub-folder names (created automatically by each script)
OUTPUT_SUBDIR_ACTIVATION_SWEEP         = 'activation_sweep'
OUTPUT_SUBDIR_SINGLESTREAM_VS_OFFLINE  = 'singleStream_vs_offline'
OUTPUT_SUBDIR_INPUT_RES_SINGLESTREAM   = 'inputResolution_singleStream'
OUTPUT_SUBDIR_INPUT_RES_OFFLINE        = 'inputResolution_offline'

# ============================================================================
# 4. MODEL NAME NORMALISATION
# ============================================================================

# Suffixes stripped from raw benchmark_model names to create a canonical key.
# Add / remove entries here when on-boarding a new accelerator toolchain.
MODEL_NAME_STRIP_SUFFIXES = [
    '_bgr2rgb_normalized_quantized_model_compiled',
    '_trained_opset13',
    '_pretrained_opset13',
    '_pretrained_opset14',
]

# Width-multiplier tokens that can appear in a model name
WIDTH_MULTIPLIER_MAP = {
    'w0_25': 0.25,
    'w0_5':  0.5,
    'w0_75': 0.75,
    'w1_0':  1.0,
    'w1_5':  1.5,
    'w2_0':  2.0,
}

# ============================================================================
# 5. ACTIVATION SWEEP SETTINGS
# ============================================================================

# Activation functions tested in the sweep
ACTIVATION_NAMES = [
    'elu', 'gelu', 'hardswish', 'leakyrelu', 'mish',
    'prelu', 'relu', 'relu6', 'selu', 'silu',
]

# Recommendation thresholds (accuracy degradation %)
ACTIVATION_REC_CRITICAL_ACC  = 10.0   # absolute accuracy < this → "Not Recommended"
ACTIVATION_REC_HIGH_DEG      = 15.0   # avg degradation % > this → "Problematic"
ACTIVATION_REC_MOD_DEG       =  5.0   # avg degradation % > this → "Acceptable"

# ============================================================================
# 6. INPUT RESOLUTION ANALYSIS SETTINGS
# ============================================================================

# Model families to analyse in the resolution sweep
RESOLUTION_MODEL_FAMILIES = ['MobileNetV2', 'ResNet-50', 'ResNet-101']

# Reference resolution for scaling-factor calculations (pixels)
BASELINE_RESOLUTION = 224

# Key resolutions printed in summary tables
KEY_RESOLUTIONS = [112, 160, 224, 320, 448]

# ============================================================================
# 7. CASE CLASSIFICATION (generate_cases_analysis.py)
# ============================================================================

# NPUs used for case-classification analysis
CASE_ANALYSIS_NPUS = [
    'DeepX M1',
    'Mobilint-ARIES',
    'Hailo-8',
    'Qualcomm QCS6490',
    'Apple M4 ANE',
    'RTX PRO 6000 Max-Q',
]

# Thresholds
CASE_TAU_ACCURACY_DROP_PCT   = 4.0    # below this → "accurate"
CASE_MIN_ABSOLUTE_ACCURACY   = 15.0   # below this → Case 4 (failure)
CASE_TOTAL_MODELS            = 67     # used in LaTeX table denominator

# ============================================================================
# 8. MATPLOTLIB / SEABORN DEFAULTS
# ============================================================================

MPL_STYLE          = 'seaborn-v0_8-paper'
MPL_STYLE_FALLBACK = 'seaborn-paper'
SNS_PALETTE        = 'husl'
