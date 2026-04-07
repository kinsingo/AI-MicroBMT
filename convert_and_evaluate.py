"""
=============================================================================
  Convert & Evaluate  (convert_and_evaluate.py)
  ---------------------------------------------------------------------------
  Converts raw AI-BMT evaluation CSV (newEvalResults.csv) into the variant
  folder CSV format required by the analysis pipeline, then prints a quick
  evaluation summary and pipeline compatibility diagnostics.

  This script is the recommended entry point for integrating new device
  evaluation data into the AI-MicroBMT analysis pipeline.
  
  Usage:
      python convert_and_evaluate.py
      python convert_and_evaluate.py --input myResults.csv
      python convert_and_evaluate.py --input myResults.csv --device-name "My Device"
      python convert_and_evaluate.py --append   # append to existing variant CSVs
=============================================================================
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# ============================================================================
# 1. USER-CONFIGURABLE PARAMETERS
# ============================================================================

# Default input CSV (exported from AI-BMT platform)
DEFAULT_INPUT_CSV = 'newEvalResults.csv'

# Suffixes stripped from raw benchmark_model names to normalize
MODEL_NAME_STRIP_SUFFIXES = [
    '_bgr2rgb_normalized_quantized_model_compiled',
    '_trained_opset13',
    '_pretrained_opset13',
    '_pretrained_opset14',
]

# Activation function names used in the benchmark
ACTIVATION_NAMES = [
    'elu', 'gelu', 'hardswish', 'leakyrelu', 'mish',
    'prelu', 'relu', 'relu6', 'selu', 'silu',
]

# Model family detection rules:
#   key = variant folder name (without " variant")
#   value = regex pattern to match normalized benchmark_model name
#
# Order matters: more specific patterns first, then general ones.
# Activation variants are detected separately.
MODEL_FAMILY_PATTERNS = [
    ('shufflenet',  r'^shufflenet'),
    ('resnext50',   r'^resnext50'),
    ('convnext',    r'^convnext'),
    ('densenet',    r'^densenet'),
    ('regnet',      r'^regnet'),
    ('vit',         r'^hf_vit'),
    ('resnet',      r'^resnet'),
    ('mobilenet',   r'^mobilenetv2'),
]

# Input resolution pattern in model name (e.g., "inputres_224" or "inputResolution_224")
INPUT_RES_PATTERN = re.compile(r'(?:inputres|inputResolution)[_:]?(\d+)', re.IGNORECASE)

# ============================================================================
# 2. COLUMN MAPPING  (raw CSV → variant CSV)
# ============================================================================

# Columns in raw CSV → columns in variant CSV
RAW_COL_MAP = {
    'task':                         'task',
    'scenario':                     'scenario',
    'accuracy (%)':                 'accuracy',
    'sample_latency_average (ms)':  'sample_latency_average',
    'samples_per_second (FPS)':     'samples_per_second',
    'accelerator_type':             'accelerator_type',
    'benchmark_model':              'benchmark_model',
}

# Output column order per scenario
COLS_SINGLE_STREAM = ['task', 'scenario', 'accuracy', 'sample_latency_average',
                      'accelerator_type', 'benchmark_model']
COLS_OFFLINE       = ['task', 'scenario', 'accuracy', 'samples_per_second',
                      'accelerator_type', 'benchmark_model']
COLS_RES_SS        = ['task', 'scenario', 'accuracy', 'sample_latency_average',
                      'accelerator_type', 'input_resolution', 'benchmark_model']
COLS_RES_OFF       = ['task', 'scenario', 'accuracy', 'samples_per_second',
                      'accelerator_type', 'input_resolution', 'benchmark_model']

# ============================================================================
# 3. VARIANT FOLDER DEFINITIONS
# ============================================================================

# Base model variant folders
BASE_VARIANT_FOLDERS = {
    'mobilenet':    'mobilenet variant',
    'resnet':       'resnet variant',
    'convnext':     'convnext variant',
    'resnext50':    'resnext50 variant',
    'vit':          'vit variant',
    'regnet':       'regnet variant',
    'shufflenet':   'shufflenet variant',
    'densenet':     'densenet variant',
}

# Activation variant folders
ACTIVATION_VARIANT_FOLDERS = {
    'mobilenet':  'mobilenet activation variant',
    'resnet':     'resnet activation variant',
}

# Input resolution variant folder
INPUT_RESOLUTION_FOLDER = 'input resolution variant'


# ============================================================================
# 4. CORE FUNCTIONS
# ============================================================================

def normalize_model_name(model_name):
    """Strip known suffixes to get canonical model name."""
    normalized = model_name.strip()
    for suffix in MODEL_NAME_STRIP_SUFFIXES:
        normalized = normalized.replace(suffix, '')
    return normalized


def detect_activation(normalized_name):
    """Check if model name contains an activation function variant.
    
    Returns (activation_name, base_family) or (None, None).
    Activation variants follow the pattern: {base}_{activation}_w{multiplier}
    e.g. resnet18_silu_w1_0, mobilenetv2_gelu_w1_0
    """
    for act in ACTIVATION_NAMES:
        # Match pattern like: resnet18_{act}_w  or  mobilenetv2_{act}_w
        pattern = rf'(resnet\d+|mobilenetv2)_{act}_w'
        if re.search(pattern, normalized_name):
            if 'resnet' in normalized_name:
                return act, 'resnet'
            elif 'mobilenet' in normalized_name:
                return act, 'mobilenet'
    return None, None


def detect_input_resolution(normalized_name):
    """Extract input resolution from model name if present."""
    m = INPUT_RES_PATTERN.search(normalized_name)
    if m:
        return int(m.group(1))
    return None


def detect_model_family(normalized_name):
    """Detect which model family a normalized name belongs to."""
    for family, pattern in MODEL_FAMILY_PATTERNS:
        if re.search(pattern, normalized_name):
            return family
    return None


def classify_model(normalized_name):
    """Classify a model into (category, family).
    
    Returns:
        category: 'activation', 'input_resolution', or 'base'
        family: model family string (e.g., 'mobilenet', 'resnet')
    """
    # 1. Check for input resolution variant
    res = detect_input_resolution(normalized_name)
    if res is not None:
        family = detect_model_family(normalized_name)
        return 'input_resolution', family
    
    # 2. Check for activation variant
    act, act_family = detect_activation(normalized_name)
    if act is not None:
        return 'activation', act_family
    
    # 3. Base model variant
    family = detect_model_family(normalized_name)
    return 'base', family


def load_raw_csv(input_path):
    """Load and clean the raw AI-BMT evaluation CSV."""
    df = pd.read_csv(input_path)
    
    # Drop empty rows (the CSV often has trailing empty rows)
    df = df.dropna(subset=['benchmark_model'])
    df = df[df['benchmark_model'].str.strip() != '']
    
    # Rename columns
    rename_map = {}
    for raw_col, new_col in RAW_COL_MAP.items():
        if raw_col in df.columns and raw_col != new_col:
            rename_map[raw_col] = new_col
    df = df.rename(columns=rename_map)
    
    # Ensure numeric types
    for col in ['accuracy', 'sample_latency_average', 'samples_per_second']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Normalize device names for pipeline compatibility
    if 'accelerator_type' in df.columns:
        df['accelerator_type'] = df['accelerator_type'].apply(normalize_device_name)
    
    return df


def apply_device_name_override(df, device_name):
    """Override accelerator_type with user-specified device name."""
    if device_name:
        df['accelerator_type'] = device_name
    return df


def process_and_split(df):
    """Process dataframe: normalize names, classify, split into groups.
    
    Returns dict of {(category, family, scenario): DataFrame}
    """
    results = {}
    
    for _, row in df.iterrows():
        raw_model = row['benchmark_model']
        normalized = normalize_model_name(raw_model)
        scenario = row['scenario']
        
        category, family = classify_model(normalized)
        
        if family is None:
            print(f"  [WARNING] Cannot classify model: {raw_model} → skipped")
            continue
        
        key = (category, family, scenario)
        if key not in results:
            results[key] = []
        
        row_data = row.copy()
        row_data['benchmark_model'] = normalized
        
        # For input resolution, extract and add column
        if category == 'input_resolution':
            res = detect_input_resolution(normalized)
            row_data['input_resolution'] = f'inputResolution:{res}'
        
        results[key].append(row_data)
    
    # Convert lists to DataFrames
    for key in results:
        results[key] = pd.DataFrame(results[key])
    
    return results


def get_output_path(category, family, scenario):
    """Get output file path for a given category/family/scenario."""
    if category == 'input_resolution':
        folder = INPUT_RESOLUTION_FOLDER
        if scenario == 'Single-Stream':
            filename = 'input_variant_singleStream.csv'
        else:
            filename = 'input_variant_offline.csv'
    elif category == 'activation':
        folder = ACTIVATION_VARIANT_FOLDERS.get(family)
        if folder is None:
            return None
        if scenario == 'Single-Stream':
            filename = f'{folder} single-stream results.csv'
        else:
            filename = f'{folder} offline results.csv'
    else:  # base
        folder = BASE_VARIANT_FOLDERS.get(family)
        if folder is None:
            return None
        if scenario == 'Single-Stream':
            filename = f'{folder} single-stream results.csv'
        else:
            filename = f'{folder} offline results.csv'
    
    return Path(folder) / filename


def get_output_columns(category, scenario):
    """Get the correct column list for output."""
    if category == 'input_resolution':
        return COLS_RES_SS if scenario == 'Single-Stream' else COLS_RES_OFF
    elif scenario == 'Single-Stream':
        return COLS_SINGLE_STREAM
    else:
        return COLS_OFFLINE


def save_results(results, base_dir, append=False):
    """Save processed results to variant CSV files."""
    saved_files = []
    
    for (category, family, scenario), data in results.items():
        out_path = get_output_path(category, family, scenario)
        if out_path is None:
            print(f"  [WARNING] No output folder for ({category}, {family}) → skipped")
            continue
        
        full_path = base_dir / out_path
        columns = get_output_columns(category, scenario)
        
        # Keep only relevant columns
        available_cols = [c for c in columns if c in data.columns]
        out_df = data[available_cols].copy()
        
        # Create directory if needed
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        if append and full_path.exists():
            existing = pd.read_csv(full_path)
            out_df = pd.concat([existing, out_df], ignore_index=True)
            # Remove duplicates (same accelerator + model + scenario)
            dedup_cols = ['accelerator_type', 'benchmark_model', 'scenario']
            if 'input_resolution' in out_df.columns:
                dedup_cols.append('input_resolution')
            out_df = out_df.drop_duplicates(subset=dedup_cols, keep='last')
        
        out_df.to_csv(full_path, index=False)
        saved_files.append((str(out_path), len(out_df)))
    
    return saved_files


# ============================================================================
# 5. PIPELINE COMPATIBILITY CHECK
# ============================================================================

# Known device names in the existing pipeline (from analysis_config.py)
KNOWN_PIPELINE_DEVICES = [
    'Apple M4 CPU',
    'Apple M4 ANE',
    'Mobilint-ARIES',
    'DeepX M1',
    'Hailo-8',
    'Qualcomm QCS6490',
    'RTX PRO 6000 Max-Q',
]

# Device name normalization map:  raw accelerator_type patterns → canonical name
# The raw CSV from AI-BMT may contain verbose names like
#   "CUDA 12.6 + cuDNN 9.3.0 + ONNX Runtime 1.20.0 + maxBatchSize (32)"
# in the accelerator_type or submitter column.
DEVICE_NAME_NORMALIZE = {
    r'deepx|deep\s*x.*m1':         'DeepX M1',
    r'hailo':                       'Hailo-8',
    r'mobilint|aries':              'Mobilint-ARIES',
    r'qualcomm|qcs6490':            'Qualcomm QCS6490',
    r'rtx.*pro.*6000|pro\s*6000':   'RTX PRO 6000 Max-Q',
    r'apple.*ane|ane.*apple':       'Apple M4 ANE',
    r'apple.*m4.*cpu|m4\s*cpu':     'Apple M4 CPU',
    r'jetson|orin':                 'NVIDIA Jetson Orin',
}


def normalize_device_name(raw_name):
    """Normalize a raw accelerator_type string to a canonical device name.
    
    Matches against known patterns. If no pattern matches, returns the
    stripped original string.
    """
    s = raw_name.strip()
    s_lower = s.lower()
    for pattern, canonical in DEVICE_NAME_NORMALIZE.items():
        if re.search(pattern, s_lower):
            return canonical
    return s


def check_pipeline_compatibility(df, base_dir):
    """Check if the converted data is compatible with the full analysis pipeline.
    
    Prints warnings and required manual steps if the device is not yet
    registered in the pipeline configuration files.
    """
    devices = df['accelerator_type'].unique().tolist()
    unknown_devices = [d for d in devices if d not in KNOWN_PIPELINE_DEVICES]
    
    print("\n" + "=" * 70)
    print("  PIPELINE COMPATIBILITY CHECK")
    print("=" * 70)
    
    if not unknown_devices:
        print("  [OK] All devices are already registered in the pipeline.")
        print("       You can run the analysis scripts directly.")
        print("=" * 70)
        return
    
    print(f"  [!] New device(s) detected: {', '.join(unknown_devices)}")
    print(f"      These devices are NOT yet configured in the analysis pipeline.")
    print(f"      The variant CSVs have been written, but scoring and chart")
    print(f"      scripts will ignore unknown devices.\n")
    
    for dev in unknown_devices:
        print(f"  ── Required changes for: {dev} ──\n")
        
        print(f"  1) analysis_config.py  →  ALL_ACCELERATORS")
        print(f"     Add: '{dev}'")
        print(f"")
        print(f"  2) analysis_config.py  →  ACCELERATOR_COLORS")
        print(f"     Add: '{dev}': '#<hex_color>',")
        print(f"")
        print(f"  3) analysis_config.py  →  CASE_ANALYSIS_NPUS  (if NPU)")
        print(f"     Add: '{dev}'")
        print(f"")
        print(f"  4) 1. Create UDS Scores.py  →  HARDWARE_POWER")
        print(f"     Add: '{dev}': <power_watts>,  # or None if unknown")
        print(f"")
        print(f"  5) 1. Create UDS Scores.py  →  HARDWARE_PEAK_COMPUTE")
        print(f"     Add: '{dev}': <peak_TOPS>,    # or None if unknown")
        print(f"")
        print(f"  6) 1. Create UDS Scores.py  →  normalize_device_name()")
        print(f"     Add a pattern for '{dev}'")
        print(f"")
    
    print("  After making the above changes, run:")
    print('    python "1. Create UDS Scores.py"')
    print('    python "2. UDS cases.py"')
    print("    python analyze_results_singleStream_offline.py")
    print("    python analyze_activation_sweep.py")
    print("    python analyze_results_input_resolution_singleStream.py")
    print("    python analyze_results_input_resolution_offline.py")
    print("    python generate_cases_analysis.py")
    print("=" * 70)


# ============================================================================
# 7. EVALUATION SUMMARY
# ============================================================================

def print_eval_summary(df):
    """Print a quick evaluation summary of the converted data."""
    devices = df['accelerator_type'].unique()
    scenarios = df['scenario'].unique()
    models = df['benchmark_model'].apply(normalize_model_name).unique()
    
    print("\n" + "=" * 70)
    print("  EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  Total records     : {len(df)}")
    print(f"  Devices           : {', '.join(devices)}")
    print(f"  Scenarios         : {', '.join(scenarios)}")
    print(f"  Unique models     : {len(models)}")
    
    # Per-scenario summary
    for scenario in sorted(scenarios):
        sdf = df[df['scenario'] == scenario].copy()
        sdf['benchmark_model_norm'] = sdf['benchmark_model'].apply(normalize_model_name)
        
        print(f"\n  --- {scenario} ---")
        print(f"  Records: {len(sdf)}")
        
        if scenario == 'Single-Stream' and 'sample_latency_average' in sdf.columns:
            lat = sdf['sample_latency_average']
            print(f"  Latency (ms)  : min={lat.min():.3f}, avg={lat.mean():.3f}, max={lat.max():.3f}")
        
        if scenario == 'Offline' and 'samples_per_second' in sdf.columns:
            fps = sdf['samples_per_second']
            print(f"  Throughput(FPS): min={fps.min():.1f}, avg={fps.mean():.1f}, max={fps.max():.1f}")
        
        acc = sdf['accuracy']
        print(f"  Accuracy (%)  : min={acc.min():.1f}, avg={acc.mean():.1f}, max={acc.max():.1f}")
    
    # Model classification breakdown
    print(f"\n  --- Model Classification ---")
    counts = {'base': {}, 'activation': {}, 'input_resolution': {}}
    for m in models:
        cat, fam = classify_model(m)
        if fam:
            counts[cat][fam] = counts[cat].get(fam, 0) + 1
    
    if counts['base']:
        print(f"  Base variants      : {sum(counts['base'].values())} models")
        for fam, cnt in sorted(counts['base'].items()):
            print(f"    {fam:20s}: {cnt}")
    
    if counts['activation']:
        print(f"  Activation variants: {sum(counts['activation'].values())} models")
        for fam, cnt in sorted(counts['activation'].items()):
            print(f"    {fam:20s}: {cnt}")
    
    if counts['input_resolution']:
        print(f"  Resolution variants: {sum(counts['input_resolution'].values())} models")
        for fam, cnt in sorted(counts['input_resolution'].items()):
            print(f"    {fam:20s}: {cnt}")
    
    # Top-5 fastest & slowest (Single-Stream)
    ss = df[df['scenario'] == 'Single-Stream'].copy()
    if not ss.empty and 'sample_latency_average' in ss.columns:
        ss['model_norm'] = ss['benchmark_model'].apply(normalize_model_name)
        ss_sorted = ss.sort_values('sample_latency_average')
        
        print(f"\n  --- Top 5 Fastest (Single-Stream Latency) ---")
        for _, r in ss_sorted.head(5).iterrows():
            print(f"    {r['model_norm']:45s}: {r['sample_latency_average']:8.3f} ms  "
                  f"(acc: {r['accuracy']:.1f}%)")
        
        print(f"\n  --- Top 5 Slowest (Single-Stream Latency) ---")
        for _, r in ss_sorted.tail(5).iterrows():
            print(f"    {r['model_norm']:45s}: {r['sample_latency_average']:8.3f} ms  "
                  f"(acc: {r['accuracy']:.1f}%)")
    
    # Top-5 highest throughput (Offline)
    off = df[df['scenario'] == 'Offline'].copy()
    if not off.empty and 'samples_per_second' in off.columns:
        off['model_norm'] = off['benchmark_model'].apply(normalize_model_name)
        off_sorted = off.sort_values('samples_per_second', ascending=False)
        
        print(f"\n  --- Top 5 Throughput (Offline FPS) ---")
        for _, r in off_sorted.head(5).iterrows():
            print(f"    {r['model_norm']:45s}: {r['samples_per_second']:8.1f} FPS  "
                  f"(acc: {r['accuracy']:.1f}%)")
    
    # Top-5 highest accuracy
    print(f"\n  --- Top 5 Accuracy ---")
    df_copy = df.copy()
    df_copy['model_norm'] = df_copy['benchmark_model'].apply(normalize_model_name)
    acc_sorted = df_copy.drop_duplicates(subset=['model_norm']).sort_values('accuracy', ascending=False)
    for _, r in acc_sorted.head(5).iterrows():
        print(f"    {r['model_norm']:45s}: {r['accuracy']:.1f}%")
    
    print("\n" + "=" * 70)


# ============================================================================
# 8. MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Convert raw AI-BMT evaluation CSV to variant folder format.')
    parser.add_argument('--input', '-i', default=DEFAULT_INPUT_CSV,
                        help=f'Input CSV file (default: {DEFAULT_INPUT_CSV})')
    parser.add_argument('--device-name', '-d', default=None,
                        help='Override accelerator_type name (e.g., "NVIDIA Jetson Orin")')
    parser.add_argument('--append', '-a', action='store_true',
                        help='Append to existing variant CSVs instead of overwriting')
    parser.add_argument('--output-dir', '-o', default='.',
                        help='Base output directory (default: current directory)')
    parser.add_argument('--no-summary', action='store_true',
                        help='Skip evaluation summary output')
    parser.add_argument('--no-compat-check', action='store_true',
                        help='Skip pipeline compatibility check')
    args = parser.parse_args()
    
    base_dir = Path(args.output_dir)
    input_path = base_dir / args.input if not Path(args.input).is_absolute() else Path(args.input)
    
    print(f"\n{'=' * 70}")
    print(f"  AI-MicroBMT — Convert & Evaluate")
    print(f"{'=' * 70}")
    print(f"  Input  : {input_path}")
    print(f"  Mode   : {'Append' if args.append else 'Overwrite'}")
    
    # Load raw data
    print(f"\n  [1/5] Loading raw CSV...")
    df = load_raw_csv(input_path)
    print(f"        → {len(df)} valid records loaded")
    
    # Show normalized device names
    devices = df['accelerator_type'].unique()
    print(f"        → Device(s): {', '.join(devices)}")
    
    # Apply device name override (takes precedence over auto-normalization)
    if args.device_name:
        df = apply_device_name_override(df, args.device_name)
        print(f"        → Device name overridden to: {args.device_name}")
    
    # Process and classify
    print(f"  [2/5] Classifying models...")
    grouped = process_and_split(df)
    print(f"        → {len(grouped)} groups created")
    
    # Save to variant folders
    print(f"  [3/5] Saving variant CSVs...")
    saved = save_results(grouped, base_dir, append=args.append)
    for fpath, count in sorted(saved):
        print(f"        → {fpath}  ({count} rows)")
    
    # Evaluation summary
    if not args.no_summary:
        print(f"  [4/5] Generating evaluation summary...")
        print_eval_summary(df)
    
    # Pipeline compatibility check
    if not args.no_compat_check:
        print(f"  [5/5] Checking pipeline compatibility...")
        check_pipeline_compatibility(df, base_dir)
    
    print(f"\n  Done! {len(saved)} CSV files written.\n")
    return 0


if __name__ == '__main__':
    sys.exit(main())
