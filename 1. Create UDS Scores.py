"""
Unified Deployment Score (UDS) - Sub-Score Calculator (S1~S7)
=============================================================
Step 1 of 2: This script calculates the 7 individual sub-scores for AI-MicroBMT
evaluation results. Weighted composite UDS scores are computed separately in
'2. UDS cases.py' using user-defined weight profiles.

Sub-scores:
- S1: Coverage/Correctness score (model-level) - uses BASE SUITE
- S2: Efficiency score (normalized speedup) - uses BASE SUITE
- S3: Resolution-robust scaling score - uses RESOLUTION-SWEEP data
- S4: Accuracy-retention score - uses BASE SUITE
- S5: Throughput-gain score - uses BASE SUITE
- S6: Peak compute efficiency (optional, requires hardware specs) - uses BASE SUITE
- S7: Power efficiency (optional, requires hardware specs) - uses BASE SUITE

Dataset separation:
- BASE SUITE (T_base): Canonical resolution r_0=224, all model variants
  Used for S1, S2, S4, S5, S6, S7
- RESOLUTION-SWEEP (T_res): Multiple resolutions from input resolution variant
  Used exclusively for S3

Outputs:
- UDS_scores_summary.csv         : Per-device sub-scores (S1~S7) and metadata
- UDS_scores_detailed_base.csv   : Per-task detailed results (base suite)
- UDS_scores_detailed_resolution.csv : Per-task detailed results (resolution sweep)

Author: AI-MicroBMT Team
Date: 2026-02-03
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# USER-CONFIGURABLE PARAMETERS (Global Variables)
# ============================================================================

# Feasibility gate parameters
TAU_ACC = 4.0  # Maximum tolerable accuracy drop (%) from CPU baseline
THETA_SPEEDUP = 1.0  # Minimum required speedup vs CPU (1.0 = parity)
A_MIN = 15.0  # Minimum absolute accuracy (%) to avoid silent failures

# Small constant to avoid division by zero
EPSILON = 1e-6  

# Note: S_MAX, U_MAX, E_PEAK_MAX, E_WATT_MAX are no longer constants.
# They are now computed from observed data as per updated UDS.tex specification.

# Data directories - BASE SUITE (canonical resolution r_0=224)
# Used for S1, S2, S4, S5 calculations
DATA_FOLDERS_BASE = [
    'densenet variant',
    'convnext variant',
    'mobilenet activation variant',
    'mobilenet variant',
    'regnet variant',
    'resnet activation variant',
    'resnet variant',
    'resnext50 variant',
    'shufflenet variant',
    'vit variant'
]

# Data directories - RESOLUTION SWEEP (multiple resolutions)
# Used exclusively for S3 scaling score calculation
DATA_FOLDERS_RES = [
    'input resolution variant'
]

# Hardware power specifications (W)
# Uses TDP or max power values where available
# Apple M4 (CPU/ANE) are SoC - power not separately measurable, excluded from S7
HARDWARE_POWER = {
    'DeepX M1': 5.0,  # 2-5W range, using max
    'Hailo-8': 8.65,  # TDP
    'Mobilint-ARIES': 25.0,  # TDP
    'Qualcomm QCS6490': 9.0,  # 6-9W range, using max
    'RTX PRO 6000 Max-Q': 300.0,  # max power
    'Apple M4 ANE': None,  # SoC - not separately measurable
    'Apple M4 CPU': None,  # SoC - not separately measurable
}

# Hardware peak compute specifications - from hardware_info.tex
# Used for S6 calculation (Eq. 21-22 in UDS.tex)
# Apple M4 (CPU/ANE) are SoC - excluded from S6
HARDWARE_PEAK_COMPUTE = {
    'DeepX M1': 25.0,  # INT8 TOPS
    'Hailo-8': 26.0,  # INT8 TOPS
    'Mobilint-ARIES': 80.0,  # INT8 TOPS
    'Qualcomm QCS6490': 12.0,  # INT8 TOPS
    'RTX PRO 6000 Max-Q': 110.0,  # TFLOPS (FP32)
    'Apple M4 ANE': None,  # SoC - not separately measurable
    'Apple M4 CPU': None,  # SoC - not separately measurable
}

# Output options
SAVE_DETAILED_RESULTS = True
OUTPUT_CSV_BASE = 'UDS_scores_detailed_base.csv'
OUTPUT_CSV_RES = 'UDS_scores_detailed_resolution.csv'
OUTPUT_SUMMARY_CSV = 'UDS_scores_summary.csv'


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_resolution_from_model(model_name: str) -> Optional[int]:
    """Extract input resolution from model name (e.g., 'inputres_224' -> 224)"""
    if 'inputres_' in model_name.lower():
        try:
            parts = model_name.lower().split('inputres_')
            res_str = parts[1].split('_')[0]
            return int(res_str)
        except:
            return None
    return None


def extract_resolution_from_column(df: pd.DataFrame) -> pd.Series:
    """Extract resolution from input_resolution column or model name"""
    if 'input_resolution' in df.columns:
        # Format: "inputResolution:224"
        return df['input_resolution'].str.extract(r'inputResolution:(\d+)')[0].astype(float)
    else:
        # Extract from model name
        return df['benchmark_model'].apply(extract_resolution_from_model)


def normalize_device_name(device: str) -> str:
    """Normalize device/accelerator names for consistent matching"""
    device = device.strip()
    
    # Map variations to standard names
    if 'deepx' in device.lower() or 'M1' in device:
        return 'DeepX M1'
    elif 'hailo' in device.lower():
        return 'Hailo-8'
    elif 'mobilint' in device.lower() or 'aries' in device.lower():
        return 'Mobilint-ARIES'
    elif 'qualcomm' in device.lower() or 'qcs6490' in device.lower():
        return 'Qualcomm QCS6490'
    elif 'rtx' in device.lower() or 'pro 6000' in device.lower():
        return 'RTX PRO 6000 Max-Q'
    elif 'ane' in device.lower():
        return 'Apple M4 ANE'
    elif 'apple m4 cpu' in device.lower():
        return 'Apple M4 CPU'
    
    return device


def extract_base_model_name(model_name: str) -> str:
    """Extract base model name without resolution/variant suffixes"""
    # Remove common suffixes
    name = model_name.lower()
    
    # Remove opset, pretrained, quantized, etc.
    for suffix in ['_opset', '_pretrained', '_quantized', '_model', '_compiled', 
                   '_trained', '_bgr2rgb', '_normalized', '_inputres']:
        if suffix in name:
            name = name.split(suffix)[0]
    
    # Handle specific patterns
    if 'densenet' in name:
        if 'densenet121' in name:
            return 'densenet121'
        elif 'densenet169' in name:
            return 'densenet169'
        elif 'densenet201' in name:
            return 'densenet201'
    
    if 'resnet' in name:
        # Match resnet18, resnet50, resnet101, etc.
        import re
        match = re.search(r'resnet\d+', name)
        if match:
            return match.group(0)
    
    if 'mobilenetv2' in name or 'mobilenet_v2' in name:
        return 'mobilenetv2'
    
    if 'vit' in name:
        if 'tiny' in name:
            return 'vit_tiny'
        elif 'small' in name:
            return 'vit_small'
        elif 'base' in name:
            return 'vit_base'
    
    # Generic extraction - take first meaningful part
    parts = name.replace('-', '_').split('_')
    for part in parts:
        if part and not part.isdigit() and part not in ['w1', 'w0']:
            return part
    
    return model_name.split('_')[0]


# ============================================================================
# DATA LOADING
# ============================================================================

def load_benchmark_data(base_path: str, folder_list: List[str], dataset_type: str) -> Dict[str, pd.DataFrame]:
    """
    Load benchmark CSV files from specified folders.
    
    Args:
        base_path: Base directory path
        folder_list: List of folders to load
        dataset_type: 'base' or 'resolution' for logging
    
    Returns:
        Dictionary with keys: 'offline' and 'singlestream' DataFrames
    """
    offline_dfs = []
    singlestream_dfs = []
    
    for folder in folder_list:
        folder_path = Path(base_path) / folder
        
        if not folder_path.exists():
            print(f"  Warning: Folder not found: {folder_path}")
            continue
        
        # Find offline results CSV
        offline_files = list(folder_path.glob('*offline*.csv'))
        if offline_files:
            df = pd.read_csv(offline_files[0])
            df['source_folder'] = folder
            df['dataset_type'] = dataset_type
            offline_dfs.append(df)
        
        # Find single-stream results CSV
        singlestream_files = list(folder_path.glob('*single*.csv'))
        if singlestream_files:
            df = pd.read_csv(singlestream_files[0])
            df['source_folder'] = folder
            df['dataset_type'] = dataset_type
            singlestream_dfs.append(df)
    
    # Combine all data
    offline_data = pd.concat(offline_dfs, ignore_index=True) if offline_dfs else pd.DataFrame()
    singlestream_data = pd.concat(singlestream_dfs, ignore_index=True) if singlestream_dfs else pd.DataFrame()
    
    print(f"  [{dataset_type}] Loaded {len(offline_data)} offline results from {len(offline_dfs)} folders")
    print(f"  [{dataset_type}] Loaded {len(singlestream_data)} single-stream results from {len(singlestream_dfs)} folders")
    
    return {
        'offline': offline_data,
        'singlestream': singlestream_data
    }


# ============================================================================
# UDS CALCULATION - CORE COMPONENTS
# ============================================================================

def prepare_unified_dataset(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge offline and single-stream data into unified dataset.
    Adds resolution extraction and device normalization.
    """
    offline = data['offline'].copy()
    singlestream = data['singlestream'].copy()
    
    print(f"  Initial offline rows: {len(offline)}")
    print(f"  Initial single-stream rows: {len(singlestream)}")
    
    # Normalize device names
    offline['device'] = offline['accelerator_type'].apply(normalize_device_name)
    singlestream['device'] = singlestream['accelerator_type'].apply(normalize_device_name)
    
    # Extract base model names
    offline['base_model'] = offline['benchmark_model'].apply(extract_base_model_name)
    singlestream['base_model'] = singlestream['benchmark_model'].apply(extract_base_model_name)
    
    # Extract resolutions
    offline['resolution'] = extract_resolution_from_column(offline)
    singlestream['resolution'] = extract_resolution_from_column(singlestream)
    
    # Rename columns for clarity
    offline = offline.rename(columns={
        'samples_per_second': 'throughput_offline',
        'accuracy': 'accuracy_offline'
    })
    
    singlestream = singlestream.rename(columns={
        'sample_latency_average': 'latency_singlestream',
        'accuracy': 'accuracy_singlestream'
    })
    
    # Merge on common keys - use suffixes to distinguish
    merge_keys = ['device', 'base_model', 'benchmark_model', 'resolution']
    
    # Keep necessary columns
    offline_cols = merge_keys + ['throughput_offline', 'accuracy_offline']
    singlestream_cols = merge_keys + ['latency_singlestream', 'accuracy_singlestream']
    
    offline_merge = offline[offline_cols].copy()
    singlestream_merge = singlestream[singlestream_cols].copy()
    
    print(f"  Offline rows before merge: {len(offline_merge)}")
    print(f"  Single-stream rows before merge: {len(singlestream_merge)}")
    
    # Merge with outer join to keep all data
    merged = pd.merge(
        offline_merge,
        singlestream_merge,
        on=merge_keys,
        how='outer',
        suffixes=('_off', '_ss')
    )
    
    print(f"  Merged rows (outer join): {len(merged)}")
    
    # Combine accuracy columns (prefer non-null values)
    merged['accuracy'] = merged['accuracy_offline'].fillna(merged['accuracy_singlestream'])
    merged = merged.drop(columns=['accuracy_offline', 'accuracy_singlestream'])
    
    # Calculate single-stream throughput (1000ms / latency_ms) where available
    merged['throughput_singlestream'] = np.where(
        merged['latency_singlestream'].notna(),
        1000.0 / merged['latency_singlestream'],
        np.nan
    )
    
    # Calculate scaling efficiency η(r) where both throughputs available
    merged['eta'] = np.where(
        (merged['throughput_offline'].notna()) & (merged['throughput_singlestream'].notna()),
        merged['throughput_offline'] / merged['throughput_singlestream'],
        np.nan
    )
    
    print(f"  Rows with offline data: {merged['throughput_offline'].notna().sum()}")
    print(f"  Rows with singlestream data: {merged['latency_singlestream'].notna().sum()}")
    print(f"  Rows with both: {((merged['throughput_offline'].notna()) & (merged['latency_singlestream'].notna())).sum()}")
    
    return merged


def get_cpu_baseline(unified_df: pd.DataFrame) -> pd.DataFrame:
    """Extract CPU baseline data (Apple M4 CPU)"""
    cpu_data = unified_df[unified_df['device'] == 'Apple M4 CPU'].copy()
    
    if len(cpu_data) == 0:
        print("WARNING: No CPU baseline data found!")
        return pd.DataFrame()
    
    # Rename columns for baseline
    cpu_data = cpu_data.rename(columns={
        'accuracy': 'accuracy_cpu',
        'latency_singlestream': 'latency_cpu',
        'throughput_offline': 'throughput_cpu',
        'throughput_singlestream': 'throughput_singlestream_cpu'
    })
    
    baseline_cols = ['base_model', 'benchmark_model', 'resolution', 
                     'accuracy_cpu', 'latency_cpu', 'throughput_cpu', 'throughput_singlestream_cpu']
    
    return cpu_data[baseline_cols]


def calculate_metrics_with_baseline(unified_df: pd.DataFrame, cpu_baseline: pd.DataFrame, 
                                    dataset_type: str) -> pd.DataFrame:
    """
    Join device data with CPU baseline and calculate comparative metrics.
    
    Args:
        dataset_type: 'base' or 'resolution' - determines which feasibility gate to apply
            'base': Eq.4 - compile_success ∧ accuracy≥A_min ∧ Δ≤τ_acc ∧ speedup≥θ
            'resolution': Eq.5 - compile_success only
    """
    # Remove CPU from device data (we only compare NPUs/accelerators to CPU)
    device_data = unified_df[unified_df['device'] != 'Apple M4 CPU'].copy()
    
    print(f"  [{dataset_type}] Device data (non-CPU): {len(device_data)} rows")
    
    # Merge with CPU baseline
    merge_keys = ['base_model', 'benchmark_model', 'resolution']
    data_with_baseline = pd.merge(
        device_data,
        cpu_baseline,
        on=merge_keys,
        how='left'
    )
    
    print(f"  [{dataset_type}] After CPU baseline merge: {len(data_with_baseline)} rows")
    print(f"  [{dataset_type}] Rows with CPU baseline: {data_with_baseline['accuracy_cpu'].notna().sum()}")
    
    # Calculate metrics
    # Accuracy drop (%) - only where both accuracies are available
    data_with_baseline['accuracy_drop'] = np.where(
        (data_with_baseline['accuracy_cpu'].notna()) & (data_with_baseline['accuracy'].notna()),
        (data_with_baseline['accuracy_cpu'] - data_with_baseline['accuracy']) / 
        data_with_baseline['accuracy_cpu'] * 100,
        np.nan
    )
    
    # Speedup (single-stream) - only where both latencies are available
    data_with_baseline['speedup'] = np.where(
        (data_with_baseline['latency_cpu'].notna()) & (data_with_baseline['latency_singlestream'].notna()),
        data_with_baseline['latency_cpu'] / data_with_baseline['latency_singlestream'],
        np.nan
    )
    
    # Compilation/run success indicator (if we have data, it compiled)
    data_with_baseline['compile_success'] = 1
    
    # Apply feasibility gate based on dataset type
    if dataset_type == 'base':
        # Base suite feasibility: includes accuracy constraints (Eq. 4 in UDS.tex)
        data_with_baseline['feasible'] = (
            (data_with_baseline['compile_success'] == 1) &
            (data_with_baseline['accuracy'].notna()) &
            (data_with_baseline['accuracy'] >= A_MIN) &
            (data_with_baseline['accuracy_drop'].notna()) &
            (data_with_baseline['accuracy_drop'] <= TAU_ACC) &
            (data_with_baseline['speedup'].notna()) &
            (data_with_baseline['speedup'] >= THETA_SPEEDUP)
        ).astype(int)
    else:  # dataset_type == 'resolution'
        # Resolution-sweep feasibility (Eq. 5 in UDS.tex):
        # I^res_{d,t} = [t ∈ T_res ∧ c_{d,t} = 1]
        # Only compile success matters - no accuracy, speedup, or base model feasibility checks
        data_with_baseline['feasible'] = (
            (data_with_baseline['compile_success'] == 1)
        ).astype(int)
    
    print(f"  [{dataset_type}] Feasible tasks: {data_with_baseline['feasible'].sum()}")
    print(f"  [{dataset_type}] Tasks with latency data: {data_with_baseline['latency_singlestream'].notna().sum()}")
    print(f"  [{dataset_type}] Tasks with throughput data: {data_with_baseline['throughput_offline'].notna().sum()}")
    
    return data_with_baseline


# ============================================================================
# UDS SUB-SCORES
# ============================================================================

def calculate_s1_coverage(df_base: pd.DataFrame, n_m_global: int = None) -> Tuple[Dict[str, float], int]:
    """
    S1: Coverage/Correctness score (variant-level)
    
    Uses BASE SUITE data only (canonical resolution r_0=224).
    
    Per UDS.tex Eq. 6-7:
    - M = set of (model, variant) identifiers, i.e., benchmark_model
    - N_M = |M| = number of model variants (GLOBAL, same for all devices)
    - J_{d,m} = I^base_{d,(m,r_0)} = feasibility indicator for variant m
    - S1(d) = (1/N_M) * sum_m J_{d,m}
    
    This counts what fraction of ALL variants (benchmark_model) are feasible,
    NOT what fraction of base_models have at least one feasible variant.
    
    CRITICAL: N_M must be the TOTAL number of variants in the benchmark suite (67),
    not the number of variants each device was able to run. Compile failures
    count as coverage failures and lower S1.
    
    Args:
        df_base: Base suite data containing all device results
        n_m_global: Global N_M value (if provided, use this; otherwise compute from data)
    
    Returns:
        Tuple of (scores dict, N_M used)
    """
    scores = {}
    
    devices = [d for d in df_base['device'].unique() if d != 'Apple M4 CPU']
    
    # M = set of (model, variant) identifiers = benchmark_model
    # N_M = TOTAL number of variants in the entire benchmark suite
    # This should be computed from CPU baseline (which runs all 67 models)
    # or provided as n_m_global parameter
    if n_m_global is not None:
        N_M = n_m_global
    else:
        # Get N_M from CPU baseline (CPU runs all models)
        cpu_data = df_base[df_base['device'] == 'Apple M4 CPU']
        if len(cpu_data) > 0:
            N_M = cpu_data['benchmark_model'].nunique()
        else:
            # Fallback: use all unique benchmark_models across all devices
            N_M = df_base['benchmark_model'].nunique()
    
    for device in devices:
        device_df = df_base[df_base['device'] == device]
        
        # J_{d,m} = 1 if variant m is feasible on device d
        # Count number of feasible variants (benchmark_model)
        feasible_variants = device_df[device_df['feasible'] == 1]['benchmark_model'].unique()
        
        # S1(d) = |{m : J_{d,m} = 1}| / N_M
        # N_M is GLOBAL (67), not per-device!
        s1 = len(feasible_variants) / N_M if N_M > 0 else 0.0
        scores[device] = s1
    
    return scores, N_M


def phi_speedup_transform(s: float, theta: float, s_max: float) -> float:
    """
    Transform speedup to [0,1] with diminishing returns.
    φ(s) = min(1, log(s/θ) / log(s_max/θ))
    
    Per UDS.tex Eq. 9: s_max is computed from observed data, not a constant.
    """
    if s < theta:
        return 0.0
    
    if s >= s_max:
        return 1.0
    
    numerator = np.log(s / theta)
    denominator = np.log(s_max / theta)
    
    return min(1.0, numerator / denominator)


def calculate_s2_efficiency(df_base: pd.DataFrame) -> Dict[str, float]:
    """
    S2: Efficiency score (normalized speedup)
    
    Uses BASE SUITE data only (canonical resolution r_0=224).
    Average φ(speedup) over all feasible tasks.
    
    Per UDS.tex Eq. 9: s_max is computed as the maximum observed feasible speedup + ε.
    """
    scores = {}
    
    devices = df_base['device'].unique()
    
    # Compute s_max from observed data (Eq. 9)
    # s_max = max over all devices and feasible tasks of speedup + ε
    feasible_df = df_base[df_base['feasible'] == 1]
    if len(feasible_df) > 0 and feasible_df['speedup'].notna().any():
        s_max = feasible_df['speedup'].max() + EPSILON
    else:
        s_max = THETA_SPEEDUP + EPSILON  # Fallback
    
    for device in devices:
        device_df = df_base[(df_base['device'] == device) & (df_base['feasible'] == 1)]
        
        if len(device_df) == 0:
            scores[device] = 0.0
            continue
        
        # Apply φ transform to each speedup with computed s_max
        phi_values = device_df['speedup'].apply(lambda s: phi_speedup_transform(s, THETA_SPEEDUP, s_max))
        
        s2 = phi_values.mean()
        scores[device] = s2
    
    return scores


def calculate_s3_scaling(df_res: pd.DataFrame) -> Dict[str, float]:
    """
    S3: Resolution-robust scaling score
    
    Uses RESOLUTION-SWEEP data only (multiple resolutions from input resolution variant).
    
    Per UDS.tex Eq. 11-13:
    - M_res is a subset of M (model, variant identifiers) = benchmark_model
    - η*_{d,m} = max over resolutions of η for feasible tasks
    - g_{d,(m,r)} = min(1, η / η*)  -- normalized scaling
    - ρ_{d,m} = fraction of feasible resolutions for model m
    - K_{d,m} = count of feasible resolutions
    - g̅_{d,m} = average g over feasible resolutions
    - S3(d) = (1/|M_res|) * Σ_m ρ_{d,m} * g̅_{d,m}
    
    Note: Per UDS.tex, M is defined as (model, variant) identifiers, so we use
    benchmark_model (not base_model) to correctly handle variant-level granularity.
    """
    scores = {}
    
    devices = df_res['device'].unique()
    # M_res = set of (model, variant) identifiers = benchmark_model (per UDS.tex definition)
    all_models_res = df_res['benchmark_model'].unique()
    M_res = len(all_models_res)  # |M_res|
    
    # Get all resolutions in the dataset
    all_resolutions = df_res['resolution'].dropna().unique()
    R_res = len(all_resolutions)  # |R_res|
    
    for device in devices:
        device_df = df_res[df_res['device'] == device]
        
        if len(device_df) == 0 or M_res == 0:
            scores[device] = 0.0
            continue
        
        model_contributions = []
        
        for model in all_models_res:
            model_df = device_df[device_df['benchmark_model'] == model]
            
            # K_{d,m}: count of feasible resolutions for this model
            feasible_df = model_df[(model_df['feasible'] == 1) & (model_df['eta'].notna())]
            K_dm = len(feasible_df)
            
            if K_dm == 0:
                # No feasible resolutions for this model
                model_contributions.append(0.0)
                continue
            
            # ρ_{d,m}: fraction of feasible resolutions
            rho_dm = K_dm / R_res if R_res > 0 else 0.0
            
            # η*_{d,m}: best η across feasible resolutions
            eta_star = feasible_df['eta'].max()
            
            if eta_star <= 0:
                model_contributions.append(0.0)
                continue
            
            # g_{d,(m,r)}: normalized scaling for each feasible resolution
            g_values = (feasible_df['eta'] / eta_star).clip(upper=1.0)
            
            # g̅_{d,m}: average g over feasible resolutions
            g_bar_dm = g_values.mean()
            
            # Contribution: ρ_{d,m} * g̅_{d,m}
            model_contributions.append(rho_dm * g_bar_dm)
        
        # S3(d) = (1/|M_res|) * Σ_m ρ_{d,m} * g̅_{d,m}
        s3 = sum(model_contributions) / M_res if M_res > 0 else 0.0
        scores[device] = s3
    
    return scores


def calculate_s4_accuracy_retention(df_base: pd.DataFrame) -> Dict[str, float]:
    """
    S4: Accuracy-retention score (feasible-set average)
    
    Per UDS.tex Eq. 14-16:
    - Ā_d = average accuracy over feasible tasks
    - Ā_cpu = average CPU baseline accuracy over same tasks
    - Δ^avg_d = max(0, (Ā_cpu - Ā_d) / Ā_cpu × 100)
    - S4(d) = 1 - min(1, Δ^avg_d / τ_acc)
    """
    scores = {}
    
    devices = df_base['device'].unique()
    
    for device in devices:
        feasible_df = df_base[(df_base['device'] == device) & (df_base['feasible'] == 1)]
        
        N_d_base = len(feasible_df)
        
        if N_d_base == 0:
            scores[device] = 0.0
            continue
        
        # Average accuracy over feasible tasks
        A_bar_d = feasible_df['accuracy'].mean()
        A_bar_cpu = feasible_df['accuracy_cpu'].mean()
        
        if A_bar_cpu <= 0:
            scores[device] = 0.0
            continue
        
        # Relative average accuracy drop
        delta_avg = max(0, (A_bar_cpu - A_bar_d) / A_bar_cpu * 100)
        
        # Normalize to [0,1] using τ_acc
        s4 = 1 - min(1.0, delta_avg / TAU_ACC)
        scores[device] = s4
    
    return scores


def calculate_s5_throughput_gain(df_base: pd.DataFrame) -> Dict[str, float]:
    """
    S5: Offline-throughput gain score (feasible-set average)
    
    Per UDS.tex Eq. 17-18:
    - Q̄^off_d = average offline throughput over feasible tasks
    - Q̄^off_cpu = average CPU baseline throughput over same tasks
    - u_d = Q̄^off_d / Q̄^off_cpu (throughput speedup)
    - u_max = max over all devices with N_d^base > 0 of u_d + ε
    - ψ(u) = min(1, log(max(u,1)) / log(u_max))
    - S5(d) = ψ(u_d)
    """
    scores = {}
    u_values = {}  # Store u_d for each device
    devices = df_base['device'].unique()
    
    # First pass: compute u_d for each device
    for device in devices:
        feasible_df = df_base[(df_base['device'] == device) & (df_base['feasible'] == 1)]
        
        N_d_base = len(feasible_df)
        
        if N_d_base == 0:
            u_values[device] = None
            continue
        
        # Filter to tasks with throughput data
        throughput_df = feasible_df[
            (feasible_df['throughput_offline'].notna()) & 
            (feasible_df['throughput_cpu'].notna())
        ]
        
        if len(throughput_df) == 0:
            u_values[device] = None
            continue
        
        # Average offline throughput
        Q_bar_d = throughput_df['throughput_offline'].mean()
        Q_bar_cpu = throughput_df['throughput_cpu'].mean()
        
        if Q_bar_cpu <= 0:
            u_values[device] = None
            continue
        
        # Throughput speedup
        u_d = Q_bar_d / Q_bar_cpu
        u_values[device] = u_d
    
    # Compute u_max from observed data (Eq. 18)
    # u_max = max over devices with N_d^base > 0 of u_d + ε
    valid_u_values = [u for u in u_values.values() if u is not None]
    if len(valid_u_values) > 0:
        u_max = max(valid_u_values) + EPSILON
    else:
        u_max = 1.0 + EPSILON  # Fallback
    
    # Second pass: compute ψ(u_d) with computed u_max
    for device in devices:
        u_d = u_values.get(device)
        
        if u_d is None:
            scores[device] = 0.0
            continue
        
        # ψ transform: log scale with computed u_max
        if u_d <= 1:
            psi = 0.0
        elif u_d >= u_max:
            psi = 1.0
        else:
            psi = min(1.0, np.log(max(u_d, 1)) / np.log(u_max))
        
        scores[device] = psi
    
    return scores


def calculate_s6_peak_compute_efficiency(df: pd.DataFrame, peak_compute_specs: Dict[str, float]) -> Dict[str, Optional[float]]:
    """
    S6: Peak-compute-normalized offline efficiency (optional)
    
    Per UDS.tex Eq. 19-22:
    - Q̄^off_d = (1/N_d^base) * Σ I^base_{d,t} * Q^off_{d,t}  (same as S5)
    - e^peak_d = Q̄^off_d / Π_d  (average throughput / peak compute)
    - e^peak_min = min over devices in D_Π of e^peak_d
    - e^peak_max = max over devices in D_Π of e^peak_d + ε
    - ω(x) = min(1, log(x / e^peak_min) / log(e^peak_max / e^peak_min))
    - S6(d) = ω(e^peak_d)
    
    Uses vendor-reported peak compute Π_d (TOPS/TFLOPS) from hardware_info.tex.
    Apple M4 CPU/ANE are SoC devices where peak compute is not separately attributable.
    
    Note: Q̄^off_d uses the same definition as in S5 (Eq. 17), which is the average
    offline throughput over feasible tasks that have both device and CPU throughput data.
    """
    scores = {}
    e_peak_values = {}  # Store e^peak_d for each device
    
    devices = df['device'].unique()
    
    # First pass: compute e^peak_d for each device with peak compute specs
    for device in devices:
        if device not in peak_compute_specs or peak_compute_specs[device] is None:
            e_peak_values[device] = None
            continue
        
        device_df = df[(df['device'] == device) & (df['feasible'] == 1)]
        
        if len(device_df) == 0:
            e_peak_values[device] = None
            continue
        
        # Filter to tasks with both device and CPU throughput data (same as S5)
        throughput_df = device_df[
            (device_df['throughput_offline'].notna()) & 
            (device_df['throughput_cpu'].notna())
        ]
        
        if len(throughput_df) == 0:
            e_peak_values[device] = None
            continue
        
        peak_compute = peak_compute_specs[device]  # Π_d in TOPS or TFLOPS
        
        # Q̄^off_d = average offline throughput over feasible tasks (same definition as S5)
        # e^peak_d = Q̄^off_d / Π_d (Eq. 20)
        Q_bar_d = throughput_df['throughput_offline'].mean()
        e_peak_d = Q_bar_d / peak_compute
        e_peak_values[device] = e_peak_d
    
    # Compute e^peak_min and e^peak_max from observed data (Eq. 22)
    valid_e_peak = [e for e in e_peak_values.values() if e is not None]
    
    if len(valid_e_peak) == 0:
        # No devices with peak compute specs
        for device in devices:
            scores[device] = None
        return scores
    
    e_peak_min = min(valid_e_peak)
    e_peak_max = max(valid_e_peak) + EPSILON
    
    # Second pass: compute ω(e^peak_d) with computed min/max
    for device in devices:
        e_peak_d = e_peak_values.get(device)
        
        if e_peak_d is None:
            scores[device] = None
            continue
        
        # ω transform: log scale with min/max normalization (Eq. 22)
        if e_peak_d <= e_peak_min:
            omega = 0.0
        elif e_peak_d >= e_peak_max:
            omega = 1.0
        else:
            omega = min(1.0, np.log(e_peak_d / e_peak_min) / np.log(e_peak_max / e_peak_min))
        
        scores[device] = omega
    
    return scores


def calculate_s7_power_normalized_efficiency(df: pd.DataFrame, power_specs: Dict[str, float]) -> Dict[str, Optional[float]]:
    """
    S7: Power-normalized offline efficiency (optional)
    
    Per UDS.tex Eq. 23-24:
    - e^watt_d = (1/N_d^base) * Σ I^base_{d,t} * (Q^off_{d,t} / p_{d,t})
    - e^watt_min = min over devices in D_P of e^watt_d
    - e^watt_max = max over devices in D_P of e^watt_d + ε
    - ζ(x) = min(1, log(x / e^watt_min) / log(e^watt_max / e^watt_min))
    - S7(d) = ζ(e^watt_d)
    
    Note: S7 computes the average of per-task (throughput/power) ratios over feasible tasks.
    Unlike S6, this is explicitly defined by Eq. 23 as a per-task computation.
    We use feasible tasks with throughput data (same filtering as S5/S6 for consistency).
    """
    scores = {}
    e_watt_values = {}  # Store e^watt_d for each device
    
    devices = df['device'].unique()
    
    # First pass: compute e^watt_d for each device with power specs
    for device in devices:
        if device not in power_specs or power_specs[device] is None:
            e_watt_values[device] = None
            continue
        
        device_df = df[(df['device'] == device) & (df['feasible'] == 1)]
        
        N_d_base = len(device_df)
        
        if N_d_base == 0:
            e_watt_values[device] = None
            continue
        
        # Filter to tasks with throughput data (consistent with S5/S6)
        throughput_df = device_df[
            (device_df['throughput_offline'].notna()) & 
            (device_df['throughput_cpu'].notna())
        ]
        
        if len(throughput_df) == 0:
            e_watt_values[device] = None
            continue
        
        power = power_specs[device]
        
        # S7 uses average of per-task (throughput/power) ratios (Eq.23)
        # e^watt_d = (1/N) * Σ (Q^off_{d,t} / p_{d,t})
        e_watt_d = (throughput_df['throughput_offline'] / power).mean()
        e_watt_values[device] = e_watt_d
    
    # Compute e^watt_min and e^watt_max from observed data (Eq. 24)
    valid_e_watt = [e for e in e_watt_values.values() if e is not None]
    
    if len(valid_e_watt) == 0:
        # No devices with power specs
        for device in devices:
            scores[device] = None
        return scores
    
    e_watt_min = min(valid_e_watt)
    e_watt_max = max(valid_e_watt) + EPSILON
    
    # Second pass: compute ζ(e^watt_d) with computed min/max
    for device in devices:
        e_watt_d = e_watt_values.get(device)
        
        if e_watt_d is None:
            scores[device] = None
            continue
        
        # ζ transform: log scale with min/max normalization (Eq. 24)
        if e_watt_d <= e_watt_min:
            zeta = 0.0
        elif e_watt_d >= e_watt_max:
            zeta = 1.0
        else:
            zeta = min(1.0, np.log(e_watt_d / e_watt_min) / np.log(e_watt_max / e_watt_min))
        
        scores[device] = zeta
    
    return scores


# ============================================================================
# MAIN UDS CALCULATION
# ============================================================================

def calculate_uds_scores(df_base: pd.DataFrame, df_res: pd.DataFrame, 
                         peak_compute_specs: Dict[str, float],
                         power_specs: Dict[str, float],
                         n_m_global: int = None) -> pd.DataFrame:
    """
    Calculate all UDS sub-scores (S1-S7) for each device.
    
    This script only produces the individual sub-scores.
    Weighted composite UDS scores are computed in '2. UDS cases.py'.
    
    Per UDS.tex specification:
    - S1 (Coverage): Model coverage from base suite (Eq. 6-7)
    - S2 (Efficiency): Latency efficiency from base suite (Eq. 8-10)
    - S3 (Scaling): Resolution scaling from resolution-sweep (Eq. 12-13)
    - S4 (Accuracy Retention): Accuracy retention from base suite (Eq. 14-16)
    - S5 (Throughput Gain): Throughput gain from base suite (Eq. 17-18)
    - S6 (Peak Compute Efficiency): Peak-compute-normalized throughput (Eq. 19-22) - optional
    - S7 (Power Efficiency): Power-normalized efficiency (Eq. 23-24) - optional
    
    Args:
        df_base: Base suite data (canonical r_0=224) for S1, S2, S4, S5, S6, S7
        df_res: Resolution-sweep data for S3
        peak_compute_specs: Hardware peak compute (TOPS/TFLOPS) for S6
        power_specs: Hardware power specifications for S7
        n_m_global: Fixed benchmark suite size N_M (from CPU baseline). If None, computed from data.
    
    Returns:
        DataFrame with devices as rows and sub-scores (S1-S7) as columns
    """
    # Calculate all sub-scores
    # S1: use n_m_global if provided (from CPU baseline, fixed benchmark suite size)
    s1_scores, N_M_used = calculate_s1_coverage(df_base, n_m_global)
    s2_scores = calculate_s2_efficiency(df_base)
    s4_scores = calculate_s4_accuracy_retention(df_base)
    s5_scores = calculate_s5_throughput_gain(df_base)
    
    # S3: use resolution-sweep data
    s3_scores = calculate_s3_scaling(df_res)
    
    # S6: optional peak-compute-normalized score (use base suite)
    s6_scores = calculate_s6_peak_compute_efficiency(df_base, peak_compute_specs)
    
    # S7: optional power-normalized score (use base suite)
    s7_scores = calculate_s7_power_normalized_efficiency(df_base, power_specs)
    
    # Combine into DataFrame
    # Get all devices from both datasets, excluding SoC devices (Apple M4 CPU/ANE)
    # Apple M4 CPU is the baseline, Apple M4 ANE is SoC with no separate power/compute specs
    EXCLUDED_DEVICES = {'Apple M4 CPU', 'Apple M4 ANE'}
    all_devices = (set(s1_scores.keys()) | set(s3_scores.keys())) - EXCLUDED_DEVICES
    
    results = []
    for device in all_devices:
        row = {
            'Device': device,
            'S1_Coverage': s1_scores.get(device, 0.0),
            'S2_Efficiency': s2_scores.get(device, 0.0),
            'S3_Scaling': s3_scores.get(device, 0.0),
            'S4_AccuracyRetention': s4_scores.get(device, 0.0),
            'S5_ThroughputGain': s5_scores.get(device, 0.0),
            'S6_PeakComputeEff': s6_scores.get(device),
            'S7_PowerEff': s7_scores.get(device),
        }
        
        # Add metadata from base suite
        device_df_base = df_base[df_base['device'] == device]
        device_df_res = df_res[df_res['device'] == device]
        
        # N_M_used is the benchmark suite size used for S1 (should be fixed from CPU baseline)
        row['N_M_Global'] = N_M_used
        row['N_FeasibleBaseTasks'] = (device_df_base['feasible'] == 1).sum()
        row['N_ResTasks'] = len(device_df_res)
        row['N_FeasibleResTasks'] = (device_df_res['feasible'] == 1).sum()
        row['N_Models_Base'] = device_df_base['base_model'].nunique()
        row['N_FeasibleModels_Base'] = device_df_base[device_df_base['feasible'] == 1]['base_model'].nunique()
        row['PeakCompute_TOPS'] = peak_compute_specs.get(device)
        row['Power_W'] = power_specs.get(device)
        
        results.append(row)
    
    results_df = pd.DataFrame(results)
    
    # Sort by S1_Coverage descending (default ordering)
    results_df = results_df.sort_values('S1_Coverage', ascending=False)
    
    return results_df


# ============================================================================
# OUTPUT AND REPORTING
# ============================================================================

def print_uds_summary(uds_df: pd.DataFrame):
    """Print formatted UDS summary to console"""
    print("\n" + "="*80)
    print("UDS SUB-SCORES (S1~S7) SUMMARY")
    print("="*80)
    print(f"\nParameters:")
    print(f"  - Max accuracy drop (τ_acc): {TAU_ACC}%")
    print(f"  - Min speedup (θ): {THETA_SPEEDUP}x")
    print(f"  - Min absolute accuracy: {A_MIN}%")
    print(f"\nNote: s_max, u_max, e^peak_min/max, e^watt_min/max are computed from observed data")
    print("\n" + "-"*80)
    
    for _, row in uds_df.iterrows():
        print(f"\n{row['Device']}")
        print(f"  S1 (Coverage):           {row['S1_Coverage']:.4f}")
        print(f"  S2 (Efficiency):         {row['S2_Efficiency']:.4f}")
        print(f"  S3 (Scaling):            {row['S3_Scaling']:.4f}")
        print(f"  S4 (Accuracy Retention): {row['S4_AccuracyRetention']:.4f}")
        print(f"  S5 (Throughput Gain):    {row['S5_ThroughputGain']:.4f}")
        if row['S6_PeakComputeEff'] is not None and not np.isnan(row['S6_PeakComputeEff']):
            print(f"  S6 (Peak Compute Eff):   {row['S6_PeakComputeEff']:.4f}")
        else:
            print(f"  S6 (Peak Compute Eff):   N/A")
        if row['S7_PowerEff'] is not None and not np.isnan(row['S7_PowerEff']):
            print(f"  S7 (Power Eff):          {row['S7_PowerEff']:.4f}")
        else:
            print(f"  S7 (Power Eff):          N/A")
        print(f"  ---")
        # N_M_Global is the TOTAL benchmark variants (67) - same for all devices per UDS.tex
        # N_FeasibleBaseTasks is the numerator of S1 for this device
        print(f"  Base Suite: {row['N_FeasibleBaseTasks']}/{row['N_M_Global']} feasible variants (S1), "
              f"{row['N_FeasibleModels_Base']}/{row['N_Models_Base']} base models")
        print(f"  Resolution Sweep: {row['N_FeasibleResTasks']}/{row['N_ResTasks']} feasible tasks")
        # Hardware specs
        hw_info = []
        if row.get('PeakCompute_TOPS') is not None and not np.isnan(row.get('PeakCompute_TOPS', float('nan'))):
            hw_info.append(f"Peak: {row['PeakCompute_TOPS']} TOPS")
        if row.get('Power_W') is not None and not np.isnan(row.get('Power_W', float('nan'))):
            hw_info.append(f"Power: {row['Power_W']} W")
        if hw_info:
            print(f"  Hardware: {', '.join(hw_info)}")
    
    print("\n" + "="*80)


def save_detailed_results(df: pd.DataFrame, output_path: str, dataset_type: str):
    """Save detailed per-task results"""
    # Select relevant columns
    output_cols = [
        'device', 'base_model', 'benchmark_model', 'resolution',
        'accuracy', 'accuracy_cpu', 'accuracy_drop',
        'latency_singlestream', 'latency_cpu', 'speedup',
        'throughput_singlestream', 'throughput_offline', 'eta',
        'feasible'
    ]
    
    output_df = df[output_cols].copy()
    output_df = output_df.sort_values(['device', 'base_model', 'resolution'])
    
    output_df.to_csv(output_path, index=False)
    print(f"  [{dataset_type}] Detailed results saved to: {output_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*80)
    print("UDS (Unified Deployment Score) - Sub-Score Calculator (S1~S7)")
    print("="*80)
    print("\nBased on methodology from 'Unified Deployment Score (UDS).tex'")
    print("  - S1, S2, S4, S5, S6, S7: Base suite (canonical r_0=224)")
    print("  - S3: Resolution-sweep subset (multiple resolutions)")
    print("  - Weighted UDS scores are computed in '2. UDS cases.py'")
    print("="*80)
    
    # Get base path (current script directory)
    base_path = Path(__file__).parent
    
    # Load BASE SUITE data (for S1, S2, S6, S7)
    print("\n[1/8] Loading BASE SUITE data (canonical r_0=224)...")
    data_base = load_benchmark_data(base_path, DATA_FOLDERS_BASE, 'base')
    
    if data_base['offline'].empty or data_base['singlestream'].empty:
        print("ERROR: No base suite data loaded. Please check data folders.")
        return
    
    # Load RESOLUTION-SWEEP data (for S3)
    print("\n[2/8] Loading RESOLUTION-SWEEP data (for S3)...")
    data_res = load_benchmark_data(base_path, DATA_FOLDERS_RES, 'resolution')
    
    if data_res['offline'].empty or data_res['singlestream'].empty:
        print("WARNING: No resolution-sweep data loaded. S3 will be 0.")
    
    # Prepare unified datasets
    print("\n[3/8] Preparing unified datasets...")
    print("  Base suite:")
    unified_df_base = prepare_unified_dataset(data_base)
    print(f"    {len(unified_df_base)} rows, {unified_df_base['device'].nunique()} devices, {unified_df_base['base_model'].nunique()} base models")
    
    print("  Resolution sweep:")
    unified_df_res = prepare_unified_dataset(data_res)
    print(f"    {len(unified_df_res)} rows, {unified_df_res['device'].nunique()} devices, {unified_df_res['base_model'].nunique()} base models")
    
    # Get CPU baselines
    print("\n[4/8] Extracting CPU baselines...")
    cpu_baseline_base = get_cpu_baseline(unified_df_base)
    print(f"  Base suite CPU baseline: {len(cpu_baseline_base)} entries")
    
    # CRITICAL: N_M must be fixed from CPU baseline (benchmark suite size)
    # Per UDS.tex, N_M = |M| is the total number of model variants in the suite
    # This ensures S1 denominator is consistent across all devices
    N_M_global = cpu_baseline_base['benchmark_model'].nunique()
    print(f"  Benchmark suite size (N_M): {N_M_global} variants")
    
    cpu_baseline_res = get_cpu_baseline(unified_df_res)
    print(f"  Resolution-sweep CPU baseline: {len(cpu_baseline_res)} entries")
    
    # Calculate comparative metrics
    print("\n[5/8] Calculating comparative metrics...")
    metrics_df_base = calculate_metrics_with_baseline(unified_df_base, cpu_baseline_base, 'base')
    print(f"  Base suite: {len(metrics_df_base)} device-task combinations")
    metrics_df_res = calculate_metrics_with_baseline(unified_df_res, cpu_baseline_res, 'resolution')
    print(f"  Resolution sweep: {len(metrics_df_res)} device-task combinations")
    
    # Calculate UDS scores
    print("\n[6/8] Calculating UDS scores...")
    uds_scores = calculate_uds_scores(
        metrics_df_base, metrics_df_res,
        HARDWARE_PEAK_COMPUTE, HARDWARE_POWER,
        n_m_global=N_M_global  # Pass fixed benchmark suite size from CPU baseline
    )
    
    # Print summary
    print_uds_summary(uds_scores)
    
    # Save results
    print("\n[7/8] Saving results...")
    uds_scores.to_csv(base_path / OUTPUT_SUMMARY_CSV, index=False)
    print(f"  UDS summary saved to: {OUTPUT_SUMMARY_CSV}")
    
    if SAVE_DETAILED_RESULTS:
        save_detailed_results(metrics_df_base, base_path / OUTPUT_CSV_BASE, 'base')
        save_detailed_results(metrics_df_res, base_path / OUTPUT_CSV_RES, 'resolution')
    
    # Print data statistics
    print("\n[8/8] Data statistics...")
    print("\n" + "="*80)
    print("DATA STATISTICS")
    print("="*80)
    print(f"\nBASE SUITE (T_base, r_0=224):")
    print(f"  Total evaluation points: {len(metrics_df_base)}")
    print(f"  With latency (single-stream): {metrics_df_base['latency_singlestream'].notna().sum()}")
    print(f"  With throughput (offline): {metrics_df_base['throughput_offline'].notna().sum()}")
    print(f"  With accuracy: {metrics_df_base['accuracy'].notna().sum()}")
    print(f"  Feasible for UDS: {metrics_df_base['feasible'].sum()}")
    
    print(f"\nRESOLUTION-SWEEP (T_res):")
    print(f"  Total evaluation points: {len(metrics_df_res)}")
    print(f"  With latency (single-stream): {metrics_df_res['latency_singlestream'].notna().sum()}")
    print(f"  With throughput (offline): {metrics_df_res['throughput_offline'].notna().sum()}")
    print(f"  With eta (scaling efficiency): {metrics_df_res['eta'].notna().sum()}")
    print(f"  Feasible for S3: {metrics_df_res['feasible'].sum()}")
    
    print("\n" + "="*80)
    print("Sub-score calculation completed successfully!")
    print("Run '2. UDS cases.py' next to compute weighted UDS scores.")
    print("="*80)


if __name__ == "__main__":
    main()
