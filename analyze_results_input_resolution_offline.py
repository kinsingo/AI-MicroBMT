"""
Input Resolution Variant Analysis - Offline Throughput
Throughput performance analysis across input resolutions (Single-Stream vs Offline comparison).
"""

from utils import *
from analysis_config import (
    INPUT_RESOLUTION_FOLDER,
    RESOLUTION_MODEL_FAMILIES,
    BASELINE_RESOLUTION,
    KEY_RESOLUTIONS,
    BASE_OUTPUT_DIR,
    OUTPUT_SUBDIR_INPUT_RES_OFFLINE,
)
import re
import numpy as np
import matplotlib.pyplot as plt

# Chart formatting settings
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['lines.linewidth'] = 1.0
plt.rcParams['axes.linewidth'] = 0.7
plt.rcParams['figure.titlesize'] = 0

# Create output directory for input resolution offline analysis
output_dir = BASE_OUTPUT_DIR / OUTPUT_SUBDIR_INPUT_RES_OFFLINE
output_dir.mkdir(exist_ok=True)

# ============================================================================
# Data Loading and Preprocessing
# ============================================================================
def load_input_resolution_offline_data():
    """Load and preprocess input resolution variant offline data"""
    df = pd.read_csv(f'{INPUT_RESOLUTION_FOLDER}/input_variant_offline.csv')
    
    # Extract input resolution as integer (case-insensitive)
    df['input_size'] = pd.to_numeric(df['input_resolution'].str.extract(r'input[_]?[Rr]esolution:(\d+)', re.IGNORECASE)[0], errors='coerce')
    
    # Drop rows with missing input_size
    df = df.dropna(subset=['input_size'])
    df['input_size'] = df['input_size'].astype(int)
    
    # Extract base model family and width multiplier
    def extract_base_model_info(row):
        model_name = row['benchmark_model']
        
        # Extract model family
        if 'mobilenetv2' in model_name.lower():
            family = 'MobileNetV2'
        elif 'resnet101' in model_name.lower():
            family = 'ResNet-101'
        elif 'resnet50' in model_name.lower():
            family = 'ResNet-50'
        elif 'resnet34' in model_name.lower():
            family = 'ResNet-34'
        elif 'resnet18' in model_name.lower():
            family = 'ResNet-18'
        elif 'resnet10' in model_name.lower():
            family = 'ResNet-10'
        else:
            family = 'Unknown'
        
        # Extract width multiplier (default 1.0)
        width_match = re.search(r'w(\d+)_(\d+)', model_name)
        if width_match:
            width = float(f"{width_match.group(1)}.{width_match.group(2)}")
        else:
            width = 1.0
        
        return pd.Series({'model_family': family, 'width_multiplier': width})
    
    df[['model_family', 'width_multiplier']] = df.apply(extract_base_model_info, axis=1)
    
    # Use samples_per_second as the throughput metric
    df['throughput_offline'] = df['samples_per_second']
    
    # Calculate computational complexity (FLOPS increases quadratically with resolution)
    df['relative_flops'] = (df['input_size'] / float(BASELINE_RESOLUTION)) ** 2
    
    print("✓ Input resolution offline data loaded and preprocessed")
    print(f"Accelerator types: {df['accelerator_type'].unique()}")
    print(f"Model families: {df['model_family'].unique()}")
    print(f"Input resolutions: {sorted(df['input_size'].unique())}")
    print(f"Total data points: {len(df)}\n")
    
    return df

def load_input_resolution_singlestream_data():
    """Load single-stream data and calculate throughput from latency"""
    df = pd.read_csv(f'{INPUT_RESOLUTION_FOLDER}/input_variant_singleStream.csv')
    
    # Extract input resolution as integer (case-insensitive)
    df['input_size'] = pd.to_numeric(df['input_resolution'].str.extract(r'input[_]?[Rr]esolution:(\d+)', re.IGNORECASE)[0], errors='coerce')
    
    # Drop rows with missing input_size
    df = df.dropna(subset=['input_size'])
    df['input_size'] = df['input_size'].astype(int)
    
    # Extract base model family and width multiplier
    def extract_base_model_info(row):
        model_name = row['benchmark_model']
        
        # Extract model family
        if 'mobilenetv2' in model_name.lower():
            family = 'MobileNetV2'
        elif 'resnet101' in model_name.lower():
            family = 'ResNet-101'
        elif 'resnet50' in model_name.lower():
            family = 'ResNet-50'
        elif 'resnet34' in model_name.lower():
            family = 'ResNet-34'
        elif 'resnet18' in model_name.lower():
            family = 'ResNet-18'
        elif 'resnet10' in model_name.lower():
            family = 'ResNet-10'
        else:
            family = 'Unknown'
        
        # Extract width multiplier (default 1.0)
        width_match = re.search(r'w(\d+)_(\d+)', model_name)
        if width_match:
            width = float(f"{width_match.group(1)}.{width_match.group(2)}")
        else:
            width = 1.0
        
        return pd.Series({'model_family': family, 'width_multiplier': width})
    
    df[['model_family', 'width_multiplier']] = df.apply(extract_base_model_info, axis=1)
    
    # Convert latency (ms) to throughput (samples/second)
    # throughput = 1000ms / latency_ms = samples per second
    df['throughput_singlestream'] = 1000.0 / df['sample_latency_average']
    
    print("✓ Input resolution single-stream data loaded")
    print(f"Total single-stream data points: {len(df)}\n")
    
    return df[['model_family', 'width_multiplier', 'input_size', 'accelerator_type', 'throughput_singlestream']]

df_offline = load_input_resolution_offline_data()
df_singlestream = load_input_resolution_singlestream_data()

# Merge single-stream and offline data
df_combined = df_offline.merge(
    df_singlestream,
    on=['model_family', 'width_multiplier', 'input_size', 'accelerator_type'],
    how='left'
)

# Calculate multi-core efficiency (speedup from single to offline)
df_combined['multicore_efficiency'] = df_combined['throughput_offline'] / df_combined['throughput_singlestream']

print("="*70)
print("Generating Input Resolution Offline Analysis Charts")
print("="*70)

# ============================================================================
# Chart 1: Single-Stream vs Offline Throughput Comparison
# ============================================================================
def plot_singlestream_vs_offline_throughput():
    print("Generating single-stream vs offline throughput comparison...")
    
    # Select key model families for analysis (width=1.0)
    model_families = RESOLUTION_MODEL_FAMILIES
    df_w1 = df_combined[(df_combined['model_family'].isin(model_families)) & 
                        (df_combined['width_multiplier'] == 1.0)]
    
    # Calculate global y-axis range for consistent scaling
    all_throughput_values = []
    all_throughput_values.extend(df_w1['throughput_singlestream'].dropna().values)
    all_throughput_values.extend(df_w1['throughput_offline'].dropna().values)
    
    global_min = min(all_throughput_values)
    global_max = max(all_throughput_values)
    y_margin = 0.15  # 15% margin
    ylim_min = global_min * (1 - y_margin)
    ylim_max = global_max * (1 + y_margin)
    
    # 1x3 layout for 3 models
    # Adjust figure height to accommodate legend at top
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes = axes.flatten()
    
    # Store legend handles and labels (will be collected from first subplot)
    legend_handles_ss = []
    legend_labels_ss = []
    legend_handles_offline = []
    legend_labels_offline = []
    
    for idx, model_family in enumerate(model_families):
        ax = axes[idx]
        model_data = df_w1[df_w1['model_family'] == model_family]
        
        if len(model_data) == 0:
            ax.text(0.5, 0.5, f'No data for {model_family}', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Plot each accelerator - single-stream and offline with matching colors
        accelerators = sorted(model_data['accelerator_type'].unique())
        for accelerator in accelerators:
            acc_data = model_data[model_data['accelerator_type'] == accelerator]
            acc_data = acc_data.sort_values('input_size')
            
            # Get color for this accelerator (fallback to auto-color if not in palette)
            color = ACCELERATOR_COLORS.get(accelerator, None)
            
            # Single-stream throughput (dashed line, lighter/transparent)
            line_ss, = ax.plot(acc_data['input_size'], acc_data['throughput_singlestream'], 
                   marker='o', linewidth=2, markersize=6, 
                   linestyle='--', alpha=0.4, color=color,
                   label=f'{accelerator} (Single-Stream)')
            
            # Offline throughput (solid line, darker/opaque)
            line_offline, = ax.plot(acc_data['input_size'], acc_data['throughput_offline'], 
                   marker='s', linewidth=2.5, markersize=6, 
                   linestyle='-', alpha=0.9, color=color,
                   label=f'{accelerator} (Offline)')
            
            # Collect legend items from first subplot only
            if idx == 0:
                legend_handles_ss.append(line_ss)
                legend_labels_ss.append(accelerator)
                legend_handles_offline.append(line_offline)
                legend_labels_offline.append(accelerator)
        
        ax.set_xlabel('Input Resolution (pixels)')
        ax.set_ylabel('Throughput (samples/second)')
        panel_labels = ['(a)', '(b)', '(c)']
        ax.text(0.02, 0.98, panel_labels[idx], transform=ax.transAxes, fontweight='bold', va='top')
        # Remove individual legends
        # ax.legend(loc='upper right', ncol=1)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        ax.set_ylim(ylim_min, ylim_max)
    
    # Add shared legend at the top in two rows
    # First row: Single-Stream
    legend1 = fig.legend(legend_handles_ss, legend_labels_ss, 
                        loc='upper center', bbox_to_anchor=(0.5, 1.00),
                        ncol=len(legend_handles_ss), fontsize=9, 
                        frameon=True, title='Single-Stream')
    try:
        fig.add_artist(legend1)
    except AttributeError:
        pass  # older matplotlib: legend already added by fig.legend
    
    # Second row: Offline (positioned below first legend)
    fig.legend(legend_handles_offline, legend_labels_offline,
              loc='upper center', bbox_to_anchor=(0.5, 0.92),
              ncol=len(legend_handles_offline), fontsize=9,
              frameon=True, title='Offline')
    
    # Adjust layout to accommodate top legends
    plt.tight_layout(rect=[0, 0, 1, 0.82])
    plt.savefig(output_dir / 'singlestream_vs_offline_throughput.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Chart 1: Single-Stream vs Offline Throughput saved")

# ============================================================================
# Chart 3: Multi-Core Efficiency Analysis (3 Models)
# ============================================================================
def plot_multicore_efficiency_3_models():
    print("Generating multi-core efficiency analysis (3 models)...")
    
    # Select 3 model families (width=1.0, exclude ResNet-18)
    model_families = RESOLUTION_MODEL_FAMILIES
    df_w1 = df_combined[(df_combined['model_family'].isin(model_families)) & 
                        (df_combined['width_multiplier'] == 1.0) &
                        (df_combined['multicore_efficiency'].notna())]
    
    # Calculate global y-axis range for consistent scaling
    all_efficiency_values = df_w1['multicore_efficiency'].dropna().values
    global_min = min(all_efficiency_values.min(), 1.0)
    global_max = all_efficiency_values.max()
    y_margin = 0.15
    ylim_min = global_min * (1 - y_margin)
    ylim_max = global_max * (1 + y_margin)
    
    # 1x3 layout for 3 models
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    axes = axes.flatten()
    
    for idx, model_family in enumerate(model_families):
        ax = axes[idx]
        model_data = df_w1[df_w1['model_family'] == model_family]
        
        if len(model_data) == 0:
            ax.text(0.5, 0.5, f'No data for {model_family}', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Plot multi-core efficiency for each accelerator with consistent colors
        for accelerator in sorted(model_data['accelerator_type'].unique()):
            acc_data = model_data[model_data['accelerator_type'] == accelerator]
            acc_data = acc_data.sort_values('input_size')
            
            color = ACCELERATOR_COLORS.get(accelerator, None)
            ax.plot(acc_data['input_size'], acc_data['multicore_efficiency'], 
                   marker='o', linewidth=2, markersize=6, label=accelerator, 
                   alpha=0.8, color=color)
        
        # Add reference line at 1.0x (no improvement)
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, 
                  alpha=0.5, label='No Improvement (1.0x)')
        
        ax.set_xlabel('Input Resolution (pixels)')
        ax.set_ylabel('Multi-Core Efficiency (η(r))')
        panel_labels = ['(a)', '(b)', '(c)']
        ax.text(0.02, 0.98, panel_labels[idx], transform=ax.transAxes, fontweight='bold', va='top')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(ylim_min, ylim_max)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'multicore_efficiency_by_resolution_3_models.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Chart 3: Multi-Core Efficiency (3 models) saved")

# ============================================================================
# Summary Statistics
# ============================================================================
def generate_summary_statistics():
    print("Generating summary statistics...")
    
    # Calculate statistics for key resolutions
    key_resolutions = KEY_RESOLUTIONS
    summary_data = []
    
    for model_family in ['MobileNetV2', 'ResNet-50']:
        for resolution in key_resolutions:
            res_data = df_combined[(df_combined['model_family'] == model_family) & 
                                   (df_combined['input_size'] == resolution) &
                                   (df_combined['width_multiplier'] == 1.0)]
            
            if len(res_data) > 0:
                # Calculate statistics per accelerator
                for accelerator in res_data['accelerator_type'].unique():
                    acc_data = res_data[res_data['accelerator_type'] == accelerator]
                    
                    summary_data.append({
                        'model': model_family,
                        'resolution': resolution,
                        'accelerator': accelerator,
                        'throughput_singlestream': acc_data['throughput_singlestream'].mean(),
                        'throughput_offline': acc_data['throughput_offline'].mean(),
                        'multicore_efficiency': acc_data['multicore_efficiency'].mean(),
                        'relative_flops': acc_data['relative_flops'].mean()
                    })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.round(2)
    
    # Save to CSV
    summary_df.to_csv(output_dir / 'offline_summary_statistics.csv', index=False)
    
    print("✓ Summary statistics saved")
    print("\nOffline Throughput Analysis Summary:")
    print(summary_df.head(20).to_string(index=False))

# ============================================================================
# Detailed Multi-Core Efficiency Analysis
# ============================================================================
def analyze_multicore_efficiency_detailed():
    """Detailed offline multi-core scaling analysis with quantitative metrics"""
    print("\n" + "="*70)
    print("OFFLINE MULTI-CORE SCALING ANALYSIS")
    print("="*70)
    
    # Analyze by model and accelerator
    model_families = RESOLUTION_MODEL_FAMILIES
    df_w1 = df_combined[(df_combined['model_family'].isin(model_families)) & 
                        (df_combined['width_multiplier'] == 1.0) &
                        (df_combined['multicore_efficiency'].notna())]
    
    accelerators = df_w1['accelerator_type'].unique()
    
    print("\n" + "="*70)
    print("EFFICIENCY BY RESOLUTION AND ACCELERATOR")
    print("="*70)
    
    for model in model_families:
        print(f"\n{'='*70}")
        print(f"Model: {model}")
        print(f"{'='*70}")
        
        model_data = df_w1[df_w1['model_family'] == model].copy()
        
        for acc in sorted(accelerators):
            acc_data = model_data[model_data['accelerator_type'] == acc].sort_values('input_size')
            
            if len(acc_data) == 0:
                continue
            
            print(f"\n{acc}:")
            print(f"{'Resolution':<12} {'Single(sps)':<15} {'Offline(sps)':<15} {'Efficiency':>12}")
            print("-" * 60)
            
            for _, row in acc_data.iterrows():
                res = int(row['input_size'])
                single = row['throughput_singlestream']
                offline = row['throughput_offline']
                eff = row['multicore_efficiency']
                print(f"{res}x{res:<8} {single:>12.1f}   {offline:>12.1f}   {eff:>10.2f}x")
    
    # Statistical analysis by resolution range
    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS BY RESOLUTION RANGE")
    print("="*70)
    
    resolution_ranges = [
        ("Very Low (48-112)", 48, 112),
        ("Low (128-176)", 128, 176),
        ("Medium (192-256)", 192, 256),
        ("High (272-352)", 272, 352),
        ("Very High (368-448)", 368, 448)
    ]
    
    for range_name, min_res, max_res in resolution_ranges:
        range_data = df_w1[(df_w1['input_size'] >= min_res) & (df_w1['input_size'] <= max_res)]
        
        if len(range_data) == 0:
            continue
        
        print(f"\n{range_name}:")
        print(f"  Efficiency range: {range_data['multicore_efficiency'].min():.2f}x - {range_data['multicore_efficiency'].max():.2f}x")
        print(f"  Mean efficiency: {range_data['multicore_efficiency'].mean():.2f}x")
        print(f"  Std deviation: {range_data['multicore_efficiency'].std():.2f}x")
        
        # By accelerator
        print(f"  By accelerator:")
        for acc in sorted(accelerators):
            acc_range = range_data[range_data['accelerator_type'] == acc]
            if len(acc_range) > 0:
                print(f"    {acc}: {acc_range['multicore_efficiency'].min():.2f}x - {acc_range['multicore_efficiency'].max():.2f}x (mean: {acc_range['multicore_efficiency'].mean():.2f}x)")
    
    # Key observations
    print("\n" + "="*70)
    print("KEY OBSERVATIONS")
    print("="*70)
    
    # MobileNetV2 specific analysis (most measurements)
    mobilenet_data = df_w1[df_w1['model_family'] == 'MobileNetV2'].copy()
    
    print("\nMobileNetV2 Analysis:")
    for acc in sorted(accelerators):
        acc_data = mobilenet_data[mobilenet_data['accelerator_type'] == acc].sort_values('input_size')
        
        if len(acc_data) < 2:
            continue
        
        min_res_row = acc_data.iloc[0]
        max_res_row = acc_data.iloc[-1]
        
        print(f"\n{acc}:")
        print(f"  @ {int(min_res_row['input_size'])}x{int(min_res_row['input_size'])}: {min_res_row['multicore_efficiency']:.2f}x efficiency")
        print(f"  @ {int(max_res_row['input_size'])}x{int(max_res_row['input_size'])}: {max_res_row['multicore_efficiency']:.2f}x efficiency")
        print(f"  Efficiency degradation: {min_res_row['multicore_efficiency'] / max_res_row['multicore_efficiency']:.2f}x")
        
        # Find peak efficiency
        peak_row = acc_data.loc[acc_data['multicore_efficiency'].idxmax()]
        print(f"  Peak efficiency: {peak_row['multicore_efficiency']:.2f}x @ {int(peak_row['input_size'])}x{int(peak_row['input_size'])}")
    
    # Arithmetic intensity context (from roofline analysis)
    print("\n" + "="*70)
    print("ARITHMETIC INTENSITY CONTEXT (MobileNetV2)")
    print("="*70)
    print("Reference values from roofline analysis:")
    print("  32x32:   AI = 0.47 FLOP/Byte (deeply memory-bound)")
    print("  224x224: AI = 23.27 FLOP/Byte (transitional)")
    print("  448x448: AI = 93.09 FLOP/Byte (approaching compute-bound)")
    print("\nRidge points:")
    print("  Hailo-8: 63.7 FLOP/Byte")
    print("  DeepX M1: 66.7 FLOP/Byte")
    print("  Mobilint-ARIES: 312.9 FLOP/Byte")
    
    # Save detailed analysis
    df_w1.to_csv(output_dir / 'offline_efficiency_detailed.csv', index=False)
    print(f"\n✓ Detailed efficiency analysis saved to offline_efficiency_detailed.csv")
    
    print("\n" + "="*70)
    print("MULTI-CORE EFFICIENCY ANALYSIS COMPLETE")
    print("="*70)

# ============================================================================
# Main Execution
# ============================================================================
if __name__ == "__main__":
    plot_singlestream_vs_offline_throughput()
    plot_multicore_efficiency_3_models()
    generate_summary_statistics()
    analyze_multicore_efficiency_detailed()
    
    print("\n" + "="*70)
    print("Input Resolution Offline Analysis Complete!")
    print(f"Charts saved to: {output_dir.absolute()}")
    print("="*70)
