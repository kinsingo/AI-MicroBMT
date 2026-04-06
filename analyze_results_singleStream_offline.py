"""
Radar Chart Analysis: Single-Stream Speed-up & Offline Throughput
"""

from utils import *
from analysis_config import (
    BASELINE_DEVICE,
    ALL_ACCELERATORS,
    BASE_OUTPUT_DIR,
    OUTPUT_SUBDIR_SINGLESTREAM_VS_OFFLINE,
)
import numpy as np
import matplotlib.pyplot as plt
from math import pi

# Chart formatting settings
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9

# Create output directory
output_dir = BASE_OUTPUT_DIR / OUTPUT_SUBDIR_SINGLESTREAM_VS_OFFLINE
output_dir.mkdir(exist_ok=True)

# ============================================================================
# Load Data
# ============================================================================
# Load data from utils (includes activation variants)
df_ss = load_single_stream_data()
df_offline_temp = load_offline_data()

# For models without offline data, raise error instead of calculating from single-stream
offline_models_acc = set((row['normalized_model'], row['accelerator_type']) 
                         for _, row in df_offline_temp.iterrows())

missing_offline_combinations = []
for _, row in df_ss.iterrows():
    model_name = row['normalized_model']
    acc_type = row['accelerator_type']
    
    # Check if this model+accelerator combo exists in offline data
    if (model_name, acc_type) not in offline_models_acc:
        missing_offline_combinations.append((model_name, acc_type))

# If there are missing offline combinations, report and raise error
if missing_offline_combinations:
    print("\n" + "="*70)
    print("❌ ERROR: Missing Offline Data")
    print("="*70)
    print(f"Found {len(missing_offline_combinations)} model+accelerator combinations")
    print("that have single-stream data but NO offline data:\n")
    
    # Group by accelerator for better readability
    from collections import defaultdict
    by_accelerator = defaultdict(list)
    for model, acc in missing_offline_combinations:
        by_accelerator[acc].append(model)
    
    for acc in sorted(by_accelerator.keys()):
        print(f"\n{acc}:")
        for model in sorted(by_accelerator[acc]):
            print(f"  - {model}")
    
    print("\n" + "="*70)
    raise ValueError(f"Missing offline data for {len(missing_offline_combinations)} combinations. "
                     "Please ensure all single-stream models have corresponding offline data.")

df_offline = df_offline_temp

print("="*70)
print("Generating Radar Chart Analysis (with Activation Variants)")
print("="*70)
print(f"Total single-stream models: {len(df_ss['normalized_model'].unique())}")
print(f"Total offline models: {len(df_offline['normalized_model'].unique())}")
print(f"✓ All single-stream models have corresponding offline data")
print("="*70)


# ============================================================================
# Chart: Combined Side-by-Side Radar Chart with CSV Export
# ============================================================================
def plot_combined_radar_chart_with_scaling(scaling_method='log10'):
    """
    Generate combined radar chart with different scaling methods.
    
    Args:
        scaling_method: 'linear', 'log10', 'sqrt'
    """
    print(f"\nGenerating combined side-by-side radar chart (scaling: {scaling_method})...")
    
    # Target accelerators for both single-stream and offline
    accelerators_ss = list(ALL_ACCELERATORS)
    
    # Offline accelerators - now includes Apple M4 CPU and ANE
    accelerators_offline = list(ALL_ACCELERATORS)
    
    # Get all models from single-stream data
    all_models = sorted(df_ss['normalized_model'].unique())
    
    # ========== Calculate Single-Stream Throughput Data ==========
    throughput_ss_data = {}
    for acc in accelerators_ss:
        throughput_ss_data[acc] = []
    
    for model in all_models:
        for acc in accelerators_ss:
            acc_data = df_ss[(df_ss['normalized_model'] == model) & 
                            (df_ss['accelerator_type'] == acc)]
            
            if len(acc_data) > 0:
                latency = acc_data.iloc[0]['sample_latency_average']
                throughput = 1000.0 / latency
                throughput_ss_data[acc].append({'model': model, 'throughput': throughput})
            else:
                throughput_ss_data[acc].append({'model': model, 'throughput': 0})
    
    # Apply transformation based on scaling method
    def apply_scaling(value, method):
        if value <= 0:
            return 0
        if method == 'linear':
            return value
        elif method == 'log10':
            return np.log10(value + 1)
        elif method == 'sqrt':
            return np.sqrt(value)
        else:
            return value
    
    for acc in accelerators_ss:
        for item in throughput_ss_data[acc]:
            item['throughput_scaled_raw'] = apply_scaling(item['throughput'], scaling_method)
    
    # ========== Calculate Offline Throughput Data ==========
    throughput_offline_data = {}
    for acc in accelerators_offline:
        throughput_offline_data[acc] = []
    
    for model in all_models:
        for acc in accelerators_offline:
            offline_data = df_offline[(df_offline['normalized_model'] == model) & 
                                      (df_offline['accelerator_type'] == acc)]
            
            if len(offline_data) > 0:
                throughput = offline_data.iloc[0]['samples_per_second']
                throughput_offline_data[acc].append({'model': model, 'throughput': throughput})
            else:
                ss_data = df_ss[(df_ss['normalized_model'] == model) & 
                               (df_ss['accelerator_type'] == acc)]
                
                if len(ss_data) > 0:
                    latency = ss_data.iloc[0]['sample_latency_average']
                    throughput = 1000.0 / latency
                    throughput_offline_data[acc].append({'model': model, 'throughput': throughput})
                else:
                    throughput_offline_data[acc].append({'model': model, 'throughput': 0})
    
    # Apply transformation for offline throughput
    for acc in accelerators_offline:
        for item in throughput_offline_data[acc]:
            item['throughput_scaled_raw'] = apply_scaling(item['throughput'], scaling_method)
    
    # ========== Unified Scaling ==========
    # Find maximum scaled throughput across both single-stream and offline data
    max_scaled_ss = max([item['throughput_scaled_raw'] for acc in accelerators_ss for item in throughput_ss_data[acc]])
    max_scaled_offline = max([item['throughput_scaled_raw'] for acc in accelerators_offline for item in throughput_offline_data[acc]])
    max_scaled = max(max_scaled_ss, max_scaled_offline)
    
    # Calculate unified scaling factor to make max scaled throughput = 100
    scaling_factor = 100.0 / max_scaled if max_scaled > 0 else 1.0
    
    print(f"Max single-stream throughput: {max([item['throughput'] for acc in accelerators_ss for item in throughput_ss_data[acc]]):.2f} samples/sec")
    print(f"Max offline throughput: {max([item['throughput'] for acc in accelerators_offline for item in throughput_offline_data[acc]]):.2f} samples/sec")
    print(f"Using {scaling_method} scale for both charts")
    print(f"Unified scaling factor: {scaling_factor:.6f}")
    
    # Apply unified scaling factor to both datasets
    for acc in accelerators_ss:
        for item in throughput_ss_data[acc]:
            item['throughput_scaled'] = item['throughput_scaled_raw'] * scaling_factor
    
    for acc in accelerators_offline:
        for item in throughput_offline_data[acc]:
            item['throughput_scaled'] = item['throughput_scaled_raw'] * scaling_factor
    
    # ========== Save CSV Files ==========
    # Save single-stream data to CSV
    csv_data_ss = []
    for acc in accelerators_ss:
        for item in throughput_ss_data[acc]:
            csv_data_ss.append({
                'accelerator': acc,
                'model': item['model'],
                'throughput_original': item['throughput'],
                'throughput_scaled': item['throughput_scaled']
            })
    
    pd.DataFrame(csv_data_ss).to_csv(output_dir / 'single_stream_throughput_data.csv', index=False)
    print(f"✓ Saved single-stream throughput data ({len(csv_data_ss)} data points)")
    
    # Save offline data to CSV
    csv_data_offline = []
    for acc in accelerators_offline:
        for item in throughput_offline_data[acc]:
            csv_data_offline.append({
                'accelerator': acc,
                'model': item['model'],
                'throughput_original': item['throughput'],
                'throughput_scaled': item['throughput_scaled']
            })
    
    pd.DataFrame(csv_data_offline).to_csv(output_dir / 'offline_throughput_data.csv', index=False)
    print(f"✓ Saved offline throughput data ({len(csv_data_offline)} data points)")
    
    # ========== Create Combined Plot ==========
    models = sorted([item['model'] for item in throughput_ss_data[accelerators_ss[0]]])
    num_vars = len(models)
    
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(20, 10))
    
    # Left subplot: Single-Stream Throughput
    ax1 = fig.add_subplot(121, projection='polar')
    
    for acc in accelerators_ss:
        values = [item['throughput_scaled'] for item in throughput_ss_data[acc]]
        values += values[:1]
        
        color = ACCELERATOR_COLORS.get(acc, 'gray')
        ax1.plot(angles, values, 'o-', linewidth=2, label=acc, color=color)
        ax1.fill(angles, values, alpha=0.15, color=color)
    
    ax1.set_theta_offset(pi / 2)
    ax1.set_theta_direction(-1)
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(models, size=8)
    ax1.set_ylim(0, 100)
    ax1.set_yticks([20, 40, 60, 80, 100])
    ax1.set_yticklabels(['20', '40', '60', '80', '100'], size=9)
    ax1.set_title('(a) Single-Stream Throughput', size=12, pad=15)
    
    # Right subplot: Offline Throughput
    ax2 = fig.add_subplot(122, projection='polar')
    
    for acc in accelerators_offline:
        values = [item['throughput_scaled'] for item in throughput_offline_data[acc]]
        values += values[:1]
        
        color = ACCELERATOR_COLORS.get(acc, 'gray')
        ax2.plot(angles, values, 'o-', linewidth=2, label=acc, color=color)
        ax2.fill(angles, values, alpha=0.15, color=color)
    
    ax2.set_theta_offset(pi / 2)
    ax2.set_theta_direction(-1)
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(models, size=8)
    ax2.set_ylim(0, 100)
    ax2.set_yticks([20, 40, 60, 80, 100])
    ax2.set_yticklabels(['20', '40', '60', '80', '100'], size=9)
    ax2.set_title('(b) Offline Throughput', size=12, pad=15)
    
    # Add single legend at the top center
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), 
               ncol=3, fontsize=12, frameon=True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / f'combined_radar_chart_{scaling_method}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Chart: Combined side-by-side radar chart ({scaling_method}) saved")

def plot_combined_radar_chart():
    """Generate radar charts with multiple scaling methods"""
    # Generate charts with different scaling methods
    plot_combined_radar_chart_with_scaling('linear')  # No transformation
    plot_combined_radar_chart_with_scaling('sqrt')    # Square root
    plot_combined_radar_chart_with_scaling('log10')   # log10

# ============================================================================
# Chart: Accuracy Drop Heatmap by Accelerator and Model
# ============================================================================
def plot_accuracy_drop_heatmap():
    print("\nGenerating accuracy drop heatmap...")
    
    # Filter out baseline device and models without baseline
    filtered_df = df_ss[(df_ss['accelerator_type'] != BASELINE_DEVICE) & 
                        (df_ss['baseline_accuracy'].notna())]
    
    print(f"Heatmap data points: {len(filtered_df)}")
    
    # Save detailed data to CSV
    heatmap_data = filtered_df[['normalized_model', 'model_family', 'accelerator_type', 
                                 'accuracy', 'baseline_accuracy', 'accuracy_drop', 
                                 'accuracy_drop_percent']].copy()
    heatmap_data.to_csv(output_dir / 'accuracy_drop_heatmap_data.csv', index=False)
    print(f"✓ Saved heatmap data to CSV ({len(heatmap_data)} data points)")
    
    # Create pivot table for heatmap using normalized model names
    pivot_data = filtered_df.pivot_table(
        index='normalized_model', 
        columns='accelerator_type', 
        values='accuracy_drop_percent',
        aggfunc='mean'
    )
    
    # Sort by model family and width
    model_order = sorted(pivot_data.index, 
                        key=lambda x: (extract_model_family(x) if extract_model_family(x) != 'Unknown' else 'ZZZ', 
                                      extract_width_multiplier(x) if extract_width_multiplier(x) is not None else 0))
    pivot_data = pivot_data.reindex(model_order)
    
    fig, ax = plt.subplots(figsize=(12, 18))
    sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlGn_r', 
                center=0, vmin=-2, vmax=10, ax=ax, cbar_kws={'label': 'Accuracy Drop (%)'})
    
    ax.set_xlabel('Accelerator Type')
    ax.set_ylabel('Benchmark Model')
    
    # Simplify y-axis labels
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_drop_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Chart: Accuracy Drop Heatmap saved")

# ============================================================================
# Main Execution
# ============================================================================
if __name__ == "__main__":
    plot_combined_radar_chart()
    plot_accuracy_drop_heatmap()
    
    print("\n" + "="*70)
    print("Analysis Complete!")
    print(f"Charts saved to: {output_dir.absolute()}")
    print("="*70)

