#!/usr/bin/env python3
"""
Activation Function Sweep Analysis
Focus: NPU activation function support, latency and accuracy (single-stream scenario)
Baseline: Apple M4 CPU (floating-point reference)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import utilities and central config
from utils import ACCELERATOR_COLORS
from analysis_config import (
    BASELINE_DEVICE,
    ALL_ACCELERATORS,
    NPU_ACCELERATORS,
    ACTIVATION_VARIANT_FOLDERS,
    ACTIVATION_NAMES,
    ACTIVATION_REC_CRITICAL_ACC,
    ACTIVATION_REC_HIGH_DEG,
    ACTIVATION_REC_MOD_DEG,
    BASE_OUTPUT_DIR,
    OUTPUT_SUBDIR_ACTIVATION_SWEEP,
)

# Chart formatting settings
plt.rcParams['figure.max_open_warning'] = 50
plt.rcParams['font.size'] = 11           # Base font size
plt.rcParams['axes.labelsize'] = 12      # Axis labels
plt.rcParams['axes.titlesize'] = 13      # Subplot titles
plt.rcParams['xtick.labelsize'] = 10     # X-axis tick labels
plt.rcParams['ytick.labelsize'] = 10     # Y-axis tick labels
plt.rcParams['legend.fontsize'] = 10     # Legend text
try:
    plt.rcParams['legend.title_fontsize'] = 11 # Legend title
except KeyError:
    pass  # older matplotlib versions don't support this
plt.rcParams['lines.linewidth'] = 1.2
plt.rcParams['axes.linewidth'] = 0.8     # Axis border width
plt.rcParams['figure.titlesize'] = 0
sns.set_palette("Set2")

# Create output directory
OUTPUT_DIR = BASE_OUTPUT_DIR / OUTPUT_SUBDIR_ACTIVATION_SWEEP
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load all activation sweep data files (paths from analysis_config)"""
    print("Loading activation sweep data...")
    
    result = {}
    for key, folder in ACTIVATION_VARIANT_FOLDERS.items():
        result[f'{key}_offline'] = pd.read_csv(f'{folder}/{folder} offline results.csv')
        result[f'{key}_singlestream'] = pd.read_csv(f'{folder}/{folder} single-stream results.csv')
    
    return result

def extract_activation_name(model_name):
    """Extract activation function name from model name"""
    activations = [a.lower() for a in ACTIVATION_NAMES]
    for act in activations:
        if f'_{act}_' in model_name:
            return act.upper()
    return 'UNKNOWN'

def get_hardware_name(row):
    """Get standardized hardware name"""
    if pd.notna(row.get('accelerator_type')):
        return row['accelerator_type']
    return 'Unknown'

def is_npu_hardware(hardware_name):
    """Check if hardware is NPU (not CPU)"""
    cpu_keywords = ['CPU', 'cpu']
    return not any(keyword in hardware_name for keyword in cpu_keywords)

def calculate_accuracy_degradation(data, baseline_hardware=BASELINE_DEVICE):
    """
    Calculate accuracy degradation relative to Apple M4 CPU baseline
    Returns: DataFrame with degradation percentages
    """
    result = []
    
    for model in data['model'].unique() if 'model' in data.columns else [data['benchmark_model'].iloc[0].split('_')[0]]:
        for activation in data['activation'].unique():
            # Get baseline accuracy (Apple M4 CPU)
            baseline_data = data[
                (data['hardware'] == baseline_hardware) & 
                (data['activation'] == activation)
            ]
            
            if baseline_data.empty:
                continue
                
            baseline_acc = baseline_data['accuracy'].mean()
            
            # Calculate degradation for each NPU
            for hardware in data['hardware'].unique():
                if not is_npu_hardware(hardware):
                    continue
                    
                hw_data = data[
                    (data['hardware'] == hardware) & 
                    (data['activation'] == activation)
                ]
                
                if not hw_data.empty:
                    npu_acc = hw_data['accuracy'].mean()
                    degradation = baseline_acc - npu_acc  # Positive = degradation
                    degradation_pct = (degradation / baseline_acc * 100) if baseline_acc > 0 else 0
                    
                    result.append({
                        'Model': model if 'model' in data.columns else 'Unknown',
                        'Hardware': hardware,
                        'Activation': activation,
                        'Baseline_Accuracy': baseline_acc,
                        'NPU_Accuracy': npu_acc,
                        'Accuracy_Degradation': degradation,
                        'Degradation_Percentage': degradation_pct
                    })
    
    return pd.DataFrame(result)

def analyze_activation_support_and_degradation(data, model_name):
    """
    Analyze NPU activation function support and accuracy degradation
    Focus on latency and accuracy (single-stream scenario only)
    Returns data and degradation dataframe (no longer creates individual PNG files)
    """
    print(f"\nAnalyzing {model_name} activation functions...")
    
    # Use only single-stream data (latency and accuracy focused)
    singlestream_data = data.copy()
    singlestream_data['activation'] = singlestream_data['benchmark_model'].apply(extract_activation_name)
    singlestream_data['hardware'] = singlestream_data.apply(get_hardware_name, axis=1)
    singlestream_data['model'] = model_name
    
    # Calculate accuracy degradation vs CPU baseline
    degradation_df = calculate_accuracy_degradation(singlestream_data)
    
    return singlestream_data, degradation_df

def create_combined_accuracy_analysis(mobilenet_data, mobilenet_deg, resnet_data, resnet_deg):
    """
    Create combined accuracy degradation analysis for both models
    Shows only accuracy degradation heatmaps (left charts from original)
    """
    print("\nCreating combined accuracy analysis...")
    
    npu_platforms = list(NPU_ACCELERATORS)
    
    # Create figure with 2 subplots (one per model) - vertical layout for 1-column figure
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    
    # MobileNetV2 (top subplot)
    ax1 = axes[0]
    if not mobilenet_deg.empty:
        heatmap_data = mobilenet_deg.pivot_table(
            index='Activation',
            columns='Hardware',
            values='Degradation_Percentage',
            aggfunc='mean'
        )
        
        # Reorder columns
        column_order = [col for col in npu_platforms if col in heatmap_data.columns]
        heatmap_data = heatmap_data[column_order]
        
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn_r', 
                   ax=ax1, center=0, vmin=-5, vmax=30,
                   cbar_kws={'label': 'Relative Accuracy Drop (%)'})
        ax1.set_xlabel('NPU Platform')
        ax1.set_ylabel('MobileNetV2 Activation')
        
        # Tilt x-axis labels to prevent overlap
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # ResNet18 (bottom subplot)
    ax2 = axes[1]
    if not resnet_deg.empty:
        heatmap_data = resnet_deg.pivot_table(
            index='Activation',
            columns='Hardware',
            values='Degradation_Percentage',
            aggfunc='mean'
        )
        
        # Reorder columns
        column_order = [col for col in npu_platforms if col in heatmap_data.columns]
        heatmap_data = heatmap_data[column_order]
        
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn_r', 
                   ax=ax2, center=0, vmin=-5, vmax=30,
                   cbar_kws={'label': 'Relative Accuracy Drop (%)'})
        ax2.set_xlabel('NPU Platform')
        ax2.set_ylabel('ResNet18 Activation')
        
        # Tilt x-axis labels to prevent overlap
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'combined_accuracy_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: combined_accuracy_analysis.png")

def create_latency_comparison_chart(mobilenet_data, resnet_data):
    """
    Compare latency across all hardware platforms by activation function
    X-axis: Activation functions, Y-axis: Latency
    Shows both MobileNetV2 and ResNet18 separately with grouped bars
    """
    print("\nCreating latency comparison chart...")
    
    # Prepare separate latency data for each model
    mob_latency = mobilenet_data.groupby(['hardware', 'activation'])['sample_latency_average'].mean().reset_index()
    mob_latency.columns = ['hardware', 'activation', 'latency_ms']
    mob_latency['model'] = 'MobileNetV2'
    
    res_latency = resnet_data.groupby(['hardware', 'activation'])['sample_latency_average'].mean().reset_index()
    res_latency.columns = ['hardware', 'activation', 'latency_ms']
    res_latency['model'] = 'ResNet18'
    
    # Get all activations
    activations = sorted(mob_latency['activation'].unique())
    hardware_platforms = list(ALL_ACCELERATORS)
    
    # Create figure for 1-column format
    #fig, ax = plt.subplots(figsize=(6.6, 6))
    fig, ax = plt.subplots(figsize=(12, 5))
    
    
    # Prepare data for plotting
    x = np.arange(len(activations))
    # Use colors from utils.ACCELERATOR_COLORS
    colors = [ACCELERATOR_COLORS.get(hw, '#999999') for hw in hardware_platforms]
    
    # Calculate max latency for log scale determination
    all_latencies = []
    
    # Plot bars for each hardware (grouped by model)
    for i, hardware in enumerate(hardware_platforms):
        # MobileNetV2 bars (solid, more opaque)
        mob_latencies = []
        for activation in activations:
            hw_data = mob_latency[
                (mob_latency['hardware'] == hardware) & 
                (mob_latency['activation'] == activation)
            ]
            if not hw_data.empty:
                val = hw_data['latency_ms'].values[0]
                mob_latencies.append(val)
                all_latencies.append(val)
            else:
                mob_latencies.append(0)
        
        # ResNet18 bars (hatched pattern, slightly transparent)
        res_latencies = []
        for activation in activations:
            hw_data = res_latency[
                (res_latency['hardware'] == hardware) & 
                (res_latency['activation'] == activation)
            ]
            if not hw_data.empty:
                val = hw_data['latency_ms'].values[0]
                res_latencies.append(val)
                all_latencies.append(val)
            else:
                res_latencies.append(0)
        
        # Position bars side by side
        width = 0.05  # Width of each bar (narrower for grouped bars)
        offset_base = (i - len(hardware_platforms)/2 + 0.5) * (width * 1.8)
        
        # MobileNetV2 bar (left, solid)
        ax.bar(x + offset_base - width/2, mob_latencies, width, 
               color=colors[i], alpha=0.9, edgecolor='black', linewidth=0.3)
        
        # ResNet18 bar (right, with hatching)
        ax.bar(x + offset_base + width/2, res_latencies, width,
               color=colors[i], alpha=0.6, hatch='//', edgecolor='black', linewidth=0.3)
    
    ax.set_xlabel('Activation Function')
    ax.set_ylabel('Latency (ms)')
    ax.set_xticks(x)
    ax.set_xticklabels(activations, rotation=45, ha='right')
    
    # Add vertical grid lines to separate activation functions
    ax.grid(True, alpha=0.3, axis='y')
    for i in range(len(activations) + 1):
        ax.axvline(x=i - 0.5, color='gray', linestyle='-', linewidth=1.0, alpha=0.5)
    
    # Create two separate legends for clarity
    from matplotlib.patches import Patch
    
    # Legend 1: Hardware (by color) - moved to upper right to avoid blocking y-axis
    hw_handles = [Patch(facecolor=colors[i], label=hw, alpha=0.8, edgecolor='black', linewidth=0.3) 
                  for i, hw in enumerate(hardware_platforms)]
    legend1 = ax.legend(handles=hw_handles, loc='upper right', title='Hardware', 
                       framealpha=0.95, fontsize=8)
    ax.add_artist(legend1)  # Add first legend manually
    
    # Legend 2: Model (by pattern) - moved to upper left
    model_handles = [
        Patch(facecolor='gray', alpha=0.9, edgecolor='black', linewidth=0.3, label='MobileNetV2'),
        Patch(facecolor='gray', alpha=0.6, hatch='//', edgecolor='black', linewidth=0.3, label='ResNet18')
    ]
    legend2 = ax.legend(handles=model_handles, loc='upper left', title='Model', 
                       framealpha=0.95, fontsize=8)
    
    # Use log scale if there's a large range
    if len(all_latencies) > 0:
        latency_range = max(all_latencies) / min([l for l in all_latencies if l > 0])
        if latency_range > 100:
            ax.set_yscale('log')
            ax.set_ylabel('Latency (ms, log scale)')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'activation_latency_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: activation_latency_comparison.png")
    
    # Save latency data - separate CSV for each model
    mob_pivot = mob_latency.pivot_table(
        index='activation', 
        columns='hardware', 
        values='latency_ms'
    )
    mob_pivot.to_csv(OUTPUT_DIR / 'activation_latency_mobilenetv2.csv')
    print(f"  ✓ Saved: activation_latency_mobilenetv2.csv")
    
    res_pivot = res_latency.pivot_table(
        index='activation', 
        columns='hardware', 
        values='latency_ms'
    )
    res_pivot.to_csv(OUTPUT_DIR / 'activation_latency_resnet18.csv')
    print(f"  ✓ Saved: activation_latency_resnet18.csv")

def create_speedup_comparison_chart(mobilenet_data, resnet_data):
    """
    Compare speedup (vs Apple M4 CPU) across all NPU platforms by activation function
    X-axis: Activation functions, Y-axis: Speedup
    Shows MobileNetV2 and ResNet18 in separate subplots (vertical layout)
    """
    print("\nCreating speedup comparison chart...")
    
    # Get CPU baseline latency for each activation
    mob_cpu = mobilenet_data[mobilenet_data['hardware'] == BASELINE_DEVICE].groupby('activation')['sample_latency_average'].mean()
    res_cpu = resnet_data[resnet_data['hardware'] == BASELINE_DEVICE].groupby('activation')['sample_latency_average'].mean()
    
    # Prepare speedup data for each model
    mob_speedup_list = []
    for _, row in mobilenet_data.iterrows():
        if row['hardware'] != BASELINE_DEVICE and row['activation'] in mob_cpu.index:
            speedup = mob_cpu[row['activation']] / row['sample_latency_average']
            mob_speedup_list.append({
                'hardware': row['hardware'],
                'activation': row['activation'],
                'speedup': speedup,
                'model': 'MobileNetV2'
            })
    mob_speedup = pd.DataFrame(mob_speedup_list)
    
    res_speedup_list = []
    for _, row in resnet_data.iterrows():
        if row['hardware'] != BASELINE_DEVICE and row['activation'] in res_cpu.index:
            speedup = res_cpu[row['activation']] / row['sample_latency_average']
            res_speedup_list.append({
                'hardware': row['hardware'],
                'activation': row['activation'],
                'speedup': speedup,
                'model': 'ResNet18'
            })
    res_speedup = pd.DataFrame(res_speedup_list)
    
    # Aggregate speedup by hardware and activation
    mob_speedup_agg = mob_speedup.groupby(['hardware', 'activation'])['speedup'].mean().reset_index()
    res_speedup_agg = res_speedup.groupby(['hardware', 'activation'])['speedup'].mean().reset_index()
    
    # Get all activations
    activations = sorted(mob_speedup_agg['activation'].unique())
    hardware_platforms = list(NPU_ACCELERATORS)
    
    # Create figure with 2 subplots (vertical layout)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.7, 8))
    
    # Prepare data for plotting
    x = np.arange(len(activations))
    width = 0.1  # Wider bars since we only show one model per subplot
    # Use colors from utils.ACCELERATOR_COLORS (excluding CPU)
    colors = [ACCELERATOR_COLORS.get(hw, '#999999') for hw in hardware_platforms]
    
    # Plot MobileNetV2 (top subplot)
    for i, hardware in enumerate(hardware_platforms):
        mob_speedups = []
        for activation in activations:
            hw_data = mob_speedup_agg[
                (mob_speedup_agg['hardware'] == hardware) & 
                (mob_speedup_agg['activation'] == activation)
            ]
            if not hw_data.empty:
                mob_speedups.append(hw_data['speedup'].values[0])
            else:
                mob_speedups.append(0)
        
        offset = (i - len(hardware_platforms)/2 + 0.5) * width
        ax1.bar(x + offset, mob_speedups, width, 
               color=colors[i], alpha=0.85, edgecolor='black', linewidth=0.3,
               label=hardware)
    
    ax1.set_ylabel('Speedup (log scale)')
    ax1.set_yscale('log')  # Use logarithmic scale for better visualization
    ax1.set_xticks(x)
    ax1.set_xticklabels(activations, rotation=45, ha='right')
    ax1.axhline(y=1, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.grid(True, alpha=0.3, axis='y')
    for i in range(len(activations) + 1):
        ax1.axvline(x=i - 0.5, color='gray', linestyle='-', linewidth=1.0, alpha=0.8)
    ax1.legend(loc='upper right', fontsize=8, title='Hardware', framealpha=0.95)
    ax1.text(0.02, 0.98, 'MobileNetV2', transform=ax1.transAxes, 
             fontsize=11, fontweight='bold', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot ResNet18 (bottom subplot)
    for i, hardware in enumerate(hardware_platforms):
        res_speedups = []
        for activation in activations:
            hw_data = res_speedup_agg[
                (res_speedup_agg['hardware'] == hardware) & 
                (res_speedup_agg['activation'] == activation)
            ]
            if not hw_data.empty:
                res_speedups.append(hw_data['speedup'].values[0])
            else:
                res_speedups.append(0)
        
        offset = (i - len(hardware_platforms)/2 + 0.5) * width
        ax2.bar(x + offset, res_speedups, width, 
               color=colors[i], alpha=0.85, edgecolor='black', linewidth=0.3)
    
    ax2.set_xlabel('Activation Function')
    ax2.set_ylabel('Speedup (log scale)')
    ax2.set_yscale('log')  # Use logarithmic scale for better visualization
    ax2.set_xticks(x)
    ax2.set_xticklabels(activations, rotation=45, ha='right')
    ax2.axhline(y=1, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.grid(True, alpha=0.3, axis='y')
    for i in range(len(activations) + 1):
        ax2.axvline(x=i - 0.5, color='gray', linestyle='-', linewidth=1.0, alpha=0.8)
    # No legend for bottom subplot - already shown in top subplot
    ax2.text(0.02, 0.98, 'ResNet18', transform=ax2.transAxes, 
             fontsize=11, fontweight='bold', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'activation_speedup_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: activation_speedup_comparison.png")
    
    # Save speedup data - separate CSV for each model
    mob_pivot = mob_speedup_agg.pivot_table(
        index='activation', 
        columns='hardware', 
        values='speedup'
    )
    mob_pivot.to_csv(OUTPUT_DIR / 'activation_speedup_mobilenetv2.csv')
    print(f"  ✓ Saved: activation_speedup_mobilenetv2.csv")
    
    res_pivot = res_speedup_agg.pivot_table(
        index='activation', 
        columns='hardware', 
        values='speedup'
    )
    res_pivot.to_csv(OUTPUT_DIR / 'activation_speedup_resnet18.csv')
    print(f"  ✓ Saved: activation_speedup_resnet18.csv")

def create_activation_recommendations(combined_deg):
    """
    Create actionable recommendations for activation function selection
    """
    print("\nCreating activation recommendations...")
    
    
    
    # Create single recommendation figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    fig.suptitle('NPU Activation Function Recommendations', 
                  fontweight='bold', y=0.98)
    
    # Calculate recommendation score for each activation
    # Lower degradation = higher score
    recommendations = []
    
    for activation in combined_deg['Activation'].unique():
        act_data = combined_deg[combined_deg['Activation'] == activation]
        
        avg_degradation = act_data['Degradation_Percentage'].mean()
        max_degradation = act_data['Degradation_Percentage'].max()
        min_accuracy = act_data['NPU_Accuracy'].min()
        
        # Calculate score: penalize high degradation and low accuracy
        if min_accuracy < ACTIVATION_REC_CRITICAL_ACC:  # Critical failure
            score = 0
            category = 'Not Recommended'
        elif avg_degradation > ACTIVATION_REC_HIGH_DEG:  # High degradation
            score = 1
            category = 'Problematic'
        elif avg_degradation > ACTIVATION_REC_MOD_DEG:  # Moderate degradation
            score = 2
            category = 'Acceptable'
        else:  # Low degradation
            score = 3
            category = 'Recommended'
        
        recommendations.append({
            'Activation': activation,
            'Avg_Degradation': avg_degradation,
            'Max_Degradation': max_degradation,
            'Min_Accuracy': min_accuracy,
            'Score': score,
            'Category': category
        })
    
    rec_df = pd.DataFrame(recommendations).sort_values('Score', ascending=False)
    
    # Create horizontal bar chart
    colors = {'Recommended': '#1a9850', 'Acceptable': '#91cf60', 
              'Problematic': '#fee08b', 'Not Recommended': '#d73027'}
    bar_colors = [colors[cat] for cat in rec_df['Category']]
    
    y_pos = np.arange(len(rec_df))
    bars = ax.barh(y_pos, rec_df['Avg_Degradation'], color=bar_colors, alpha=0.85)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(rec_df['Activation'])
    ax.set_xlabel('Average Accuracy Degradation (%)')
    ax.set_ylabel('Activation Function')
    ax.set_title('Ranked by Quantization Robustness (Lower = Better)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.axvline(x=5, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='5% threshold')
    ax.axvline(x=15, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='15% threshold')
    
    # Add value labels
    for i, (bar, row) in enumerate(zip(bars, rec_df.itertuples())):
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
               f'{width:.1f}%', ha='left', va='center')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors['Recommended'], label='Recommended (<5% deg.)'),
        Patch(facecolor=colors['Acceptable'], label='Acceptable (5-15% deg.)'),
        Patch(facecolor=colors['Problematic'], label='Problematic (>15% deg.)'),
        Patch(facecolor=colors['Not Recommended'], label='Not Recommended (acc<10%)')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'activation_recommendations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed recommendations
    rec_df.to_csv(OUTPUT_DIR / 'activation_recommendations.csv', index=False)
    
    print(f"  ✓ Saved: activation_recommendations.png")
    print(f"  ✓ Saved: activation_recommendations.csv")

def generate_summary_statistics(combined_deg):
    """Generate NPU-focused summary statistics"""
    print("\nGenerating summary statistics...")
    
    # Summary by activation function
    activation_summary = combined_deg.groupby('Activation').agg({
        'Degradation_Percentage': ['mean', 'std', 'min', 'max'],
        'NPU_Accuracy': ['mean', 'min'],
        'Baseline_Accuracy': 'mean'
    }).round(2)
    
    activation_summary.columns = ['_'.join(col).strip() for col in activation_summary.columns.values]
    activation_summary = activation_summary.reset_index()
    activation_summary.to_csv(OUTPUT_DIR / 'activation_degradation_summary.csv', index=False)
    
    # Summary by NPU platform
    npu_summary = combined_deg.groupby('Hardware').agg({
        'Degradation_Percentage': ['mean', 'std', 'max'],
        'NPU_Accuracy': ['mean', 'std', 'min'],
    }).round(2)
    
    npu_summary.columns = ['_'.join(col).strip() for col in npu_summary.columns.values]
    npu_summary = npu_summary.reset_index()
    npu_summary.to_csv(OUTPUT_DIR / 'npu_platform_summary.csv', index=False)
    
    # Detailed degradation data
    combined_deg.to_csv(OUTPUT_DIR / 'detailed_degradation_data.csv', index=False)
    
    print(f"  ✓ Saved: activation_degradation_summary.csv")
    print(f"  ✓ Saved: npu_platform_summary.csv")
    print(f"  ✓ Saved: detailed_degradation_data.csv")
    
    print(f"\nKey Statistics:")
    print(f"  - Activations analyzed: {combined_deg['Activation'].nunique()}")
    print(f"  - NPU platforms: {combined_deg['Hardware'].nunique()}")
    print(f"  - Models: {combined_deg['Model'].nunique()}")
    print(f"  - Avg degradation: {combined_deg['Degradation_Percentage'].mean():.2f}%")
    print(f"  - Max degradation: {combined_deg['Degradation_Percentage'].max():.2f}%")

def main():
    """Main analysis pipeline"""
    print("="*80)
    print("NPU Activation Function Support Analysis")
    print("Focus: Quantization impact vs CPU baseline")
    print("="*80)
    
    # Load data
    data = load_data()
    
    # Analyze MobileNet activations (single-stream only: latency + accuracy)
    mobilenet_data, mobilenet_deg = analyze_activation_support_and_degradation(
        data['mobilenet_activation_singlestream'],
        'MobileNetV2'
    )
    
    # Analyze ResNet activations (single-stream only: latency + accuracy)
    resnet_data, resnet_deg = analyze_activation_support_and_degradation(
        data['resnet_activation_singlestream'],
        'ResNet18'
    )
    
    # Combine degradation data
    combined_deg = pd.concat([mobilenet_deg, resnet_deg], ignore_index=True)
    
    # Create combined accuracy analysis chart (replaces individual model charts)
    create_combined_accuracy_analysis(mobilenet_data, mobilenet_deg, resnet_data, resnet_deg)
    
    # Create latency comparison chart
    create_latency_comparison_chart(mobilenet_data, resnet_data)
    
    # Create speedup comparison chart (CPU baseline)
    create_speedup_comparison_chart(mobilenet_data, resnet_data)
    
    # Generate summary statistics
    generate_summary_statistics(combined_deg)
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print(f"Publication-ready charts saved to: {OUTPUT_DIR}")
    print("\nGenerated Charts (3 total):")
    print("  1. combined_accuracy_analysis.png - Accuracy degradation for both models")
    print("  2. activation_latency_comparison.png - Latency comparison across all hardware")
    print("  3. activation_speedup_comparison.png - Speedup vs Apple M4 CPU")
    print("="*80)

if __name__ == "__main__":
    main()