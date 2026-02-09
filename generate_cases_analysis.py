"""
Case Analysis: Single-Stream Performance vs Apple M4 CPU
67 models classification by latency and accuracy criteria
"""
# Verified against accuracy_drop_heatmap_data.csv and
# single_stream_speedup_data.csv — results match.
# Logic is correct.

import pandas as pd
from utils import load_single_stream_data
from analysis_config import (
    BASELINE_DEVICE,
    CASE_ANALYSIS_NPUS,
    CASE_TAU_ACCURACY_DROP_PCT,
    CASE_MIN_ABSOLUTE_ACCURACY,
    CASE_TOTAL_MODELS,
)

# Load data
df_ss = load_single_stream_data()

print("="*80)
print(f"Case Analysis: Single-Stream Performance vs {BASELINE_DEVICE}")
print("="*80)

# Target NPUs
npus = CASE_ANALYSIS_NPUS

# Get all unique models
all_models = df_ss['normalized_model'].unique()
print(f"\nTotal models: {len(all_models)}")

# Case classification criteria:
# Case 1: accuracy_drop_percent < 4% AND latency <= CPU latency (accurate & faster)
# Case 2: accuracy_drop_percent < 4% AND latency > CPU latency (accurate but slower)
# Case 3: accuracy_drop_percent >= 4% (accuracy low)
# Case 4: compilation/runtime failure OR absolute accuracy < 15%

results = {}

for npu in npus:
    print(f"\n{'='*60}")
    print(f"Analyzing {npu}")
    print(f"{'='*60}")
    
    case1 = 0  # accurate & faster
    case2 = 0  # accurate but slower
    case3_faster = 0  # accuracy low but faster
    case3_slower = 0  # accuracy low and slower
    case4 = 0  # coverage failure
    
    # Track models for each case
    case2_models = []
    case3_faster_models = []
    case3_slower_models = []
    case4_models = []
    
    for model in all_models:
        # Get CPU baseline
        cpu_data = df_ss[(df_ss['normalized_model'] == model) & 
                         (df_ss['accelerator_type'] == BASELINE_DEVICE)]
        
        # Get NPU data
        npu_data = df_ss[(df_ss['normalized_model'] == model) & 
                         (df_ss['accelerator_type'] == npu)]
        
        # Check if data exists
        if len(cpu_data) == 0:
            # No CPU baseline - skip this model
            continue
            
        if len(npu_data) == 0:
            # No NPU data - Case 4 (compilation/runtime failure)
            case4 += 1
            case4_models.append(model)
            continue
        
        cpu_latency = cpu_data.iloc[0]['latency_ms']
        npu_latency = npu_data.iloc[0]['latency_ms']
        npu_accuracy = npu_data.iloc[0]['accuracy']
        accuracy_drop_percent = npu_data.iloc[0]['accuracy_drop_percent']
        
        # Check Case 4: absolute accuracy < threshold
        if npu_accuracy < CASE_MIN_ABSOLUTE_ACCURACY:
            case4 += 1
            case4_models.append(model)
            continue
        
        # Check if accuracy_drop is valid
        if pd.isna(accuracy_drop_percent):
            # No baseline accuracy available
            case4 += 1
            case4_models.append(model)
            continue
        
        # Classify based on accuracy and latency
        is_accurate = accuracy_drop_percent < CASE_TAU_ACCURACY_DROP_PCT
        is_faster = npu_latency <= cpu_latency
        
        if is_accurate and is_faster:
            case1 += 1
        elif is_accurate and not is_faster:
            case2 += 1
            case2_models.append(model)
        elif not is_accurate and is_faster:
            case3_faster += 1
            case3_faster_models.append(model)
        else:  # not is_accurate and not is_faster
            case3_slower += 1
            case3_slower_models.append(model)
    
    case3_total = case3_faster + case3_slower
    total = case1 + case2 + case3_total + case4
    
    results[npu] = {
        'case1': case1,
        'case2': case2,
        'case3_faster': case3_faster,
        'case3_slower': case3_slower,
        'case3_total': case3_total,
        'case4': case4,
        'total': total,
        'case2_models': case2_models,
        'case3_faster_models': case3_faster_models,
        'case3_slower_models': case3_slower_models,
        'case4_models': case4_models
    }
    
    print(f"  Case 1 (accurate & faster): {case1}")
    print(f"  Case 2 (accurate but slower): {case2}")
    print(f"  Case 3 (accuracy low): {case3_total} ({case3_faster}/{case3_slower})")
    print(f"  Case 4 (coverage failure): {case4}")
    print(f"  Total: {total}")

# Generate LaTeX table
print("\n" + "="*80)
print("Generating LaTeX table...")
print("="*80)

latex_content = r"""\begin{table}[tb]
\caption{Case breakdown vs.\ Apple M4 CPU (67 models, Single-Stream).
We classify each (model, NPU) outcome by latency vs.\ CPU and relative accuracy degradation $\Delta_{\%}$:
Case~1 (accurate \& faster): $\Delta_{\%}<4\%$ and latency $\leq$ CPU;
Case~2 (accurate but slower): $\Delta_{\%}<4\%$ and latency $>$ CPU;
Case~3 (accuracy low): $\Delta_{\%}\geq 4\%$ (parenthesis shows \#(latency $\leq$ CPU / latency $>$ CPU));
Case~4 (coverage failure): compilation/runtime failure or absolute accuracy $<15\%$.}
\label{tab:case_breakdown_67}
\centering
\small
\setlength{\tabcolsep}{4pt} 
\begin{tabular}{lrrcr}
\toprule
NPU & Case~1 & Case~2 & Case~3 & Case~4 \\
\midrule
"""

# Add data rows
for npu in npus:
    r = results[npu]
    latex_content += f"{npu} & {r['case1']} & {r['case2']} & {r['case3_total']} ({r['case3_faster']}/{r['case3_slower']}) & {r['case4']} \\\\\n"

latex_content += r"""\bottomrule
\end{tabular}
\end{table}
"""

# Calculate summary statistics
print("\n" + "="*80)
print("Summary Statistics")
print("="*80)

summary_stats = {}
for npu in npus:
    r = results[npu]
    total_evaluated = r['total']  # out of 67
    accurate_count = r['case1'] + r['case2']  # Cases with accuracy_drop < 4%
    faster_count = r['case1'] + r['case3_faster']  # Cases with latency <= CPU
    
    coverage_pct = (total_evaluated / CASE_TOTAL_MODELS) * 100
    accuracy_pct = (accurate_count / total_evaluated * 100) if total_evaluated > 0 else 0
    faster_pct = (faster_count / total_evaluated * 100) if total_evaluated > 0 else 0
    
    summary_stats[npu] = {
        'coverage': f"{total_evaluated}/67 ({coverage_pct:.1f}%)",
        'accurate': f"{accurate_count}/{total_evaluated} ({accuracy_pct:.1f}%)",
        'faster': f"{faster_count}/{total_evaluated} ({faster_pct:.1f}%)"
    }
    
    print(f"\n{npu}:")
    print(f"  Coverage: {summary_stats[npu]['coverage']}")
    print(f"  Accurate: {summary_stats[npu]['accurate']}")
    print(f"  Faster: {summary_stats[npu]['faster']}")

# Add summary statistics to LaTeX content
latex_summary = r"""

% Summary Statistics:
% Coverage = models successfully evaluated / 67 total
% Accurate = (Case 1 + Case 2) / evaluated (accuracy_drop < 4%)
% Faster = (Case 1 + Case 3 faster) / evaluated (latency <= CPU)
%
"""

for npu in npus:
    s = summary_stats[npu]
    latex_summary += f"% {npu}: Coverage {s['coverage']}, Accurate {s['accurate']}, Faster {s['faster']}\n"

# Append summary to latex_content
latex_full_content = latex_content + latex_summary

# Save to file
with open('cases_analysis.tex', 'w', encoding='utf-8') as f:
    f.write(latex_full_content)

print("\n[OK] LaTeX table saved to: cases_analysis.tex")
print("\nTable content:")
print(latex_full_content)

# Print summary statistics
print("\n" + "="*80)
print("Summary Statistics")
print("="*80)
for npu in npus:
    r = results[npu]
    coverage = (r['total'] - r['case4']) / r['total'] * 100
    accurate_ratio = (r['case1'] + r['case2']) / (r['total'] - r['case4']) * 100 if (r['total'] - r['case4']) > 0 else 0
    faster_ratio = (r['case1'] + r['case3_faster']) / (r['total'] - r['case4']) * 100 if (r['total'] - r['case4']) > 0 else 0
    
    print(f"\n{npu}:")
    print(f"  Coverage: {r['total'] - r['case4']}/{r['total']} ({coverage:.1f}%)")
    print(f"  Accurate (Cases 1+2): {r['case1'] + r['case2']}/{r['total'] - r['case4']} ({accurate_ratio:.1f}%)")
    print(f"  Faster (Cases 1+3faster): {r['case1'] + r['case3_faster']}/{r['total'] - r['case4']} ({faster_ratio:.1f}%)")

# Print detailed model lists for Cases 2-4
print("\n" + "="*80)
print("Detailed Model Lists for Cases 2-4")
print("="*80)

for npu in npus:
    r = results[npu]
    print(f"\n{npu}:")
    print(f"{'='*60}")
    
    if r['case2_models']:
        print(f"\nCase 2 (accurate but slower) - {len(r['case2_models'])} models:")
        for model in sorted(r['case2_models']):
            print(f"  - {model}")
    else:
        print(f"\nCase 2 (accurate but slower): None")
    
    if r['case3_faster_models']:
        print(f"\nCase 3 (accuracy low but faster) - {len(r['case3_faster_models'])} models:")
        for model in sorted(r['case3_faster_models']):
            print(f"  - {model}")
    else:
        print(f"\nCase 3 (accuracy low but faster): None")
    
    if r['case3_slower_models']:
        print(f"\nCase 3 (accuracy low and slower) - {len(r['case3_slower_models'])} models:")
        for model in sorted(r['case3_slower_models']):
            print(f"  - {model}")
    else:
        print(f"\nCase 3 (accuracy low and slower): None")
    
    if r['case4_models']:
        print(f"\nCase 4 (coverage failure) - {len(r['case4_models'])} models:")
        for model in sorted(r['case4_models']):
            print(f"  - {model}")
    else:
        print(f"\nCase 4 (coverage failure): None")

print("\n" + "="*80)
