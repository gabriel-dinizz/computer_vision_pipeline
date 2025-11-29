#!/usr/bin/env python3
"""
Export benchmark results in multiple useful formats
"""
import json
import csv
from pathlib import Path

# Load results
with open('results_comprehensive/results_complete.json') as f:
    data = json.load(f)

results = data['results']

# Create exports directory
export_dir = Path('results_comprehensive/exports')
export_dir.mkdir(exist_ok=True)

# ============================================================================
# 1. SUMMARY TABLE (for thesis)
# ============================================================================
with open(export_dir / 'summary_table.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Resolution', 'Filter', '1T (ms)', '2T (ms)', '4T (ms)', '8T (ms)',
                     'Speedup (8T)', 'Efficiency (8T)', 'Performance Gain (%)'])

    for res in ['480p', '720p', '1080p', '1440p']:
        if res not in results:
            continue
        for filt in results[res]:
            t1 = results[res][filt].get('1', {}).get('mean_ms', 0)
            t2 = results[res][filt].get('2', {}).get('mean_ms', 0)
            t4 = results[res][filt].get('4', {}).get('mean_ms', 0)
            t8 = results[res][filt].get('8', {}).get('mean_ms', 0)
            speedup = results[res][filt].get('8', {}).get('speedup', 0)
            efficiency = results[res][filt].get('8', {}).get('efficiency', 0)
            gain = ((t1 - t8) / t1 * 100) if t1 > 0 else 0

            writer.writerow([res, filt, f'{t1:.1f}', f'{t2:.1f}', f'{t4:.1f}',
                           f'{t8:.1f}', f'{speedup:.2f}x', f'{efficiency:.2%}', f'{gain:.1f}%'])

print("✅ Created: summary_table.csv")

# ============================================================================
# 2. BEST PERFORMERS (for highlights)
# ============================================================================
with open(export_dir / 'best_performers.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Rank', 'Filter', 'Resolution', 'Speedup', 'Efficiency', 'Time Saved (ms)'])

    performers = []
    for res in results:
        for filt in results[res]:
            if '8' in results[res][filt] and '1' in results[res][filt]:
                data_8t = results[res][filt]['8']
                data_1t = results[res][filt]['1']
                if 'speedup' in data_8t:
                    speedup = data_8t['speedup']
                    efficiency = data_8t.get('efficiency', 0)
                    time_saved = data_1t['mean_ms'] - data_8t['mean_ms']
                    performers.append((filt, res, speedup, efficiency, time_saved))

    # Sort by speedup
    performers.sort(key=lambda x: x[2], reverse=True)

    for rank, (filt, res, speedup, eff, saved) in enumerate(performers, 1):
        writer.writerow([rank, filt, res, f'{speedup:.2f}x', f'{eff:.2%}', f'{saved:.1f}'])

print("✅ Created: best_performers.csv")

# ============================================================================
# 3. VARIANCE ANALYSIS (for methodology discussion)
# ============================================================================
with open(export_dir / 'variance_analysis.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Resolution', 'Filter', 'Threads', 'Mean (ms)', 'Std Dev (ms)',
                     'Variance %', 'Min (ms)', 'Max (ms)', 'Status'])

    for res in results:
        for filt in results[res]:
            for threads in results[res][filt]:
                d = results[res][filt][threads]
                if 'mean_ms' in d and 'std_ms' in d:
                    variance_pct = (d['std_ms'] / d['mean_ms'] * 100) if d['mean_ms'] > 0 else 0
                    status = 'Good' if variance_pct < 10 else 'High' if variance_pct < 25 else 'Very High'

                    writer.writerow([res, filt, f'{threads}T', f"{d['mean_ms']:.1f}",
                                   f"{d['std_ms']:.1f}", f"{variance_pct:.1f}%",
                                   f"{d.get('min_ms', 0):.1f}", f"{d.get('max_ms', 0):.1f}", status])

print("✅ Created: variance_analysis.csv")

# ============================================================================
# 4. THREAD SCALING TABLE (for analysis section)
# ============================================================================
with open(export_dir / 'thread_scaling.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Filter', 'Resolution', '1→2T Speedup', '1→4T Speedup', '1→8T Speedup',
                     '2T Efficiency', '4T Efficiency', '8T Efficiency'])

    for res in results:
        for filt in results[res]:
            if '1' in results[res][filt]:
                t1 = results[res][filt]['1']['mean_ms']

                speedup_2 = results[res][filt].get('2', {}).get('speedup', 0) or (t1 / results[res][filt].get('2', {}).get('mean_ms', t1))
                speedup_4 = results[res][filt].get('4', {}).get('speedup', 0)
                speedup_8 = results[res][filt].get('8', {}).get('speedup', 0)

                eff_2 = results[res][filt].get('2', {}).get('efficiency', speedup_2/2) if speedup_2 > 0 else 0
                eff_4 = results[res][filt].get('4', {}).get('efficiency', 0)
                eff_8 = results[res][filt].get('8', {}).get('efficiency', 0)

                writer.writerow([filt, res, f'{speedup_2:.2f}x', f'{speedup_4:.2f}x', f'{speedup_8:.2f}x',
                               f'{eff_2:.2%}', f'{eff_4:.2%}', f'{eff_8:.2%}'])

print("✅ Created: thread_scaling.csv")

# ============================================================================
# 5. EXECUTIVE SUMMARY (one-pager)
# ============================================================================
with open(export_dir / 'executive_summary.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("BENCHMARK EXECUTIVE SUMMARY\n")
    f.write("=" * 80 + "\n\n")

    # Calculate overall stats
    all_speedups = []
    all_effs = []
    for res in results:
        for filt in results[res]:
            if '8' in results[res][filt] and 'speedup' in results[res][filt]['8']:
                all_speedups.append(results[res][filt]['8']['speedup'])
                all_effs.append(results[res][filt]['8'].get('efficiency', 0))

    avg_speedup = sum(all_speedups) / len(all_speedups) if all_speedups else 0
    avg_eff = sum(all_effs) / len(all_effs) if all_effs else 0
    best_speedup = max(all_speedups) if all_speedups else 0
    worst_speedup = min(all_speedups) if all_speedups else 0

    f.write(f"Test Configuration:\n")
    f.write(f"  - Filters tested: {len(results.get('480p', {}))}\n")
    f.write(f"  - Resolutions: {len(results)}\n")
    f.write(f"  - Thread counts: 1, 2, 4, 8\n")
    f.write(f"  - Total test combinations: {len(results) * len(results.get('480p', {})) * 4}\n\n")

    f.write(f"Overall Performance (8 threads):\n")
    f.write(f"  - Average speedup: {avg_speedup:.2f}x\n")
    f.write(f"  - Average efficiency: {avg_eff:.1%}\n")
    f.write(f"  - Best speedup: {best_speedup:.2f}x\n")
    f.write(f"  - Worst speedup: {worst_speedup:.2f}x\n\n")

    # Find best performers
    performers_sorted = sorted(performers, key=lambda x: x[2], reverse=True)
    f.write("Top 5 Performers:\n")
    for rank, (filt, res, speedup, eff, saved) in enumerate(performers_sorted[:5], 1):
        f.write(f"  {rank}. {filt} @ {res}: {speedup:.2f}x speedup, {eff:.1%} efficiency\n")

    f.write("\n" + "=" * 80 + "\n")

print("✅ Created: executive_summary.txt")

print("\n📁 All exports saved to: results_comprehensive/exports/")
print("\nGenerated files:")
print("  - summary_table.csv          (complete results table)")
print("  - best_performers.csv        (ranked by speedup)")
print("  - variance_analysis.csv      (stability analysis)")
print("  - thread_scaling.csv         (scaling progression)")
print("  - executive_summary.txt      (one-page overview)")
