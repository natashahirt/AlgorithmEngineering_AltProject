import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def main():
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "benchmark_results.csv"

    # 1. Load Data
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: '{csv_path}' not found. Run parse_logs.py first.")
        return

    # 2. Filter for Problem Size Scaling
    # Keep rows where filename DOES NOT contain "SCALING" AND solver is not "Unknown"
    domain_df = df[
        (~df['file'].str.contains("SCALING")) & 
        (df['solver'] != "Unknown")
    ].copy()
    
    if domain_df.empty:
        print("No domain scaling data found.")
        return

    # Sort by problem size
    domain_df.sort_values('elements', inplace=True)

    # Keep colors/styles consistent across all plots
    solver_order = ["Multigrid", "Jacobi Coarsest", "Jacobi Finest", "MATLAB"]
    solver_palette = {
        "Multigrid": "#800080",  # purple
        "Jacobi Coarsest": "#4fc3f7",   # light blue
        "Jacobi Finest": "#1565c0",     # dark blue
        "MATLAB": "#d81b60",            # magenta
    }
    # Solid lines for C++ variants, dotted for MATLAB
    solver_dashes = [(None, None), (None, None), (None, None), (2, 2)]

    # --- Plot 1: Total Time vs Elements (Log Scale) ---
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=domain_df,
        x='elements',
        y='total_time',
        hue='solver',
        style='solver',
        hue_order=solver_order,
        style_order=solver_order,
        palette=solver_palette,
        dashes=solver_dashes,
        markers=True,
        linewidth=2,
        markersize=8,
        estimator=None,
        sort=False,
    )
    plt.title('Total Runtime vs Problem Size', fontsize=14)
    plt.xlabel('Number of Elements', fontsize=12)
    plt.ylabel('Total Time (s)', fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(base_dir / 'plot_time_vs_size.png')
    print(f"Plot saved to '{base_dir / 'plot_time_vs_size.png'}'")
    
    # --- Plot 2: Total Time vs Elements (Linear Scale) ---
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=domain_df,
        x='elements',
        y='total_time',
        hue='solver',
        style='solver',
        hue_order=solver_order,
        style_order=solver_order,
        palette=solver_palette,
        dashes=solver_dashes,
        markers=True,
        linewidth=2,
        markersize=8,
        estimator=None,
        sort=False,
    )
    plt.title('Total Runtime vs Problem Size - Linear Scale', fontsize=14)
    plt.xlabel('Number of Elements', fontsize=12)
    plt.ylabel('Total Time (s)', fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig(base_dir / 'plot_time_vs_size_linear.png')
    print(f"Plot saved to '{base_dir / 'plot_time_vs_size_linear.png'}'")

    # --- Plot 3: Final Objective vs Elements ---
    # Filter out invalid/zero objectives (failed runs) for this plot to avoid skewing
    valid_obj_df = domain_df[
        (domain_df['final_obj'] > 0) & 
        (domain_df['final_obj'].notna())
    ].copy()

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=valid_obj_df,
        x='elements',
        y='final_obj',
        hue='solver',
        style='solver',
        hue_order=solver_order,
        style_order=solver_order,
        palette=solver_palette,
        dashes=solver_dashes,
        markers=True,
        linewidth=2,
        markersize=8,
        sort=False,
    )
    plt.title('Final Compliance (Objective) vs Problem Size', fontsize=14)
    plt.xlabel('Number of Elements', fontsize=12)
    plt.ylabel('Final Compliance', fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.xscale('log')
    plt.yscale('log') # Compliance usually scales with volume/size, log-log helps visualize
    plt.tight_layout()
    plt.savefig(base_dir / 'plot_obj_vs_size.png')
    print(f"Plot saved to '{base_dir / 'plot_obj_vs_size.png'}'")

    # --- Plot 4: Avg Time per Iteration vs Elements ---
    avg_time_df = domain_df.loc[domain_df['avg_time_per_iter'] > 0]
    if avg_time_df.empty:
        print("No average time-per-iteration data found.")
    else:
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=avg_time_df,
            x='elements',
            y='avg_time_per_iter',
            hue='solver',
            style='solver',
            hue_order=solver_order,
            style_order=solver_order,
            palette=solver_palette,
            dashes=solver_dashes,
            markers=True,
            linewidth=2,
            markersize=8,
            estimator=None,
            sort=False,
        )
        plt.title('Average Time per Iteration vs Problem Size', fontsize=14)
        plt.xlabel('Number of Elements', fontsize=12)
        plt.ylabel('Avg Time per Iteration (s)', fontsize=12)
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.xscale('log')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(base_dir / 'plot_avg_time_per_iter.png')
        print(f"Plot saved to '{base_dir / 'plot_avg_time_per_iter.png'}'")

    # --- Plot 5: Scaling Runs (time + speedup vs CPU count) ---
    scaling_df = df[
        (df['file'].str.contains("SCALING")) &
        (df['solver'] != "Unknown") &
        (df['cpus'] > 0)
    ].copy()

    if scaling_df.empty:
        print("No scaling data found.")
        return

    # For non-MATLAB runs, use reported threads if available; else assume 2x threads per CPU
    def core_count_row(row):
        if row['solver'] == "MATLAB":
            return row['cpus']
        if row['threads'] and row['threads'] > 0:
            return row['threads']
        return row['cpus'] * 2

    scaling_df['core_count'] = scaling_df.apply(core_count_row, axis=1)
    scaling_df.sort_values(by=['elements', 'solver', 'core_count'], inplace=True)

    # Compute per-solver baseline time at the minimum core count to derive speedup
    scaling_df['baseline_time'] = (
        scaling_df.groupby(['solver', 'elements'])['total_time']
                  .transform('first')
    )
    scaling_df['speedup'] = scaling_df['baseline_time'] / scaling_df['total_time']

    # Time vs CPUs (log-log to emphasize scaling slope)
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=scaling_df,
        x='core_count',
        y='total_time',
        hue='solver',
        style='solver',
        hue_order=solver_order,
        style_order=solver_order,
        palette=solver_palette,
        dashes=solver_dashes,
        markers=True,
        linewidth=2,
        markersize=8,
        estimator=None,
        sort=False,
    )
    plt.title('Total Runtime vs CPU Count', fontsize=14)
    plt.xlabel('CPU Cores', fontsize=12)
    plt.ylabel('Total Time (s)', fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(base_dir / 'plot_scaling_time.png')
    print(f"Plot saved to '{base_dir / 'plot_scaling_time.png'}'")

    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(
        data=scaling_df,
        x='core_count',
        y='speedup',
        hue='solver',
        style='solver',
        hue_order=solver_order,
        style_order=solver_order,
        palette=solver_palette,
        dashes=solver_dashes,
        markers=True,
        linewidth=2,
        markersize=8,
        estimator=None,
        sort=False,
    )
    plt.title('Speedup vs CPU Count', fontsize=14)
    plt.xlabel('CPU Cores', fontsize=12)
    plt.ylabel('Speedup (vs min-core run)', fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.legend()
    plt.savefig(base_dir / 'plot_scaling_speedup.png')
    print(f"Plot saved to '{base_dir / 'plot_scaling_speedup.png'}'")

if __name__ == "__main__":
    main()
