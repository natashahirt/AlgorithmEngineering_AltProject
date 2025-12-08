import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # 1. Load Data
    try:
        df = pd.read_csv("benchmark_results.csv")
    except FileNotFoundError:
        print("Error: 'benchmark_results.csv' not found. Run parse_logs.py first.")
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
    solver_order = ["Multigrid", "Jacobi", "MATLAB"]
    solver_palette = {
        "Multigrid": "#1f77b4",  # blue
        "Jacobi": "#2ca02c",     # green
        "MATLAB": "#d62728",     # red
    }
    # Solid lines for Multigrid/Jacobi, dotted for MATLAB
    solver_dashes = [(None, None), (None, None), (2, 2)]

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
    plt.savefig('plot_time_vs_size.png')
    print("Plot saved to 'plot_time_vs_size.png'")
    
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
    plt.savefig('plot_time_vs_size_linear.png')
    print("Plot saved to 'plot_time_vs_size_linear.png'")

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
    plt.savefig('plot_obj_vs_size.png')
    print("Plot saved to 'plot_obj_vs_size.png'")

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
        plt.savefig('plot_avg_time_per_iter.png')
        print("Plot saved to 'plot_avg_time_per_iter.png'")

if __name__ == "__main__":
    main()
