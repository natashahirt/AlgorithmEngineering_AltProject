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

    # --- Plot 1: Total Time vs Elements (Log Scale) ---
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=domain_df, x='elements', y='total_time', hue='solver', style='solver', 
                 markers=True, dashes=False, linewidth=2, markersize=8)
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
    sns.lineplot(data=domain_df, x='elements', y='total_time', hue='solver', style='solver', 
                 markers=True, dashes=False, linewidth=2, markersize=8)
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
    sns.lineplot(data=valid_obj_df, x='elements', y='final_obj', hue='solver', style='solver', 
                 markers=True, dashes=False, linewidth=2, markersize=8)
    plt.title('Final Compliance (Objective) vs Problem Size', fontsize=14)
    plt.xlabel('Number of Elements', fontsize=12)
    plt.ylabel('Final Compliance', fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.xscale('log')
    plt.yscale('log') # Compliance usually scales with volume/size, log-log helps visualize
    plt.tight_layout()
    plt.savefig('plot_obj_vs_size.png')
    print("Plot saved to 'plot_obj_vs_size.png'")

if __name__ == "__main__":
    main()
