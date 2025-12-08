import os
import re
import json
import csv
import glob
import pandas as pd

def parse_log_file(filepath):
    filename = os.path.basename(filepath)
    
    # Defaults
    data = {
        "file": filename,
        "solver": "Unknown",
        "nx": 0, "ny": 0, "nz": 0,
        "cpus": 0,
        "threads": 0,
        "total_time": 0.0,
        "final_obj": 0.0,
        "iterations": [],
        "num_iterations": 0,
        "error": None
    }

    # 1. Parse Metadata from Filename
    if filename.startswith("MG"):
        data["solver"] = "Multigrid"
    elif filename.startswith("JAC"):
        data["solver"] = "Jacobi"
    elif filename.startswith("MAT"):
        data["solver"] = "MATLAB"
        
    dim_match = re.search(r"(\d+)x(\d+)x(\d+)", filename)
    if dim_match:
        data["ny"] = int(dim_match.group(1))
        data["nx"] = int(dim_match.group(2))
        data["nz"] = int(dim_match.group(3))
    
    cpu_match = re.search(r"_(\d+)cpu", filename)
    if cpu_match:
        data["cpus"] = int(cpu_match.group(1))
    else:
        data["cpus"] = 16 

    try:
        with open(filepath, 'r') as f:
            content = f.read()
            lines = content.split('\n')

        # 2. Parse Content
        if data["solver"] == "MATLAB":
            worker_match = re.search(r"Connected to parallel pool with (\d+) workers", content)
            if worker_match:
                data["cpus"] = int(worker_match.group(1))
                data["threads"] = int(worker_match.group(1))
        else:
            thread_match = re.search(r"THREADS\s+(\d+)", content)
            if thread_match:
                data["threads"] = int(thread_match.group(1))
                if not cpu_match:
                     data["cpus"] = data["threads"] // 2

        # Iteration Data
        iter_objs = {}
        iter_times = {}

        for line in lines:
            # Updated Obj Parse Regex: more permissive spacing
            # Look for "It.:", number, "Obj.:", number, "Vol.:", number
            obj_match = re.search(r"It\.:\s*(\d+)\s+Obj\.:\s*([\d\.eE\+\-]+)\s+Vol\.:\s*([\d\.eE\+\-]+)", line)
            if obj_match:
                it = int(obj_match.group(1))
                iter_objs[it] = {
                    "obj": float(obj_match.group(2)),
                    "vol": float(obj_match.group(3))
                }
            
            # Time Parse Regex
            time_match = re.search(r"It\.:\s*(\d+)\s+\(Time\)\.\.\.\s+Total per-It\.:\s*([\d\.eE\+\-]+)s", line)
            if time_match:
                it = int(time_match.group(1))
                iter_times[it] = float(time_match.group(2))

        # Combine Iteration Data
        all_its = sorted(list(set(iter_objs.keys()) | set(iter_times.keys())))
        for it in all_its:
            it_data = {
                "iteration": it,
                "objective": iter_objs.get(it, {}).get("obj", None),
                "volume": iter_objs.get(it, {}).get("vol", None),
                "time": iter_times.get(it, 0.0)
            }
            data["iterations"].append(it_data)

        if data["iterations"]:
            data["num_iterations"] = len(data["iterations"])
            last_it = data["iterations"][-1]
            if last_it["objective"] is not None:
                data["final_obj"] = last_it["objective"]
            
            total_time_match = re.search(r"total solver time:\s+([\d\.eE\+\-]+)", content)
            if total_time_match:
                data["total_time"] = float(total_time_match.group(1))
            else:
                data["total_time"] = sum(it["time"] for it in data["iterations"])

    except Exception as e:
        data["error"] = str(e)
        print(f"Error parsing {filename}: {e}")

    return data

def main():
    # Only parsing test/log per your change
    log_dirs = ["test_master/log"]
    all_results = []
    
    print("Parsing logs...")
    for d in log_dirs:
        if not os.path.exists(d):
            print(f"Directory not found: {d}")
            continue
            
        files = glob.glob(os.path.join(d, "*.out"))
        for f in files:
            result = parse_log_file(f)
            if result["num_iterations"] > 0:
                all_results.append(result)
    
    with open("benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    flat_data = []
    for r in all_results:
        times = [it["time"] for it in r["iterations"]]
        objs = [it["objective"] for it in r["iterations"]]
        
        row = {
            "file": r["file"],
            "solver": r["solver"],
            "nx": r["nx"], "ny": r["ny"], "nz": r["nz"],
            "elements": r["nx"] * r["ny"] * r["nz"],
            "cpus": r["cpus"],
            "threads": r["threads"],
            "iterations": r["num_iterations"],
            "total_time": r["total_time"],
            "final_obj": r["final_obj"],
            "avg_time_per_iter": r["total_time"] / r["num_iterations"] if r["num_iterations"] > 0 else 0,
            "time_history": str(times),
            "objective_history": str(objs)
        }
        flat_data.append(row)
        
    if not flat_data:
        print("No valid data found to save to CSV.")
        return

    df = pd.DataFrame(flat_data)
    df.sort_values(by=["solver", "elements", "cpus"], inplace=True)
    df.to_csv("benchmark_results.csv", index=False)
    
    print(f"Successfully parsed {len(all_results)} logs.")
    print("Saved to 'benchmark_results.json' and 'benchmark_results.csv'")

if __name__ == "__main__":
    main()
