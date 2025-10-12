import os
import numpy as np
import pandas as pd
import scipy.io

# =============================
# Statistical Feature Functions
# =============================
STAT_FUNCS = {
    "avg": np.mean,
    "min": np.min,
    "max": np.max,
    # Reserved for future extensions:
    # "median": np.median,
    # "std": np.std,
}

def extract_stats(data, prefix):
    """Compute multiple statistical features for an array"""
    features = {}
    for stat_name, func in STAT_FUNCS.items():
        try:
            features[f"{prefix}_{stat_name}"] = func(data)
        except Exception:
            features[f"{prefix}_{stat_name}"] = np.nan
    return features


# =============================
# Function: Process a single .mat file
# =============================
def process_battery(mat_path):
    """
    Process one .mat file:
    - Automatically detect charge/discharge cycles
    - Pair them in sequence (the i-th charge matches the i-th discharge)
    - Compute avg/min/max features
    """
    mat = scipy.io.loadmat(mat_path)
    battery_name = os.path.basename(mat_path).split(".")[0]

    # Automatically detect main key
    main_key = next((k for k in mat.keys() if not k.startswith("__")), None)
    if main_key is None or "cycle" not in mat[main_key][0, 0].dtype.names:
        print(f"{battery_name}: No valid 'cycle' structure found, skipped.")
        return pd.DataFrame()

    cycles = mat[main_key][0, 0]["cycle"][0]
    rows_charge, rows_discharge = [], []

    for i, cycle in enumerate(cycles, start=1):
        ctype = cycle["type"][0]
        if ctype not in ["charge", "discharge"]:
            continue

        data = cycle["data"][0, 0]
        voltage = data["Voltage_measured"][0]
        current = data["Current_measured"][0]
        temp = data["Temperature_measured"][0]

        features = {}
        prefix = f"{ctype}"
        features.update(extract_stats(voltage, f"{prefix}_voltage"))
        features.update(extract_stats(current, f"{prefix}_current"))
        features.update(extract_stats(temp, f"{prefix}_temp"))

        # Extra labels only for discharge
        if ctype == "discharge":
            if "Time" in data.dtype.names and len(data["Time"][0]) > 0:
                features["discharge_duration_s"] = data["Time"][0][-1]
            else:
                features["discharge_duration_s"] = np.nan

            if "Capacity" in data.dtype.names and len(data["Capacity"][0]) > 0:
                features["capacity_Ah"] = data["Capacity"][0][-1]
            else:
                features["capacity_Ah"] = np.nan

            rows_discharge.append(features)
        else:
            rows_charge.append(features)

    # ===== Pair charge and discharge =====
    n_pairs = min(len(rows_charge), len(rows_discharge))
    if n_pairs == 0:
        print(f"{battery_name}: No charge-discharge pairs found.")
        return pd.DataFrame()

    merged_rows = []
    for idx in range(n_pairs):
        merged_rows.append({
            "battery": battery_name,
            "cycle": idx + 1,
            **rows_charge[idx],
            **rows_discharge[idx]
        })

    df = pd.DataFrame(merged_rows)
    print(f"{battery_name}: Successfully matched {n_pairs} charge+discharge pairs.")
    return df


# =============================
# Main function: Process all files in batch
# =============================
def main():
    base_dir = r"C:\Users\lijun\Downloads\ESE5971-Project\5_Battery_Data_Set"
    output_dir = r"C:\Users\lijun\Downloads\ESE5971-Project\Data"
    os.makedirs(output_dir, exist_ok=True)

    all_dfs = []

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".mat"):
                path = os.path.join(root, file)
                print(f"Processing {path} ...")
                df = process_battery(path)
                if not df.empty:
                    all_dfs.append(df)

    if not all_dfs:
        print(" No DataFrames generated. Please check data structure or pairing logic.")
        return

    final_df = pd.concat(all_dfs, ignore_index=True)

    # Add Remaining Useful Life (RUL)
    final_df["RUL"] = final_df.groupby("battery")["cycle"].transform(lambda x: x.max() - x)

    out_path = os.path.join(output_dir, "battery_cycles_summary.csv")
    final_df.to_csv(out_path, index=False)
    print(f"\n All processing complete! Output file: {out_path}")
    print(f" Total rows: {len(final_df)}, Total columns: {len(final_df.columns)}\n")
    print("Preview:")
    print(final_df.head())


# =============================
# Entry point
# =============================
if __name__ == "__main__":
    main()
