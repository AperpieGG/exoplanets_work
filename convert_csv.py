import pandas as pd

# --- File paths ---
input_file = "/Users/u5500483/Downloads/TIC453147896_timeseries_15Dec2025/TIC453147896_HARPS15_DRS-3-3-6.rdb"
output_file = "/Users/u5500483/Documents/GitHub/exoplanets_work/data/HARPS.csv"
# output_file = "/Users/u5500483/Downloads/HARPS.csv"

# --- Read the .rdb file (ignore commented lines) ---
df = pd.read_csv(input_file, delim_whitespace=True, comment="#")

# --- Keep only the first 3 columns ---
df = df.iloc[:, :3]

# --- Rename columns ---
df.columns = ["time", "flux", "flux_err"]

# --- Convert time to numeric (in case it's read as string) ---
df["time"] = pd.to_numeric(df["time"], errors="coerce")

# --- Drop any rows where time conversion failed ---
df = df.dropna(subset=["time"])

# --- Convert RJD → JD (add 2,400,000.0) ---
df["time"] = df["time"] + 2400000.0

# --- Convert time to numeric (in case it's read as string) ---
df["flux"] = pd.to_numeric(df["flux"], errors="coerce")

# --- Drop any rows where time conversion failed ---
df = df.dropna(subset=["flux"])

df["flux"] = df["flux"]
# df["flux"] = df["flux"] + 70.1

df["flux_err"] = pd.to_numeric(df["flux_err"], errors="coerce")

df["flux_err"] = df["flux_err"]

# --- Convert m/s → km/s ---
df["flux"]     = df["flux"]     / 1000.0
df["flux_err"] = df["flux_err"] / 1000.0

# --- Save as CSV ---
# --- Save CSV with column names as a comment ---
with open(output_file, "w") as f:
    f.write("# time,flux,flux_err\n")  # comment header
    df.to_csv(f, index=False, header=False, float_format="%.8f")

print(f"✅ Converted {input_file} → {output_file} with commented header")
print(df.head())
