import pandas as pd
import numpy as np

def make_spectro_table(rdb_file, output_tex="spec_table.tex"):
    # Read RDB file (columns separated by whitespace)
    df = pd.read_csv(rdb_file, sep=r"\s+", engine="python")

    # Convert all numeric columns safely
    for col in ["rjd", "vrad", "svrad", "fwhm", "bis_span", "contrast", "ha", "ca", "na"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Format columns
    df["Time"]      = df["rjd"].map(lambda x: f"{x:.8f}")
    df["RV"]        = df["vrad"].map(lambda x: f"{x:.2f}")
    df["RVerr"]     = df["svrad"].map(lambda x: f"{x:.2f}")
    df["FWHM"]      = df["fwhm"].map(lambda x: f"{x:.2f}")
    df["Bisector"]  = df["bis_span"].map(lambda x: f"{x:.2f}" if pd.notnull(x) else "--")
    df["Contrast"]  = df["contrast"].map(lambda x: f"{x:.2f}")
    df["Halpha"]    = df["ha"].map(lambda x: f"{x:.6f}" if pd.notnull(x) else "--")
    df["CaHK"]      = df["ca"].map(lambda x: f"{x:.6f}" if pd.notnull(x) else "--")
    df["NaD"]       = df["na"].map(lambda x: f"{x:.6f}" if pd.notnull(x) else "--")

    # Build table rows
    rows = []
    for i in range(len(df)):
        row = (
            f"\\coralie\\,-14 & "
            f"{df['Time'][i]} & "
            f"{df['RV'][i]} & "
            f"{df['RVerr'][i]} & "
            f"{df['FWHM'][i]} & "
            f"{df['Bisector'][i]} & "
            f"{df['Contrast'][i]} & "
            f"{df['Halpha'][i]} & "
            f"{df['CaHK'][i]} & "
            f"{df['NaD'][i]} \\\\"
        )
        rows.append(row)

    # Combine
    table_body = "\n".join(rows)

    latex = f"""
\\begin{{center}}
\\begin{{table*}}
    \\centering
    \\caption{{Spectroscopic data for \\NTICID$\\slash$\\Nstar\\ from \\coralie\\,-14.}}
    \\label{{tab:specdata}}
    \\begin{{tabular}}{{l c c c c c c c c c}}
    \\hline
    \\hline
    Instrument & Time (BJD & RV & RV error & FWHM & Bisector & Contrast & H-$\\alpha$ & Ca\\,II H K & Na D \\\\
    & -2457000) & ($\\rm m\\,s^{{-1}}$) & ($\\rm m\\,s^{{-1}}$) & ($\\rm m\\,s^{{-1}}$) & ($\\rm m\\,s^{{-1}}$) & & & & \\\\
    \\hline
{table_body}
    \\hline
    \\end{{tabular}}
\\end{{table*}}
\\end{{center}}
"""

    with open(output_tex, "w") as f:
        f.write(latex)

    print(f"Saved LaTeX table to {output_tex}")

# Example usage:
PATH = '/Users/u5500483/Downloads/'
make_spectro_table(PATH + "TIC453147896_HARPS15_DRS-3-3-6.rdb")