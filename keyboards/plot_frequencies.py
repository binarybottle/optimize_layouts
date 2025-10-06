#!/usr/bin/env python3
"""Plot CSV data for letter‑bigram statistics."""

import pathlib
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1️⃣  CONFIGURATION
# ----------------------------------------------------------------------
CSV_PATH = pathlib.Path("../output/analyze_input/cumulative_bigram_analysis.csv")
OUTPUT_DIR = pathlib.Path("plots")
OUTPUT_DIR.mkdir(exist_ok=True)

# Columns
NUMERIC_COLS = [
    "letter_frequency",
    "letter_frequency_percentage",
    "new_bigrams_added",
    "new_bigram_frequency",
    "new_bigram_frequency_percentage",
    "total_bigrams_so_far",
    "cumulative_bigram_frequency",
    "cumulative_percentage",
    "letter_bigram_frequency",
    "letter_bigram_percentage",
]

# ----------------------------------------------------------------------
# 2️⃣  LOAD THE DATA
# ----------------------------------------------------------------------
def load_data(csv_path: pathlib.Path) -> pd.DataFrame:
    """Read the CSV, force numeric conversion, and drop rows that are completely empty."""
    df = pd.read_csv(csv_path, dtype=str)          # read everything as string first
    # Strip whitespace from column names (just in case)
    df.columns = df.columns.str.strip()

    # Convert the known numeric columns to float (handles scientific notation)
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where the primary key (`letter`) is missing
    df = df.dropna(subset=["letter"])
    return df


df = load_data(CSV_PATH)

# ----------------------------------------------------------------------
# 3️⃣  PLOTTING HELPERS
# ----------------------------------------------------------------------
def save_plot(fig, name: str):
    """Utility to store a figure as PNG."""
    out_file = OUTPUT_DIR / f"{name}.png"
    fig.savefig(out_file, bbox_inches="tight", dpi=300)
    print(f"✅ Saved {out_file}")

def bar_plot(x, y, xlabel, ylabel, title, rotate_xticks=False):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, y, color="#4A90E2")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if rotate_xticks:
        plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig

def line_plot(x, y, xlabel, ylabel, title):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, marker="o", linestyle="-", color="#D0021B")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()
    return fig

# ----------------------------------------------------------------------
# 4️⃣  PLOTS
# ----------------------------------------------------------------------
# 4a️⃣ Letter frequency (absolute count)
if {"letter", "letter_frequency"}.issubset(df.columns):
    fig = bar_plot(
        x=df["letter"],
        y=df["letter_frequency"],
        xlabel="Letter",
        ylabel="Frequency (raw count)",
        title="Letter Frequency (Absolute)",
    )
    save_plot(fig, "letter_frequency_absolute")

# 4b️⃣ Letter frequency percentage
if {"letter", "letter_frequency_percentage"}.issubset(df.columns):
    fig = bar_plot(
        x=df["letter"],
        y=df["letter_frequency_percentage"],
        xlabel="Letter",
        ylabel="Frequency (%)",
        title="Letter Frequency (Percentage)",
    )
    save_plot(fig, "letter_frequency_percentage")

# 4c️⃣ New bigrams added per letter (line chart)
if {"letter", "new_bigrams_added"}.issubset(df.columns):
    fig = line_plot(
        x=df["letter"],
        y=df["new_bigrams_added"],
        xlabel="Letter",
        ylabel="New Bigrams Added",
        title="New Bigrams Introduced by Each Letter",
    )
    save_plot(fig, "new_bigrams_added")

# 4d️⃣ Cumulative bigram frequency (line chart)
if {"letter", "cumulative_bigram_frequency"}.issubset(df.columns):
    fig = line_plot(
        x=df["letter"],
        y=df["cumulative_bigram_frequency"],
        xlabel="Letter",
        ylabel="Cumulative Bigram Frequency",
        title="Cumulative Bigram Frequency Across Letters",
    )
    save_plot(fig, "cumulative_bigram_frequency")

# 4e️⃣ Letter‑bigram frequency (percentage) – heat‑style bar chart
if {"letter", "letter_bigram_percentage"}.issubset(df.columns):
    fig = bar_plot(
        x=df["letter"],
        y=df["letter_bigram_percentage"],
        xlabel="Letter",
        ylabel="Bigram % (of all bigrams)",
        title="Letter‑Bigram Frequency Percentage",
        rotate_xticks=True,
    )
    save_plot(fig, "letter_bigram_percentage")
