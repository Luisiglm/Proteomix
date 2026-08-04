import pandas as pd
import os
import csv

REQUIRED_COLUMNS = [
    "Protein",
    "Protein ID",
    "Entry Name",
    "Gene",
    "Protein Length",
    "Organism",
    "Protein Existence",
    "Description",
    "Protein Probability",
    "Top Peptide Probability"
]

NUMERIC_COLUMNS = [
    "Protein Length",
    "Protein Probability",
    "Top Peptide Probability"
]

def get_csv_rows(indexes):
    """Convert dataframe index to CSV row numbers."""
    return (indexes + 2).tolist()

def validate_csv(path):

    errors = []


    # Check file exists
    if not os.path.exists(path):
        return False, ["File does not exist"]

    # Read CSV
    try:
        df = pd.read_csv(path)

    except Exception as e:
        return False, [str(e)]

    # Empty CSV
    if df.empty:
        errors.append("CSV contains no data")

    # Duplicate headers
    with open(path, newline='') as f:
        columns = next(csv.reader(f))

    # Read the rest of the data without letting pandas touch the header
    df_dup = pd.read_csv(path, header=None, skiprows=1)
    df_dup.columns = columns
    duplicates = df_dup.columns[df_dup.columns.duplicated()].tolist()

    if duplicates:
        errors.append(f"Duplicate column names: {duplicates}")

    # Required columns
    missing = []

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            missing.append(col)

    if missing:
        errors.append(f"Missing required columns: {missing}")

    # Empty rows
    empty_rows = df[df.isnull().all(axis=1)]

    if len(empty_rows) > 0:
        errors.append(f"{len(empty_rows)} completely empty rows detected")


    # Numeric columns
    for col in NUMERIC_COLUMNS:

        if col in df.columns:

            invalid = pd.to_numeric(df[col], errors="coerce").isna()

            if invalid.any():

                rows = get_csv_rows(invalid[invalid].index)

                errors.append(
                    f"{col} contains non-numeric values at CSV rows: {rows}"
                )
    # Protein Length must be positive
    if "Protein Length" in df.columns:

        values = pd.to_numeric(df["Protein Length"], errors="coerce")

        invalid = values <= 0

        if invalid.any():
            rows = get_csv_rows(invalid[invalid].index)

            errors.append(
                f"Protein Length contains values <= 0 at CSV rows: {rows}"
            )

    # Probability columns between 0 and 1
    probability_columns = [
        "Protein Probability",
        "Top Peptide Probability"
    ]

    for col in probability_columns:

        if col in df.columns:

            values = pd.to_numeric(df[col], errors="coerce")

            invalid = (values < 0) | (values > 1)

            if invalid.any():
                rows = get_csv_rows(invalid[invalid].index)

                errors.append(
                    f"{col} contains values outside range 0-1 at CSV rows: {rows}"
                )

    # Duplicate Protein IDs
    if "Protein ID" in df.columns:

        duplicates = df["Protein ID"].duplicated()

        if duplicates.any():
            rows = get_csv_rows(duplicates[duplicates].index)

            errors.append(
                f"Duplicate Protein IDs found at CSV rows: {rows}"
            )

    # Duplicate rows
    duplicate_rows = df.duplicated()

    if duplicate_rows.any():
        rows = get_csv_rows(duplicate_rows[duplicate_rows].index)

        errors.append(
            f"Duplicate rows found at CSV rows: {rows}"
        )

    # Leading/trailing spaces
    text_columns = [
        "Protein",
        "Protein ID",
        "Entry Name",
        "Gene",
        "Organism"
    ]

    for col in text_columns:

        if col in df.columns:

            spaces = df[col].astype(str).str.startswith(" ") | \
                     df[col].astype(str).str.endswith(" ")

            if spaces.any():
                rows = get_csv_rows(spaces[spaces].index)

                errors.append(
                    f"{col} contains leading/trailing spaces at CSV rows: {rows}"
                )

    # Find where measurement columns begin
    try:
        start = df.columns.get_loc("Combined Unique Spectral Count") + 1
    except KeyError:
        errors.append("'Combined Unique Spectral Count' column not found.")
        return False, errors

    measurement_columns = df.columns[start:]

    for column in measurement_columns:

        # Blank values
        missing = df[column].isnull()

        if missing.any():
            rows = (missing[missing].index + 2).tolist()
            errors.append(
                f"{column} contains blank values at rows {rows}"
            )

        # Numeric values
        invalid = pd.to_numeric(df[column], errors="coerce").isna()

        # Ignore blanks already reported
        invalid = invalid & ~missing

        if invalid.any():
            rows = (invalid[invalid].index + 2).tolist()
            errors.append(
                f"{column} contains non-numeric values at rows {rows}"
            )
    # Final result
    if errors:
        return False, errors

    return True, df