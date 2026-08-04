import unittest
import pandas as pd
import os

from validator import validate_csv


class TestCSV(unittest.TestCase):

    # Checks that a correctly formatted CSV passes validation
    def test_valid_csv(self):
        valid, message = validate_csv("../PD1_TCR_IP_fragpipe.csv")

        self.assertTrue(valid, message)


    # Checks that the validator handles missing files correctly
    def test_missing_file(self):
        valid, message = validate_csv("missing.csv")

        self.assertFalse(valid)


    # Checks that an empty CSV file is rejected
    def test_empty_csv(self):

        df = pd.DataFrame()

        df.to_csv("empty.csv", index=False)

        valid, message = validate_csv("empty.csv")

        self.assertFalse(valid)

        os.remove("empty.csv")


    # Checks that incorrect CSV structures are rejected
    def test_wrong_column_count(self):

        df = pd.DataFrame({
            "Protein": ["P001"],
            "Wrong Column": [123]
        })

        df.to_csv("wrong_column_count.csv", index=False)

        valid, message = validate_csv("wrong_column_count.csv")

        self.assertFalse(valid)

        os.remove("wrong_column_count.csv")


    # Checks that missing mandatory metadata columns are detected
    def test_missing_required_column(self):

        df = pd.DataFrame({
            "Protein": ["P001"],
            "Probability": [0.95]
        })

        df.to_csv("missing_required.csv", index=False)

        valid, message = validate_csv("missing_required.csv")

        self.assertFalse(valid)

        os.remove("missing_required.csv")


    # Checks that duplicate column headers are detected
    def test_duplicate_headers(self):

        with open("duplicate_headers.csv", "w") as file:
            file.write(
                "Protein,Probability,Protein\n"
                "P001,0.95,P001\n"
            )

        valid, message = validate_csv("duplicate_headers.csv")

        self.assertFalse(valid)

        os.remove("duplicate_headers.csv")


    # Checks that completely empty rows are detected
    def test_empty_rows(self):

        df = pd.DataFrame({
            "Protein": ["P001", None],
            "Probability": [0.95, None],
            "Protein Length": [300, None]
        })

        df.to_csv("empty_rows.csv", index=False)

        valid, message = validate_csv("empty_rows.csv")

        self.assertFalse(valid)

        os.remove("empty_rows.csv")


    # Checks that missing values in required metadata fields are detected
    def test_missing_metadata_values(self):

        df = pd.DataFrame({
            "Protein": [None],
            "Probability": [0.95],
            "Protein Length": [300]
        })

        df.to_csv("missing_metadata.csv", index=False)

        valid, message = validate_csv("missing_metadata.csv")

        self.assertFalse(valid)

        os.remove("missing_metadata.csv")


    # Checks that numeric fields reject text values
    def test_non_numeric_protein_length(self):

        df = pd.DataFrame({
            "Protein": ["P001"],
            "Protein Length": ["not_number"],
            "Probability": [0.95]
        })

        df.to_csv("non_numeric_length.csv", index=False)

        valid, message = validate_csv("non_numeric_length.csv")

        self.assertFalse(valid)

        os.remove("non_numeric_length.csv")


    # Checks probability columns only contain numeric values
    def test_non_numeric_probability(self):

        df = pd.DataFrame({
            "Protein": ["P001"],
            "Protein Length": [300],
            "Probability": ["not_number"]
        })

        df.to_csv("non_numeric_probability.csv", index=False)

        valid, message = validate_csv("non_numeric_probability.csv")

        self.assertFalse(valid)

        os.remove("non_numeric_probability.csv")


    # Checks measurement columns reject non-numeric values
    def test_non_numeric_measurement_column(self):

        df = pd.DataFrame({
            "Protein": ["P001"],
            "Protein Length": [300],
            "Probability": [0.95],
            "Sample_1": ["not_number"]
        })

        df.to_csv("non_numeric_measurement.csv", index=False)

        valid, message = validate_csv("non_numeric_measurement.csv")

        self.assertFalse(valid)

        os.remove("non_numeric_measurement.csv")


    # Checks that every measurement column after
    # "Combined Unique Spectral Count" contains only numbers
    def test_all_measurement_columns_numeric(self):

        df = pd.DataFrame({
            "Combined Unique Spectral Count": [5, 8],
            "Sample_1": [10.5, 20.1],
            "Sample_2": [15, "ABC"],  # Invalid non-numeric value
            "Sample_3": [1.2, 3.4],
            "Sample_4": [100, 200]
        })

        df.to_csv("measurement_numeric.csv", index=False)

        valid, errors = validate_csv("measurement_numeric.csv")

        self.assertFalse(valid)

        self.assertIn(
            "Sample_2 contains non-numeric values at rows [3]",
            errors
        )

        os.remove("measurement_numeric.csv")


if __name__ == "__main__":
    unittest.main()
