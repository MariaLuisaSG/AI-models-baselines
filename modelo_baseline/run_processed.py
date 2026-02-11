from src.data.preprocess import preprocess


def main():

    # breast cancer
    preprocess(
        input_path="data/raw/breast_cancer.csv",
        output_path="data/processed/breast_cancer_clean.csv",
        target_column="recurrence"
    )

    # healthcare
    preprocess(
        input_path="data/raw/Healthcare.csv",
        output_path="data/processed/healthcare_clean.csv",
        target_column="Disease"
    )


if __name__ == "__main__":
    main()
