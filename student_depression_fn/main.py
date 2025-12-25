from data_cleaning.data_cleaner import run_full_cleaning
from data_normalization.data_normalizer import normalize_dataset
from data_visualization.visualization import run_all_analysis
from storage.save_data import save_data
def main():
    file_path = "data/student_depression_dataset.csv"
    cleaned_data = run_full_cleaning(file_path)
    normalized_cleaned_data = normalize_dataset(cleaned_data)
    run_all_analysis(normalized_cleaned_data)
    save_data(normalized_cleaned_data, "data/cleaned_student_depression_dataset.csv")

if __name__ == "__main__":
    main()
