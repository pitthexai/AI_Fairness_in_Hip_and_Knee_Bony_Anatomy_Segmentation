import argparse

import numpy as np
import pandas as pd

import os

from sklearn.model_selection import train_test_split

def setup_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("anatomy", help="Corresponding bony anatomy for demographic data")
    parser.add_argument("csv_file", help="CSV file containing demographic information for given bony anatomy")
    parser.add_argument("outdir", help="Directory to save training, validation, and testing splits for experimentations")
    return parser

def generate_age_grouping(dataset):
    dataset_w_age = dataset.copy()

    # Define groups
    # 0 - <= 50
    # 1 - 51 -64
    # 2 - 65-79

    dataset_w_age["V00AGE_GROUP"] = 0
    dataset_w_age["V00AGE_GROUP"] = np.where(dataset.V00AGE <= 50, 0, dataset_w_age["V00AGE_GROUP"])
    dataset_w_age["V00AGE_GROUP"] = np.where((dataset.V00AGE > 50) & (dataset.V00AGE <= 64), 1, dataset_w_age["V00AGE_GROUP"])
    dataset_w_age["V00AGE_GROUP"] = np.where((dataset.V00AGE > 64) & (dataset.V00AGE <= 79), 2, dataset_w_age["V00AGE_GROUP"])
    return dataset_w_age

def generate_train_test_split(data_records, filter_query=None):
    data_records = data_records[data_records.id != 9025994].reset_index(drop=True)
    if filter_query:
        data_records = data_records.query(filter_query)

    train, test = train_test_split(data_records.id.unique(), test_size=0.3, random_state=42)
    valid, test = train_test_split(test, test_size=0.5, random_state=42)

    train = data_records[data_records.id.isin(train)].reset_index(drop=True)
    valid = data_records[data_records.id.isin(valid)].reset_index(drop=True)
    test = data_records[data_records.id.isin(test)].reset_index(drop=True)

    return train, valid, test

def balance_dataset(data, filters):
    filtered = {
        filt: data.query(filt) for filt in filters
    }

    min_sample_size = np.min([len(filt) for filt in filtered.values()])

    balanced_data = pd.DataFrame()
    for filt, values in filtered.items():
        balanced_data = pd.concat([balanced_data, values.sample(min_sample_size, random_state=42)])


    balanced_data = balanced_data.reset_index(drop=True)

    print(f"Training dataset reduced from size of {len(data)} samples to a balanced dataset of size {len(balanced_data)} samples")

    return balanced_data

if __name__ == "__main__":
    parser = setup_argparse()
    args = parser.parse_args()
    anatomy = args.anatomy
    csv_filename = args.csv_file
    outdir = args.outdir

    data_records = pd.read_csv(csv_filename)
    data_records = data_records[["id", "P02SEX", "P02RACE", "V00AGE"]]
    data_records_w_age_group = generate_age_grouping(data_records)

    data_records_w_age_group.replace({"1: White or Caucasian": "White_Caucasian",
             "2: Black or African American": "Black_AfricanAmerican",
             "1: Male": "Male",
             "2: Female": "Female"}, inplace=True)
    
    data_records_w_age_group["V00AGE_GROUP"].replace({
        0: "Age_50_Lower",
        1: "Age_51_64",
        2: "Age_65_79"
    }, inplace=True)

    data_records_w_age_group.to_csv(f"/data_vault/hexai/ScientificReports_Hip_Knee_Datasets/{anatomy}/{anatomy}_segmentation.csv", index=False)

    train_all, valid_all, test_all = generate_train_test_split(data_records_w_age_group)
    train_white, valid_white, test_white = generate_train_test_split(data_records_w_age_group, filter_query="P02RACE == '1: White or Caucasian'")
    train_black, valid_black, test_black = generate_train_test_split(data_records_w_age_group, filter_query="P02RACE == '2: Black or African American'")
    train_male, valid_male, test_male = generate_train_test_split(data_records_w_age_group, filter_query="P02SEX == '1: Male'")
    train_female, valid_female, test_female = generate_train_test_split(data_records_w_age_group, filter_query="P02SEX == '2: Female'")

    train_age_50_lower, valid_age_50_lower, test_age_50_lower = generate_train_test_split(data_records_w_age_group, filter_query="V00AGE_GROUP == 0")
    train_age_51_64, valid_age_51_64, test_age_51_64 = generate_train_test_split(data_records_w_age_group, filter_query="V00AGE_GROUP == 1")
    train_age_65_79, valid_age_65_79, test_age_65_79 = generate_train_test_split(data_records_w_age_group, filter_query="V00AGE_GROUP == 2")

    balanced_gender_train = balance_dataset(train_all, ["P02SEX == '1: Male'", "P02SEX == '2: Female'"])
    balanced_race_train = balance_dataset(train_all, ["P02RACE == '1: White or Caucasian'", "P02RACE == '2: Black or African American'"])
    balanced_age_train = balance_dataset(train_all, ["V00AGE_GROUP == 0", "V00AGE_GROUP == 1", "V00AGE_GROUP == 2"])

    outdir = f"{outdir}/{anatomy}"
    os.makedirs(outdir, exist_ok=True)

    # Save Baseline splits
    train_all.to_csv(f"{outdir}/{anatomy}_train_all.csv", index=False)
    valid_all.to_csv(f"{outdir}/{anatomy}_valid_all.csv", index=False)
    test_all.to_csv(f"{outdir}/{anatomy}_test_all.csv", index=False)

    # Save Race Splits
    train_white.to_csv(f"{outdir}/{anatomy}_train_white.csv", index=False)
    valid_white.to_csv(f"{outdir}/{anatomy}_valid_white.csv", index=False)
    test_white.to_csv(f"{outdir}/{anatomy}_test_white.csv", index=False)

    train_black.to_csv(f"{outdir}/{anatomy}_train_black.csv", index=False)
    valid_black.to_csv(f"{outdir}/{anatomy}_valid_black.csv", index=False)
    test_black.to_csv(f"{outdir}/{anatomy}_test_black.csv", index=False)

    # Save Sex Splits
    train_male.to_csv(f"{outdir}/{anatomy}_train_male.csv", index=False)
    valid_male.to_csv(f"{outdir}/{anatomy}_valid_male.csv", index=False)
    test_male.to_csv(f"{outdir}/{anatomy}_test_male.csv", index=False)

    train_female.to_csv(f"{outdir}/{anatomy}_train_female.csv", index=False)
    valid_female.to_csv(f"{outdir}/{anatomy}_valid_female.csv", index=False)
    test_female.to_csv(f"{outdir}/{anatomy}_test_female.csv", index=False)

    # Save Age Splits
    train_age_50_lower.to_csv(f"{outdir}/{anatomy}_train_age_50_lower.csv", index=False)
    valid_age_50_lower.to_csv(f"{outdir}/{anatomy}_valid_age_50_lower.csv", index=False)
    test_age_50_lower.to_csv(f"{outdir}/{anatomy}_test_age_50_lower.csv", index=False)

    train_age_51_64.to_csv(f"{outdir}/{anatomy}_train_age_51_64.csv", index=False)
    valid_age_51_64.to_csv(f"{outdir}/{anatomy}_valid_age_51_64.csv", index=False)
    test_age_51_64.to_csv(f"{outdir}/{anatomy}_test_age_51_64.csv", index=False)

    train_age_65_79.to_csv(f"{outdir}/{anatomy}_train_age_65_79.csv", index=False)
    valid_age_65_79.to_csv(f"{outdir}/{anatomy}_valid_age_65_79.csv", index=False)
    test_age_65_79.to_csv(f"{outdir}/{anatomy}_test_age_65_79.csv", index=False)


    # Save Balanced Splits
    balanced_race_train.to_csv(f"{outdir}/{anatomy}_race_balanced.csv", index=False)
    balanced_gender_train.to_csv(f"{outdir}/{anatomy}_gender_balanced.csv", index=False)
    balanced_age_train.to_csv(f"{outdir}/{anatomy}_age_balanced.csv", index=False)

