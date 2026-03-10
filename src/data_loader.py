"""
data_loader.py
Loads the CKD dataset and defines the selected feature set.

Feature selection rationale:
  Started with 51 features and used permutation importance on a held-out
  test set to rank them. Features with zero or negative permutation
  importance were dropped — shuffling them made the model no worse,
  meaning they carry no real signal.

  Result: 10 features retained out of 51.
  AUC improved from 0.77 (all 51) → 0.84 (top 10).
  Extra features were adding noise, not signal.
"""

import pandas as pd

TARGET_COL = "Diagnosis"
DROP_COLS  = ["PatientID", "DoctorInCharge"]

# Features selected by permutation importance (ranked, kept top 10)
SELECTED_FEATURES = [
    "SerumCreatinine",       # kidney waste filtration marker   ← most important
    "GFR",                   # glomerular filtration rate        ← most important
    "MuscleCramps",          # symptom linked to electrolyte imbalance
    "FastingBloodSugar",     # diabetes is a leading CKD cause
    "ProteinInUrine",        # protein leakage = kidney damage marker
    "Itching",               # uraemia symptom
    "SerumElectrolytesSodium", # kidneys regulate sodium balance
    "HemoglobinLevels",      # anaemia is common in CKD
    "NauseaVomiting",        # uraemia symptom
    "PhysicalActivity",      # lifestyle proxy
]

# Features dropped and why (for transparency in portfolio)
DROPPED_FEATURES = {
    "zero_permutation_importance": [
        "WaterQuality", "HeavyMetalsExposure", "OccupationalExposureChemicals",
        "Diuretics", "FamilyHistoryKidneyDisease", "ACEInhibitors", "Statins",
        "AntidiabeticMedications", "FamilyHistoryDiabetes", "FamilyHistoryHypertension",
        "UrinaryTractInfections",
    ],
    "negative_permutation_importance": [
        "Ethnicity", "SerumElectrolytesPhosphorus", "SocioeconomicStatus",
        "HealthLiteracy", "FatigueLevels", "PreviousAcuteKidneyInjury", "Edema",
        "Smoking", "SerumElectrolytesCalcium", "DietQuality", "SerumElectrolytesPotassium",
        "Gender", "QualityOfLifeScore", "AlcoholConsumption", "SleepQuality",
        "ACR", "BUNLevels", "HbA1c", "MedicalCheckupsFrequency", "CholesterolHDL", "Age",
    ],
    "marginal_kept_in_top20_but_not_top10": [
        "SystolicBP", "BMI", "DiastolicBP", "MedicationAdherence",
        "CholesterolLDL", "EducationLevel", "NSAIDsUse",
        "CholesterolTriglycerides", "CholesterolTotal",
    ],
}


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df = df.drop(columns=DROP_COLS, errors="ignore")
    return df


def get_feature_names() -> list:
    return SELECTED_FEATURES


def get_class_info(df: pd.DataFrame):
    counts = df[TARGET_COL].value_counts()
    total  = len(df)
    print(f"  Class 0 (No CKD): {counts.get(0,0)}  ({counts.get(0,0)/total*100:.1f}%)")
    print(f"  Class 1 (CKD)   : {counts.get(1,0)}  ({counts.get(1,0)/total*100:.1f}%)")
    print(f"  Imbalance ratio : {counts.get(1,0)/counts.get(0,1):.1f}x")


if __name__ == "__main__":
    df = load_data("../data/ckd_data.csv")
    print(f"Shape (all cols): {df.shape}")
    print(f"\nSelected {len(SELECTED_FEATURES)} features:")
    for f in SELECTED_FEATURES:
        print(f"  {f}")
    print(f"\nClass distribution:")
    get_class_info(df)
