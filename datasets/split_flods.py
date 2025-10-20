import os
import os.path as osp
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
##


# ============================================================
# Step 1: Patient-level stratified splitting
# ============================================================

# Define dataset name and label CSV path
NAME_DATASET = 'TCGA_ESCA'
PATH_LABEL_CSV = f'./ESCA/five_flod/{NAME_DATASET}_path_full_subtype.csv'

# Read patient-slide-label mapping file
data_csv = pd.read_csv(PATH_LABEL_CSV)
data_csv = data_csv.loc[:, ['patient_id', 'pathology_id', 'subtype', 'label']]

# Check for patients with inconsistent labels across their slides
gps = data_csv.groupby('patient_id')
for k, v in gps:
    if len(v.index) > 1:
        for i in range(len(v.index)):
            if v.iloc[i, 3] != v.iloc[0, 3]:
                print(f'The patient {k} has slides with different subtypes/labels')

print(f"There are {len(data_csv)} WSIs")
print(f"There are {len(gps)} patients")

# Deduplicate to patient-level
data_pat = data_csv.drop_duplicates(subset=['patient_id'], keep='first').loc[:, ['patient_id', 'subtype', 'label']]
data_pat = data_pat.reset_index(drop=True)

# Directory to save split results
DIR_TO_SAVE = f'./data_split/{NAME_DATASET.lower()}'

SEED = 42
skf = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)

# Perform 5-fold patient-level split
for i, (train_index, test_index) in enumerate(skf.split(data_pat['patient_id'], data_pat['label'])):
    print(f"{i+1}-th fold:")

    pat_train, y_train = data_pat['patient_id'][train_index], data_pat['label'][train_index]
    pat_test,  y_test  = data_pat['patient_id'][test_index],  data_pat['label'][test_index]

    # Further split train into train/val
    pat_train = pat_train.reset_index(drop=True)
    y_train   = y_train.reset_index(drop=True)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    
    for j, (new_train_index, new_val_index) in enumerate(sss.split(pat_train, y_train)):
        print(f"\tFurther split into train/val")
        pat_new_train, y_new_train = pat_train[new_train_index], y_train[new_train_index]
        pat_new_val,   y_new_val   = pat_train[new_val_index], y_train[new_val_index]

    # Print statistics
    print(f"\t# train/val/test: {len(pat_new_train)}/{len(pat_new_val)}/{len(pat_test)}")

    # Path to save .npz file
    PATH_TO_NPZ = osp.join('/ESCA/five_flod', f'{NAME_DATASET}-fold{i}.npz')

    # Save patient-level split
    np.savez(PATH_TO_NPZ, 
        train_patients=list(pat_new_train), 
        val_patients=list(pat_new_val), 
        test_patients=list(pat_test)
    )
    print(f"\t[info] npz file saved at {PATH_TO_NPZ}")

# ============================================================
# Step 2: Map patient-level split to slide-level
# ============================================================

def map_patients_to_slides(npz_file_path, csv_file_path, output_csv_path):
    """
    Map patient-level splits back to slide-level splits and save as CSV.
    """
    # Load split file (.npz)
    data = np.load(npz_file_path)
    train_patients = data['train_patients'].tolist()
    val_patients   = data['val_patients'].tolist()
    test_patients  = data['test_patients'].tolist()

    # Load patient-slide-label mapping file
    labels_df = pd.read_csv(csv_file_path)

    # Helper function to fetch slides for each patient
    def find_labels(patient_ids):
        results = []
        for patient in patient_ids:
            label_rows = labels_df[labels_df['patient_id'] == patient]
            if not label_rows.empty:
                for _, row in label_rows.iterrows():
                    results.append((patient, row['pathology_id'], row['subtype'], row['label']))
            else:
                results.append((patient, None, None, None))  # No match found
        return results

    # Map patients to slides
    train_results = find_labels(train_patients)
    val_results   = find_labels(val_patients)
    test_results  = find_labels(test_patients)

    # Convert to DataFrames
    train_df = pd.DataFrame(train_results, columns=['train_patient_id', 'train_pathology_id', 'train_subtype', 'train_label'])
    val_df   = pd.DataFrame(val_results,   columns=['val_patient_id', 'val_pathology_id', 'val_subtype', 'val_label'])
    test_df  = pd.DataFrame(test_results,  columns=['test_patient_id', 'test_pathology_id', 'test_subtype', 'test_label'])

    # Concatenate by columns
    combined_df = pd.concat([train_df, val_df, test_df], axis=1)

    # Save to CSV
    combined_df.to_csv(output_csv_path, index=False)
    print(f"[info] Results saved to: {output_csv_path}")

# Example usage
# map_patients_to_slides(
#     npz_file_path='./ESCA/five_flod/TCGA_ESCA-fold0.npz',
#     csv_file_path=PATH_LABEL_CSV,
#     output_csv_path=./result/ESCA/patient_labels_fold0_parallel.csv'
# )