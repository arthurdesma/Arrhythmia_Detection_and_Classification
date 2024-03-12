import os
import pandas as pd
import numpy as np
import joblib
import pywt
from sklearn.preprocessing import MinMaxScaler

def process_patient_data(patient_number, data_creation_dir=""):
    ecg_file_path = os.path.join(data_creation_dir, f"{patient_number}_ECG.csv")
    annotations_file_path = os.path.join(data_creation_dir, f"{patient_number}_Annotations.csv")
    
    patient_X = []
    patient_Y = []
    
    try:
        ecg_df = pd.read_csv(ecg_file_path)
        annotations_df = pd.read_csv(annotations_file_path)
    except FileNotFoundError:
        print(f"Files for patient {patient_number} not found. Skipping...")
        return [], []
    
    first_column_name = ecg_df.columns[0]
    second_column_name = ecg_df.columns[1] if len(ecg_df.columns) > 1 else None

    sampling_rate = 360  # Hz
    window_size_seconds = 2  # Seconds before and after annotation
    window_size_samples = window_size_seconds * sampling_rate

    for _, row in annotations_df.iterrows():
        annotation_point = row['Annotation']
        category = row['Category']
        
        start_point = max(0, annotation_point - window_size_samples)
        end_point = min(len(ecg_df), annotation_point + window_size_samples)
        
        # Process data from the first column
        window_data_first_column = ecg_df.iloc[start_point:end_point][first_column_name].to_numpy()
        if len(window_data_first_column) < window_size_samples * 2:
            window_data_first_column = np.pad(window_data_first_column, (0, window_size_samples * 2 - len(window_data_first_column)), 'constant')
        
        patient_X.append(window_data_first_column)
        patient_Y.append(category)
        
        # If there's a second column, process it and add as a new entry
        if second_column_name:
            window_data_second_column = ecg_df.iloc[start_point:end_point][second_column_name].to_numpy()
            if len(window_data_second_column) < window_size_samples * 2:
                window_data_second_column = np.pad(window_data_second_column, (0, window_size_samples * 2 - len(window_data_second_column)), 'constant')
            
            patient_X.append(window_data_second_column)
            patient_Y.append(category)  # Repeat category for the new entry
    
    return patient_X, patient_Y

# Process data for patient 100 only
patient_number = "100"
patient_X, patient_Y = process_patient_data(patient_number)

# Convert lists to arrays
X = np.array(patient_X)
Y = np.array(patient_Y)

# Perform wavelet denoising on X
def madev(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def wavelet_denoising(x, wavelet='sym4', level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1/0.6745) * madev(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode='per')

X_denoised = np.array([wavelet_denoising(x) for x in X])

# Scale X_denoised
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_denoised)

# Load SVM model
svm_classifier = joblib.load('svm_model.pkl')

# Test the model on X_scaled and get predictions
predictions = svm_classifier.predict(X_scaled)

print("Predictions:", predictions)
