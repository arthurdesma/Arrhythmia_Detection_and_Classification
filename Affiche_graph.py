import pandas as pd
import numpy as np
import pywt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

def plot_ecg_windows(csv_path, model_path):
    # Charger le modèle
    loaded_model = load_model(model_path)
    
    # Charger les données du CSV
    df_patient = pd.read_csv(csv_path)
    df_patient = df_patient.iloc[:, 0]
    
    # Convertir les données en un tableau NumPy
    data_array = df_patient.values
    
    hz = 360
    s = 1
    t = hz*s
    
    # Remodeler le tableau
    rows = len(data_array) // t
    reshaped_data = data_array[:rows * t].reshape(rows, t)
    
    # Créer le DataFrame à partir des données remodelées
    df_patient = pd.DataFrame(reshaped_data)
    
    # Normaliser les données
    scaler = MinMaxScaler()
    df_patient = scaler.fit_transform(df_patient)
    df_patient = pd.DataFrame(df_patient)
    
    # Appliquer le débruitage par ondelettes
    def madev(d, axis=None):
        return np.mean(np.absolute(d - np.mean(d, axis)), axis)
    
    def wavelet_denoising(x, wavelet='sym4', level=1):
        coeff = pywt.wavedec(x, wavelet, mode="per")
        sigma = (1/0.6745) * madev(coeff[-level])
        uthresh = sigma * np.sqrt(2 * np.log(len(x)))
        coeff[1:] = [pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:]]
        return pywt.waverec(coeff, wavelet, mode='per')
    
    df_patient = wavelet_denoising(df_patient, wavelet='sym4', level=2)
    df_patient = pd.DataFrame(df_patient)
    
    # Faire des prédictions
    prediction = loaded_model.predict(df_patient)
    
    # Convertir les prédictions en classes
    predicted_classes = (prediction > 0.5).astype(int)
    
    # Stocker les positions des prédictions positives (classe 1)
    positions_predictions = [i for i, pred_class in enumerate(predicted_classes) if pred_class == 1]
    
    # Définir la fonction pour afficher une fenêtre de signal cardiaque
    def plot_heartbeat_window(window_values):
        time_axis = np.arange(0, len(window_values))
        plt.plot(time_axis, window_values)
        plt.xlabel('Temps')
        plt.ylabel('Amplitude')
        plt.title('Signal cardiaque')
        plt.grid(True)
        plt.show()
    
    # Afficher les fenêtres des classes 0
    for position in positions_predictions:
        heartbeat_values = df_patient.iloc[position, :]
        plot_heartbeat_window(heartbeat_values)


plot_ecg_windows(r"G:\Autres ordinateurs\Mon ordinateur portable\Documents\GitHub\ProjetE4\data_creation\201_ECG.csv", r'G:\Mon Drive\ESIEE\Année 4\23_E4_PRJ_4000S2 - Projet semestre 2\PROJETET4\Conv1D.h5')
