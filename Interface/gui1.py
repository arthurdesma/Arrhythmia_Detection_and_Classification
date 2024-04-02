import tkinter as tk
from tkinter import messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pywt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import os

MODEL_PATH = r'path_to_your_model/Conv1D.h5'

def wavelet_denoising(data, wavelet='sym4', level=1):
    coeffs = pywt.wavedec(data, wavelet, mode="per")
    sigma = (1/0.6745) * np.median(np.abs(coeffs[-level] - np.median(coeffs[-level]))) / 0.6745
    uthresh = sigma * np.sqrt(2*np.log(len(data)))
    coeffs[1:] = [pywt.threshold(i, value=uthresh, mode='soft') for i in coeffs[1:]]
    reconstructed_signal = pywt.waverec(coeffs, wavelet, mode='per')
    return reconstructed_signal

def load_and_filter_data(csv_path, model):
    df_patient = pd.read_csv(csv_path)
    data_array = df_patient.iloc[:, 0].values

    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data_array.reshape(-1, 1)).flatten()
    
    cleaned_data = wavelet_denoising(data_normalized)

    window_size = 360
    stride = 360
    windows = [cleaned_data[i:i + window_size] for i in range(0, len(cleaned_data) - window_size + 1, stride)]
    
    data_for_prediction = np.array(windows)
    if len(data_for_prediction.shape) == 2:
        data_for_prediction = np.expand_dims(data_for_prediction, axis=2)

    predictions = model.predict(data_for_prediction)
    
    return windows, predictions

class ECGViewer:
    def __init__(self, master, csv_path, model):
        self.master = master
        self.windows, self.predictions = load_and_filter_data(csv_path, model)
        self.current_index = 0
        self.fenetres_validees = []
        self.setup_ui()

    def setup_ui(self):
        self.fig, self.ax = plt.subplots(figsize=(6, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)  
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.validate_button = tk.Button(self.master, text="Validé", command=self.validate)
        self.validate_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.reject_button = tk.Button(self.master, text="Erreur", command=self.next_window)
        self.reject_button.pack(side=tk.RIGHT, padx=5, pady=5)

        self.update_plot()

    def update_plot(self):
        if self.current_index < len(self.windows):
            self.ax.clear()
            self.ax.plot(self.windows[self.current_index])
            self.ax.set_title(f'Fenêtre {self.current_index+1} sur {len(self.windows)}')
            self.canvas.draw()
        else:
            self.display_results()

    def next_window(self):
        self.current_index += 1
        self.update_plot()

    def validate(self):
        if self.current_index < len(self.windows):
            self.fenetres_validees.append((self.current_index, self.predictions[self.current_index]))
        self.next_window()

    def display_results(self):
        messagebox.showinfo("Fin des données", f"Nombre de fenêtres validées : {len(self.fenetres_validees)}")
        self.master.destroy()
        # Vous pouvez également afficher les résultats dans une nouvelle fenêtre si nécessaire.

if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    root = tk.Tk()
    root.geometry("800x600")
    app = ECGViewer(root, 'path_to_your_csv_file.csv', model)
    root.mainloop()
