import tkinter as tk
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import pywt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import os
from PIL import Image, ImageTk


# Chargement du modèle
MODEL_PATH = r'G:\Mon Drive\ESIEE\Année 4\23_E4_PRJ_4000S2 - Projet semestre 2\PROJETET4\Conv1D.h5'
loaded_model = load_model(MODEL_PATH)
print("Modèle chargé avec succès.")

def load_and_filter_data(csv_path, model):
    # Chargement des données ECG depuis un fichier CSV.
    df_patient = pd.read_csv(csv_path)

 # Sélectionner uniquement 5 % des données de manière séquentielle.
    sample_size = int(len(df_patient) * 0.02)  # Calcul de 5 % de la taille totale
    df_sampled = df_patient.iloc[:sample_size]  # Sélection séquentielle des premières lignes
    
    
    data_array = df_sampled.iloc[:, 0].values
    
    # Normalisation des données.
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data_array.reshape(-1, 1)).flatten()
    
    # Application du débruitage par ondelettes.
    cleaned_data = wavelet_denoising(data_normalized)
    
    # Découpage des données nettoyées en fenêtres.
    window_size = 360  # Taille de la fenêtre en échantillons.
    stride = 360  # Pas de décalage entre les fenêtres consécutives.
    windows = [cleaned_data[i:i + window_size] for i in range(0, len(cleaned_data) - window_size + 1, stride)]
    
    # Préparation des données pour la prédiction.
    data_for_prediction = np.array(windows)
    if len(data_for_prediction.shape) == 2:
        data_for_prediction = np.expand_dims(data_for_prediction, axis=2)

    # Prédiction des classes pour chaque fenêtre.
    predictions = model.predict(data_for_prediction)
    
    # Filtrer pour obtenir uniquement les fenêtres correspondant à des prédictions de classe 1,
    # mais conserver les probabilités associées pour chaque fenêtre filtrée.
    filtered_windows = []
    filtered_predictions = []
    for i, prediction in enumerate(predictions):
        if prediction > 0.5:  # Seuil pour décider si une fenêtre correspond à la classe 1.
            filtered_windows.append(windows[i])
            filtered_predictions.append(prediction)  # Conserver la probabilité associée.
    print("nombre de prediction classe 1 : " + str(len(filtered_predictions)))
    print("nombre de prédiction en tout : " + str(len(predictions)))

    return filtered_windows, filtered_predictions





def wavelet_denoising(data, wavelet='sym4', level=1):
    coeffs = pywt.wavedec(data, wavelet, mode="per")
    sigma = (1/0.6745) * np.median(np.abs(coeffs[-level] - np.median(coeffs[-level]))) / 0.6745
    uthresh = sigma * np.sqrt(2*np.log(len(data)))
    coeffs[1:] = [pywt.threshold(i, value=uthresh, mode='soft') for i in coeffs[1:]]
    reconstructed_signal = pywt.waverec(coeffs, wavelet, mode='per')
    return reconstructed_signal

def plot_heartbeat_window(window_values):
    time_axis = np.arange(0, len(window_values))
    rect_width, rect_height = 800, 438  # Ajustez ces valeurs selon les dimensions souhaitées
    fig, ax = plt.subplots(figsize=(rect_width / 100, rect_height / 100))
    ax.plot(time_axis, window_values)  # Utilisez plot pour un tracé continu
    ax.set_xlabel('Temps')
    ax.set_ylabel('Amplitude')
    ax.set_title('Signal ECG')
    return fig


class ECGViewer:
    def __init__(self, master):
        self.master = master
        self.current_index = -1
        self.fenetres_validees = []  # Initialiser la liste des fenêtres validées
        self.filtered_windows, self.predicted_classes = load_and_filter_data(r'C:\Users\paulg\Documents\GitHub\Tkinter-Designer\build\207_ECG_test.csv', loaded_model)
        
        # Charger et afficher l'image de fond
        self.set_background(r'C:\Users\paulg\Documents\GitHub\Tkinter-Designer\build\assets\frame2\image_1.png')  # Remplacer 'image1.jpg' par le chemin de votre image
        
        self.setup_ui()

    def set_background(self, image_path):
        # Charger l'image avec PIL
        self.background_image = Image.open(image_path)
        self.background_image = self.background_image.resize((800, 600), Image.ANTIALIAS)  # Redimensionner selon les besoins de votre fenêtre
        self.background_photo = ImageTk.PhotoImage(self.background_image)
        
        # Créer un label pour l'image de fond, et l'étendre pour remplir l'espace disponible
        self.background_label = tk.Label(self.master, image=self.background_photo)
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)

    def setup_ui(self):
        
        self.label_prediction = tk.Label(self.master, text="Prédiction : ", font=("Arial", 14))
        self.label_prediction.pack(pady=10)

        self.button_validate = tk.Button(self.master, text="Validé", command=self.validation)
        self.button_validate.pack(pady=20)

        self.button_error = tk.Button(self.master, text="Erreur", command=self.plot_next_window)
        self.button_error.pack(pady=25)

        self.plot_next_window()  # Appeler initialement pour afficher la première fenêtre

    def validation(self):
        # Ajoute la fenêtre actuelle à la liste des fenêtres validées
        if self.current_index < len(self.filtered_windows):
            self.fenetres_validees.append(self.filtered_windows[self.current_index])
            self.plot_next_window()  # Passe à la fenêtre suivante après validation

    def plot_next_window(self):
        self.current_index += 1
        if self.current_index < len(self.filtered_windows):
            window = self.filtered_windows[self.current_index]
            fig = plot_heartbeat_window(window)

            if hasattr(self, 'canvas_widget'):
                self.canvas_widget.destroy()  # Supprimer le widget précédent
            self.canvas_figure = FigureCanvasTkAgg(fig, master=self.master)
            self.canvas_figure.draw()
            self.canvas_widget = self.canvas_figure.get_tk_widget()
            self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            plt.close(fig)  # Fermer la figure maintenant qu'elle est dessinée
        else:
            self.master.destroy()  # Ferme la fenêtre actuelle
            self.display_results()  # Affiche les résultats dans une nouvelle fenêtre

    def display_results(self):
        # Crée une nouvelle fenêtre pour afficher les résultats
        result_window = tk.Tk()
        result_window.geometry("600x400")  # Ajustez selon les besoins
        result_window.title("Résultats de Validation")

        # Affiche les résultats. Exemple : Nombre de fenêtres validées
        label_result = tk.Label(result_window, text=f"Nombre de fenêtres validées : {len(self.fenetres_validees)}", font=("Arial", 14))
        label_result.pack(pady=10)

        # Vous pouvez étendre cette partie pour afficher plus de détails ou des statistiques sur les fenêtres validées

        result_window.mainloop()

# Le reste du code pour initialiser l'application reste inchangé

# Le reste de votre code pour charger le modèle, définir la fonction de débruitage, etc., reste inchangé.


root = tk.Tk()
root.geometry("800x600")  # Ajuster selon les besoins
app = ECGViewer(root)
root.mainloop()
