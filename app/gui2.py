import tkinter as tk
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import pywt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk
import sys
from gui3 import GUI3
from gui4 import affiche_stat


# Obtenez le chemin absolu du répertoire contenant le script en cours d'exécution
script_dir = Path(__file__).resolve().parent

# Chargement du modèle
MODEL_FILENAME = "Conv1D.h5" # Nom du fichier du modèle
MODEL_FILENAME_MAL = "CNN_anormaux.h5"
MODEL_PATH = script_dir / MODEL_FILENAME
MODEL_PATH_MAL = script_dir / MODEL_FILENAME_MAL
ASSETS_PATH=  script_dir / "assets"

loaded_model = load_model(MODEL_PATH)
loaded_model_mal = load_model(MODEL_PATH_MAL)

print("Modèle chargé avec succès.")


def load_and_filter_data(csv_path, model, model_mal):
    global predictions, anomalies
    # Chargement des données ECG depuis un fichier CSV.

    df_patient = pd.read_csv(csv_path)



    # Sélectionner uniquement 5 % des données de manière séquentielle.
   
    #sample_size = int(len(df_patient))  # Calcul de 5 % de la taille totale
    sample_size = 60*360 # taille de 5 minutes : 5*60*360
    print("nombre de point " + str(sample_size))

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

    anomalies = []

    for i in range(len(predictions)):  # Itérer sur la longueur du tableau de prédictions
        if predictions[i] > 0.5:
            anomalies.append(predictions[i])

    print("nombre de >0.5 " + str(len(anomalies)))
    print("nombre de prédictions en tout " + str(len(predictions)))

    # Trouver les indices des prédictions > 0.5
    indices = np.where(predictions > 0.5)[0]

    print(indices)

    # Récupérer les lignes correspondantes du tableau NumPy
    df_data_predictions_1 = data_for_prediction[indices]

    print(df_data_predictions_1)

    predictions_mal = model_mal.predict(df_data_predictions_1)
    predicted_labels = np.argmax(predictions_mal, axis=1)

    print(predicted_labels)

    # Filtrer pour obtenir uniquement les fenêtres correspondant à des prédictions de classe 1,

    # mais conserver les probabilités associées pour chaque fenêtre filtrée.

    filtered_windows = []

    filtered_predictions = []



    return filtered_windows, filtered_predictions, data_for_prediction, predictions, predicted_labels





def wavelet_denoising(data, wavelet='sym4', level=1):

    coeffs = pywt.wavedec(data, wavelet, mode="per")

    sigma = (1/0.6745) * np.median(np.abs(coeffs[-level] - np.median(coeffs[-level]))) / 0.6745

    uthresh = sigma * np.sqrt(2*np.log(len(data)))

    coeffs[1:] = [pywt.threshold(i, value=uthresh, mode='soft') for i in coeffs[1:]]

    reconstructed_signal = pywt.waverec(coeffs, wavelet, mode='per')

    return reconstructed_signal




def plot_heartbeat_window(window_values, predictions):
    # Création de l'axe des temps en secondes
    time_axis = np.arange(0, len(window_values)) / 360  # Convertir en secondes

    rect_width, rect_height = 800, 438  # Ajustez ces valeurs selon les dimensions souhaitées

    fig, ax = plt.subplots(figsize=(rect_width / 100, rect_height / 100))

    prev_end = 0

    for i in range(0, len(window_values), 360):
        if i + 360 <= len(window_values):
            if predictions[i // 360] > 0.5:
                ax.plot(time_axis[i:i+360] + prev_end, window_values[i:i+360], color='red')  # Tracé en rouge si la prédiction est supérieure à 0.5
            else:
                ax.plot(time_axis[i:i+360] + prev_end, window_values[i:i+360], color='blue')  # Tracé en bleu sinon

    ax.set_xlabel('Temps (secondes)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Signal ECG')

    return fig



class ECGViewer:
    def __init__(self, master):
        self.path = sys.argv[1]
        self.master = master
        self.current_index = -1
        self.fenetres_validees = []  # Initialiser la liste des fenêtres validées
        self.filtered_windows, self.predicted_classes, self.data_for_prediction, self.predictions, self.prediction_mal = load_and_filter_data(sys.argv[1], loaded_model , loaded_model_mal)

        # Charger et afficher l'image de fond
        self.set_background(ASSETS_PATH / "frame2" / "image_1.png")  # Remplacer 'image1.jpg' par le chemin de votre image

        self.setup_ui()



    def set_background(self, image_path):

        # Charger l'image avec PIL

        self.background_image = Image.open(image_path)

        self.background_image = self.background_image.resize((900, 700), Image.Resampling.LANCZOS)  # Redimensionner selon les besoins de votre fenêtre

        self.background_photo = ImageTk.PhotoImage(self.background_image)

        

        # Créer un label pour l'image de fond, et l'étendre pour remplir l'espace disponible

        self.background_label = tk.Label(self.master, image=self.background_photo)

        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)



    def setup_ui(self):
        self.label_prediction = tk.Label(self.master, text="Prédiction : ", font=("Arial", 14))
        self.label_prediction.pack(pady=10)

        self.values = []  # Create an empty list to store the values

        self.button_normal = tk.Button(self.master, text="ECG normal", command=lambda: self.add_value(1))
        self.button_normal.pack(pady=20)

        self.button_indetermine = tk.Button(self.master, text="indeterminé", command=lambda: self.add_value(0))
        self.button_indetermine.pack(pady=25)

        self.button_anormal = tk.Button(self.master, text="ECG anormal", command=lambda: self.add_value(-1))
        self.button_anormal.pack(pady=25)

        self.plot_next_window()  # Appeler initialement pour afficher la première fenêtre

    def add_value(self, value):
        self.values.append(value)
        print(self.values)  
        self.plot_next_window()
        

    def plot_next_window(self):

        self.current_index += 5

        if self.current_index < len(self.data_for_prediction):

            window = self.data_for_prediction[self.current_index:self.current_index+5].flatten()

            predictions = self.predictions[self.current_index:self.current_index+5]

            fig = plot_heartbeat_window(window, predictions)



            if hasattr(self, 'canvas_widget'):

                self.canvas_widget.destroy()  # Supprimer le widget précédent

            self.canvas_figure = FigureCanvasTkAgg(fig, master=self.master)

            self.canvas_figure.draw()

            self.canvas_widget = self.canvas_figure.get_tk_widget()

            self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        else:

            self.display_results()  # Affiche les résultats dans une nouvelle fenêtre




    def display_results(self):
        # Crée une nouvelle fenêtre pour afficher les résultats
        result_window = tk.Toplevel(self.master)
        result_window.geometry("400x150")  # Ajustez selon les besoins
        result_window.title("Résultats de Validation")

        # Charge l'image de fond
        background_image = tk.PhotoImage(file=ASSETS_PATH / "frame2" / "image_1.png")  # Assurez-vous que le chemin est correct
        background_label = tk.Label(result_window, image=background_image)
        background_label.place(x=0, y=0, relwidth=1, relheight=1)  # Ajuste l'image à la taille de la fenêtre

        # Garde une référence de l'image pour éviter qu'elle ne soit effacée par le ramasse-miettes de Python
        background_label.image = background_image

        # Affiche les résultats. Exemple : Nombre de fenêtres validées
        label_result = tk.Label(result_window, text=f"{len(self.values)} fenêtres validées", font=("Arial", 14), bg='#1e6091',fg='white')
        label_result.pack(pady=10)

        # Bouton pour passer à la page gui3
        button_gui3 = tk.Button(result_window, text="Aller à la page d'analyse temporelle des anomalies", command=lambda: self.go_to_gui3(result_window))
        button_gui3.pack(pady=20)


    def go_to_gui3(self, result_window):
        result_window.destroy()  # Ferme la fenêtre de résultats
        root.destroy()

        # Crée une nouvelle fenêtre pour gui3
        gui3_window = tk.Tk()
        gui3_window.geometry("900x700")  # Ajustez selon les besoins
        gui3_window.title("Page gui3")

        # Passe la liste des valeurs à la page gui3
        gui3 = GUI3(gui3_window, self.values)

        gui3_window.mainloop()
        affiche_stat(str(len(anomalies)),str(len(predictions)-len(anomalies)) ,str(len(predictions)), np.sum(self.prediction_mal == 1), np.sum(self.prediction_mal == 2), np.sum(self.prediction_mal == 3), np.sum(self.prediction_mal == 4))




root = tk.Tk()

root.geometry("900x700")  # Ajuster selon les besoins

app = ECGViewer(root)

root.mainloop()