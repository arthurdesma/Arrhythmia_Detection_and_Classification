from tkinter import Tk, Canvas, PhotoImage, OptionMenu, StringVar, Button

from pathlib import Path

import matplotlib.pyplot as plt

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from io import BytesIO

import pandas as pd

import numpy as np

import pywt

import sys

from sklearn.preprocessing import MinMaxScaler

from pathlib import Path

import subprocess





def relative_to_assets(path: str) -> Path:

    OUTPUT_PATH = Path(__file__).parent

    ASSETS_PATH = OUTPUT_PATH / "assets" / "frame1"

    return ASSETS_PATH / Path(path)



def plot_ecg_from_csv(csv_path, num_beats=60):

    # Read CSV file

    df_patient = pd.read_csv(csv_path)

    df_patient = df_patient.iloc[:, 0]

    

    # Convert the Series to a NumPy array

    data_array = df_patient.values



    hz = 360

    s = 1

    t = hz * s



    # Concatenate the data of the first num_beats beats

    concatenated_data = np.concatenate([data_array[i * t:(i + 1) * t] for i in range(num_beats)])



    # Plot ECG signal for the concatenated beats

    plt.figure(figsize=(10, 4))

    plt.plot(np.arange(len(concatenated_data))/360, concatenated_data)  # Adjust x-axis for time

    plt.xlabel('Temps (s)')  # Update xlabel

    plt.ylabel('Amplitude')

    plt.title(f'Signal ECG ({num_beats} secondes)')

    plt.grid(True)

    

    # Adjust figure size to fit canvas

    canvas_size = (1321.0 - 120.0, 793.0 - 355.0)  # Width, Height of rectangle

    figure_size = (canvas_size[0] / 100, canvas_size[1] / 100)  # Convert to inches

    plt.gcf().set_size_inches(figure_size)



    # Convert plot to a tkinter-compatible image

    buffer = BytesIO()

    plt.savefig(buffer, format='png')

    buffer.seek(0)

    img_data = buffer.getvalue()

    buffer.close()

    

    return img_data





def update_beats():

    global num_beats

    num_beats = int(var.get())  # Utilise var.get() pour obtenir la nouvelle valeur du menu déroulant

    window.destroy()  # Ferme la fenêtre actuelle

    start_gui(file_path)  # Ouvre une nouvelle fenêtre avec les mises à jour



def update_page():
    window.destroy()
    subprocess.run(["python", "gui2.py", file_path])

    



def start_gui(file_path_arg):  # Renommez le paramètre pour éviter le conflit avec la variable globale

    global window, canvas, var, file_path, num_beats  # Déclarez les variables globales

    file_path = file_path_arg # Déclarez les variables globales

    window = Tk()

    window.geometry("1440x818")

    window.configure(bg="#FFFFFF")



    canvas = Canvas(

        window,

        bg="#FFFFFF",

        height=818,

        width=1440,

        bd=0,

        highlightthickness=0,

        relief="ridge"

    )

    canvas.place(x=0, y=0)



    image_image_1 = PhotoImage(file=relative_to_assets("image_1.png"))

    image_1 = canvas.create_image(720.0, 409.0, image=image_image_1)



    canvas.create_text(

        238.0,

        61.0,

        anchor="nw",

        text="Analyse des résultats",

        fill="#FFFFFF",

        font=("DoppioOne Regular", 96 * -1)

    )



    canvas.create_rectangle(

        120.0,

        355.0,

        1321.0,

        793.0,

        fill="#D9D9D9",

        outline=""

    )



    image_image_2 = PhotoImage(file=relative_to_assets("image_2.png"))

    image_2 = canvas.create_image(719.0, 259.0, image=image_image_2)



    canvas.create_text(

        403.0,

        233.0,

        anchor="nw",

        text="Affichage graphique de votre ECG",

        fill="#FFFFFF",

        font=("Nokora Regular", 40 * -1)

    )



    # Dropdown menu for selecting number of beats

    options = [str(i) for i in range(1, 61)]  # Options from 1 to 60

    var = StringVar()  # Create StringVar after creating the main window

    var.set("60")  # Default value

    dropdown = OptionMenu(window, var, *options)

    dropdown.place(x=100.0, y=243.0)  # Placez le menu déroulant à l'endroit souhaité sur la fenêtre



    # Button to update number of beats

    button_image_1 = PhotoImage(file=relative_to_assets("button_1.png"))

    update_button_1 = Button(

        window,

        image=button_image_1,

        borderwidth=0,

        highlightthickness=0,

        command=update_beats,

        relief="flat"

    )



    button_image_2 = PhotoImage(file=relative_to_assets("button_2.png"))

    update_button_2 = Button(

        window,

        image=button_image_2,

        borderwidth=0,

        highlightthickness=0,

        command=update_page,

        relief="flat"

    )  

    update_button_1.place(x=190.0, y=243.0)  # Placez le bouton à l'endroit souhaité sur la fenêtre

    update_button_2.place(x=100 , y=100)



    # Plot and display ECG signal

    img_data = plot_ecg_from_csv(file_path, num_beats=num_beats)

    img = PhotoImage(data=img_data)

    canvas.create_image(720.0, 574.0, image=img)



    image_image_3 = PhotoImage(file=relative_to_assets("image_3.png"))

    image_3 = canvas.create_image(1395.0, 769.0, image=image_image_3)

    window.iconphoto(True, image_image_3)

    window.resizable(False, False)

    window.mainloop()

    print("Chemin du fichier sélectionné:", file_path)





if __name__ == "__main__":

    if len(sys.argv) > 1:

        file_path = sys.argv[1]

    else:

        # Si aucun chemin de fichier n'est fourni en argument, demandez à l'utilisateur de saisir le chemin

        file_path = input("Veuillez entrer le chemin du fichier CSV : ")

    num_beats = 60  # Initialise num_beats

    start_gui(file_path)

