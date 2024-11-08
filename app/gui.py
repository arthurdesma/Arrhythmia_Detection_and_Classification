# This file was generated by the Tkinter Designer by Parth Jadhav
# https://github.com/ParthJadhav/Tkinter-Designer





from pathlib import Path

from gui1 import start_gui



# from tkinter import *

# Explicit imports to satisfy Flake8

import subprocess

from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage, filedialog

from pathlib import Path



# Chemin de sortie du script

OUTPUT_PATH = Path(__file__).parent



# Chemin relatif vers le dossier des ressources (assets)

ASSETS_PATH = OUTPUT_PATH / "assets" / "frame0"





def relative_to_assets(path: str) -> Path:

    return ASSETS_PATH / Path(path)





def select_file():

    file_path = filedialog.askopenfilename()

    if file_path:

        print("Chemin du fichier sélectionné:", file_path)

        window.destroy()  # Fermer la fenêtre après la sélection du fichier

        #subprocess.run(["python", "gui1.py", file_path])  # Passer le chemin du fichier comme argument

        subprocess.run(["python", "gui1.py", file_path])



window = Tk()



window.geometry("1440x818")

window.configure(bg = "#FFFFFF")





canvas = Canvas(

    window,

    bg = "#FFFFFF",

    height = 818,

    width = 1440,

    bd = 0,

    highlightthickness = 0,

    relief = "ridge"

)



canvas.place(x = 0, y = 0)

image_image_1 = PhotoImage(

    file=relative_to_assets("image_1.png"))

image_1 = canvas.create_image(

    720.0,

    409.0,

    image=image_image_1

)



button_image_1 = PhotoImage(

    file=relative_to_assets("button_1.png"))

button_1 = Button(

    image=button_image_1,

    borderwidth=0,

    highlightthickness=0,

    command=select_file,

    relief="flat"

)

button_1.place(

    x=482.0,

    y=566.0,

    width=473.0,

    height=158.0

)



canvas.create_text(

    356.0,

    388.0,

    anchor="nw",

    text="Analyse ton ECG",

    fill="#CF5E92",

    font=("DoppioOne Regular", 96 * -1)

)



image_image_2 = PhotoImage(

    file=relative_to_assets("image_2.png"))

image_2 = canvas.create_image(

    1395.0,

    769.0,

    image=image_image_2

)

window.iconphoto(True, image_image_2)

window.resizable(False, False)

window.mainloop()

