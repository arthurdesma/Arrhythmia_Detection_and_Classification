import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class GUI3:
    def __init__(self, master, values):
        self.master = master
        self.values = values
        self.start_ui()
        self.setup_ui()

    def start_ui(self):
        self.master.geometry("1440x818")
        self.master.configure(bg="#CF5E92")
        
        

    def setup_ui(self):
        # Affiche les valeurs reçues
        label_values = tk.Label(self.master, text="Analyse complète de l'ECG", font=("Doppio One", 27), bg='#CF5E92', fg="white")
        label_values.pack(pady=5)

        # Ajoute un texte descriptif avec une taille de police plus petite
        description_label = tk.Label(self.master, text="Cette représentation indique la répartition des anomalies, 1 point pour 5 secondes", font=("Arial", 15), bg='#CF5E92',fg="white")
        description_label.pack(pady=1)

        # Crée un cadre pour contenir la figure Matplotlib
        plot_frame = tk.Frame(self.master)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Crée une figure et des axes
        fig = Figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot(111)

        # Convertit les valeurs en minutes (1 point = 5 min)
        minutes = [i * 5 for i in range(len(self.values))]

        # Prépare les couleurs en fonction des valeurs des points
        point_colors = ['red' if x == -1 else 'gray' if x == 0 else 'blue' for x in self.values]

        # Trace les points avec des couleurs spécifiques
        ax.scatter(minutes, self.values, color=point_colors, marker='o')

        # Trace la ligne entre les points avec des couleurs dépendant de la position relative à zéro
        for i in range(len(minutes) - 1):
            segment_color = 'blue' if (self.values[i] + self.values[i+1]) / 2 > 0 else 'red'
            ax.plot(minutes[i:i+2], self.values[i:i+2], color=segment_color)

        # Active le quadrillage pour améliorer la lisibilité
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Définit les étiquettes et le titre
        ax.set_xlabel('Temps (secondes)')
        ax.set_ylabel('Valeurs')
        

        # Crée un canevas et ajoute la figure à celui-ci
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()

        # Emballe le canevas dans le cadre
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

