import librosa
import matplotlib.pyplot as plt
import numpy as np
import pyrubberband
import scipy.signal
from IPython.display import Audio, display

from .autotune import *


class Signal:

    FE = 44100
    y = None

    def __init__(self, fichier=None, y=None):
        self.y = None  # Initialisation claire avant toute logique
        self.t = None
        if fichier:
            self._load_audio(fichier)
        elif y is not None:  # "is not None" plus explicite que "not y is None"
            self._set_signal(y)

    def _load_audio(self, fichier):
        self.y, _ = librosa.load(fichier, sr=self.FE)
        self._update_time_axis()

    def _set_signal(self, y):
        self.y = y
        self._update_time_axis()

    def _update_time_axis(self):
        self.t = np.arange(len(self.y)) / self.FE

    def afficher(self):
        if self.y is None:
            self.generer()
        fig, ax = plt.subplots()
        ax.plot(self.t, self.y)
        ax.set_xlabel("Temps (s)")
        ax.set_ylabel("Amplitude")
        # fig.show()

    def afficher_spectrogramme(self):
        fig, ax = plt.subplots()
        D = librosa.amplitude_to_db(np.abs(librosa.stft(self.y)), ref=np.max)
        librosa.display.specshow(D, y_axis="log", sr=self.FE, x_axis="time", ax=ax)
        # fig.show()

    def jouer(self, de=None, a=None):
        if de is not None:
            y = self.y[de * self.FE : a * self.FE]
        else:
            y = self.y
        return display(Audio(y, rate=self.FE))

    def telecharger(self, nom="test.wav"):
        scipy.io.wavfile.write(nom, self.FE, self.y)

    def passe_haut(self, frequence):
        sos = scipy.signal.butter(100, frequence, "highpass", fs=self.FE, output="sos")
        self.y = scipy.signal.sosfiltfilt(sos, self.y)
        return self

    def passe_bas(self, frequence):
        sos = scipy.signal.butter(100, frequence, "lowpass", fs=self.FE, output="sos")
        self.y = scipy.signal.sosfiltfilt(sos, self.y)
        return self

    def transposer(self, demi_tons):
        # self.y = librosa.effects.pitch_shift(self.y, sr=self.FE, n_steps=demi_tons)
        self.y = pyrubberband.pyrb.pitch_shift(self.y, sr=self.FE, n_steps=demi_tons)
        return self

    def etirer(self, pourcentage):
        # self.y = librosa.effects.time_stretch(self.y, rate=pourcentage)
        self.y = pyrubberband.pyrb.time_stretch(self.y, sr=self.FE, rate=pourcentage)
        return self

    def accelerer(self, pourcentage):
        self.y = librosa.resample(
            self.y, orig_sr=self.FE, target_sr=int(self.FE / pourcentage)
        )
        return self

    def sampler(self, debut, fin):
        # return Signal(y=self.y[round(debut * self.FE) : round(fin * self.FE)])
        return Sample(
            parent=self, debut=round(debut * self.FE), fin=round(fin * self.FE)
        )

    def autotuner(self, gamme=None):
        correction_function = (
            closest_pitch
            if gamme is None
            else partial(aclosest_pitch_from_scale, scale=gamme)
        )

        pitch_corrected_y = autotune(self.y, self.FE, correction_function)

        self.y = pitch_corrected_y

    def __add__(self, other):
        return Signal(y=self.y + other.y)


class Sample(Signal):

    def __init__(self, parent, debut, fin):
        self.parent = parent
        self.debut = debut
        self.fin = fin
        self.y = parent.y[debut:fin]
        self.t = np.arange(len(self.y)) / self.FE


class Sinus(Signal):

    def __init__(self, amplitude, frequence, duree=2):
        super().__init__()
        self.amplitude = amplitude
        self.frequence = frequence
        self.generer(duree)

    def generer(self, duree=2):
        self.t = np.linspace(0, duree, int(self.FE * duree))
        self.y = self.amplitude * np.sin(2 * np.pi * self.frequence * self.t)


class Triangle(Signal):

    def __init__(self, amplitude, frequence, duree=2):
        super().__init__()
        self.amplitude = amplitude
        self.frequence = frequence
        self.generer(duree)

    def generer(self, duree=2):
        self.t = np.linspace(0, duree, int(self.FE * duree))
        self.y = self.amplitude * scipy.signal.sawtooth(
            2 * np.pi * self.frequence * self.t
        )


class Carre(Signal):

    def __init__(self, amplitude, frequence, duree=2):
        super().__init__()
        self.amplitude = amplitude
        self.frequence = frequence
        self.generer(duree)

    def generer(self, duree=2):
        self.t = np.linspace(0, duree, int(self.FE * duree))
        self.y = self.amplitude * scipy.signal.square(
            2 * np.pi * self.frequence * self.t
        )


# class Accord:
#     NOTES = {
#         "do2": 65.41,
#         "do#2": 69.30,
#         "ré2": 73.42,
#         "mib2": 77.78,
#         "mi2": 82.41,
#         "fa2": 87.31,
#         "fa#2": 92.50,
#         "sol2": 98.00,
#         "sol#2": 103.80,
#         "la2": 110.00,
#         "sib2": 116.50,
#         "si2": 123.50,
#         "do3": 130.80,
#         "do#3": 138.60,
#         "ré3": 146.80,
#         "mib3": 155.60,
#         "mi3": 164.80,
#         "fa3": 174.60,
#         "fa#3": 185.00,
#         "sol3": 196.00,
#         "sol#3": 207.70,
#         "la3": 220.00,
#         "sib3": 233.10,
#         "si3": 246.90,
#         "do4": 261.60,
#         "do#4": 277.20,
#         "ré4": 293.70,
#         "mib4": 311.10,
#         "mi4": 329.60,
#         "fa4": 349.20,
#         "fa#4": 370.00,
#         "sol4": 392.00,
#         "sol#4": 415.30,
#         "la4": 440.00,
#         "sib4": 466.20,
#         "si4": 493.90,
#         "do5": 523.30,
#         "do#5": 554.40,
#         "ré5": 587.30,
#         "mib5": 622.30,
#         "mi5": 659.30,
#         "fa5": 698.50,
#         "fa#5": 740.00,
#         "sol5": 784.00,


#     def __add__(self, other):
#         return Signal(y=self.y + other.y)


# class Sample(Signal):

#     def __init__(self, parent, debut, fin):
#         self.parent = parent
#         self.debut = debut
#         self.fin = fin
#         self.y = parent.y[debut:fin]
#         self.t = np.arange(len(self.y)) / self.FE


# class Sinus(Signal):

#     def __init__(self, amplitude, frequence, duree=2):
#         super().__init__()
#         self.amplitude = amplitude
#         self.frequence = frequence
#         self.generer(duree)

#     def generer(self, duree=2):
#         self.t = np.linspace(0, duree, int(self.FE * duree))
#         self.y = self.amplitude * np.sin(2 * np.pi * self.frequence * self.t)


# class Accord:
#     NOTES = {
#         "do2": 65.41,
#         "do#2": 69.30,
#         "ré2": 73.42,
#         "mib2": 77.78,
#         "mi2": 82.41,
#         "fa2": 87.31,
#         "fa#2": 92.50,
#         "sol2": 98.00,
#         "sol#2": 103.80,
#         "la2": 110.00,
#         "sib2": 116.50,
#         "si2": 123.50,
#         "do3": 130.80,
#         "do#3": 138.60,
#         "ré3": 146.80,
#         "mib3": 155.60,
#         "mi3": 164.80,
#         "fa3": 174.60,
#         "fa#3": 185.00,
#         "sol3": 196.00,
#         "sol#3": 207.70,
#         "la3": 220.00,
#         "sib3": 233.10,
#         "si3": 246.90,
#         "do4": 261.60,
#         "do#4": 277.20,
#         "ré4": 293.70,
#         "mib4": 311.10,
#         "mi4": 329.60,
#         "fa4": 349.20,
#         "fa#4": 370.00,
#         "sol4": 392.00,
#         "sol#4": 415.30,
#         "la4": 440.00,
#         "sib4": 466.20,
#         "si4": 493.90,
#         "do5": 523.30,
#         "do#5": 554.40,
#         "ré5": 587.30,
#         "mib5": 622.30,
#         "mi5": 659.30,
#         "fa5": 698.50,
#         "fa#5": 740.00,
#         "sol5": 784.00,


#     def __add__(self, other):
#         return Signal(y=self.y + other.y)


# class Sample(Signal):

#     def __init__(self, parent, debut, fin):
#         self.parent = parent
#         self.debut = debut
#         self.fin = fin
#         self.y = parent.y[debut:fin]
#         self.t = np.arange(len(self.y)) / self.FE


# class Sinus(Signal):

#     def __init__(self, amplitude, frequence, duree=2):
#         super().__init__()
#         self.amplitude = amplitude
#         self.frequence = frequence
#         self.generer(duree)

#     def generer(self, duree=2):
#         self.t = np.linspace(0, duree, int(self.FE * duree))
#         self.y = self.amplitude * np.sin(2 * np.pi * self.frequence * self.t)


# class Accord:
#     NOTES = {
#         "do2": 65.41,
#         "do#2": 69.30,
#         "ré2": 73.42,
#         "mib2": 77.78,
#         "mi2": 82.41,
#         "fa2": 87.31,
#         "fa#2": 92.50,
#         "sol2": 98.00,
#         "sol#2": 103.80,
#         "la2": 110.00,
#         "sib2": 116.50,
#         "si2": 123.50,
#         "do3": 130.80,
#         "do#3": 138.60,
#         "ré3": 146.80,
#         "mib3": 155.60,
#         "mi3": 164.80,
#         "fa3": 174.60,
#         "fa#3": 185.00,
#         "sol3": 196.00,
#         "sol#3": 207.70,
#         "la3": 220.00,
#         "sib3": 233.10,
#         "si3": 246.90,
#         "do4": 261.60,
#         "do#4": 277.20,
#         "ré4": 293.70,
#         "mib4": 311.10,
#         "mi4": 329.60,
#         "fa4": 349.20,
#         "fa#4": 370.00,
#         "sol4": 392.00,
#         "sol#4": 415.30,
#         "la4": 440.00,
#         "sib4": 466.20,
#         "si4": 493.90,
#         "do5": 523.30,
#         "do#5": 554.40,
#         "ré5": 587.30,
#         "mib5": 622.30,
#         "mi5": 659.30,
#         "fa5": 698.50,
#         "fa#5": 740.00,
#         "sol5": 784.00,


#     def __add__(self, other):
#         return Signal(y=self.y + other.y)


# class Sample(Signal):

#     def __init__(self, parent, debut, fin):
#         self.parent = parent
#         self.debut = debut
#         self.fin = fin
#         self.y = parent.y[debut:fin]
#         self.t = np.arange(len(self.y)) / self.FE


# class Sinus(Signal):

#     def __init__(self, amplitude, frequence, duree=2):
#         super().__init__()
#         self.amplitude = amplitude
#         self.frequence = frequence
#         self.generer(duree)

#     def generer(self, duree=2):
#         self.t = np.linspace(0, duree, int(self.FE * duree))
#         self.y = self.amplitude * np.sin(2 * np.pi * self.frequence * self.t)


class Accord:
    NOTES = {
        "do2": 65.41,
        "do#2": 69.30,
        "ré2": 73.42,
        "mib2": 77.78,
        "mi2": 82.41,
        "fa2": 87.31,
        "fa#2": 92.50,
        "sol2": 98.00,
        "sol#2": 103.80,
        "la2": 110.00,
        "sib2": 116.50,
        "si2": 123.50,
        "do3": 130.80,
        "do#3": 138.60,
        "ré3": 146.80,
        "mib3": 155.60,
        "mi3": 164.80,
        "fa3": 174.60,
        "fa#3": 185.00,
        "sol3": 196.00,
        "sol#3": 207.70,
        "la3": 220.00,
        "sib3": 233.10,
        "si3": 246.90,
        "do4": 261.60,
        "do#4": 277.20,
        "ré4": 293.70,
        "mib4": 311.10,
        "mi4": 329.60,
        "fa4": 349.20,
        "fa#4": 370.00,
        "sol4": 392.00,
        "sol#4": 415.30,
        "la4": 440.00,
        "sib4": 466.20,
        "si4": 493.90,
        "do5": 523.30,
        "do#5": 554.40,
        "ré5": 587.30,
        "mib5": 622.30,
        "mi5": 659.30,
        "fa5": 698.50,
        "fa#5": 740.00,
        "sol5": 784.00,
        "sol#5": 830.60,
        "la5": 880.00,
        "sib5": 932.30,
        "si5": 987.80,
        "do6": 1047.00,
        "do#6": 1109.0,
        "ré6": 1175.0,
        "mib6": 1245.0,
        "mi6": 1319.0,
        "fa6": 1397.0,
        "fa#6": 1480.0,
        "sol6": 1568.0,
        "sol#6": 1661.0,
        "la6": 1760.0,
        "sib6": 1865.0,
        "si6": 1976.0,
        "do7": 2093.0,
    }

    TYPES = {
        "maj": {
            "intervalles": [0, 4, 7],
            "name": "Majeur",
        },  # fondamentale, tierce majeure, quinte juste
        "min": {
            "intervalles": [0, 3, 7],
            "name": "Mineur",
        },  # fondamentale, tierce mineure, quinte juste
        "dim": {
            "intervalles": [0, 3, 6],
            "name": "Diminué",
        },  # fondamentale, tierce mineure, quinte diminuée
    }

    def __init__(self, type=None, note=None, notes=None):
        self.type = type
        self.note = note

        if notes is not None:
            self.notes = notes

    def _calculer_frequences(self):
        if self.type is not None:
            fondamentale = self.NOTES[self.note.lower()]
            return [
                fondamentale * (2 ** (i / 12))
                for i in self.TYPES[self.type]["intervalles"]
            ]

        if self.notes is not None:
            return [self.NOTES[note] for note in self.notes]


class Synthetiseur:
    def __init__(self, forme_ondes):
        self.forme_ondes = forme_ondes
        self.harmoniques = []

    def ajouter_harmonique(self, rang, amplitude):
        self.harmoniques.append((rang, amplitude))
        return self

    def generer(self, accord, duree=2):
        signal_final = None
        for frequence in accord._calculer_frequences():
            note = self._generer_note(frequence, duree)
            signal_final = note if signal_final is None else signal_final + note
        return signal_final

    def jouer(self, accord):
        signal_final = self.generer(accord)
        return signal_final.jouer()

    def _generer_note(self, frequence, duree):
        signal = None
        for rang, amplitude in self.harmoniques:
            onde = self.forme_ondes(amplitude, frequence * rang, duree)
            signal = onde if signal is None else signal + onde
        return signal


class Partition:
    def __init__(self, tempo=60):
        self.tempo = tempo
        self.elements = []  # Stocke des tuples (signal, debut, fin)

    def ajouter(self, signal, debut):
        """Ajoute un signal à la partition.
        Args:
            signal (Signal): Signal à ajouter
            debut (float): Temps de début en secondes
        """
        duree = len(signal.y) / signal.FE
        debut = debut * 60 / self.tempo
        self.elements.append((signal, debut, debut + duree))
        return self

    def jouer(self):
        """Joue la partition mixée"""
        duree_max = max([fin for _, _, fin in self.elements])
        temps = np.arange(int(duree_max * Signal.FE)) / Signal.FE
        mix = np.zeros_like(temps)

        for signal, debut, fin in self.elements:
            start_idx = int(debut * Signal.FE)
            end_idx = start_idx + len(signal.y)
            mix[start_idx:end_idx] += signal.y

        return display(Audio(mix, rate=Signal.FE))

    def afficher(self):
        """Affiche la ligne temporelle des éléments"""

        # Récupérer tous les morceaux dont sont issus les samples
        parents = set([element[0].parent for element in self.elements])
        n = len(parents)

        # Créer les axes
        fig, axs = plt.subplots(len(parents) + 1, 1)
        for i, parent in enumerate(parents):
            axs[i].plot(parent.t, parent.y)

            samples = set(
                [element[0] for element in self.elements if element[0].parent == parent]
            )
            t_min = np.min([sample.debut / sample.FE for sample in samples])
            t_max = np.max([sample.fin / sample.FE for sample in samples])
            axs[i].set_xlim([t_min, t_max])

            for sample in samples:
                sample.color = list(np.random.choice(range(256), size=3) / 255)
                sample.parent_id = i
                axs[i].axvspan(
                    sample.debut / sample.FE,
                    sample.fin / sample.FE,
                    color=sample.color,
                    alpha=0.5,
                )

        for element in self.elements:
            sample, debut, fin = element

            y_max = 1 - sample.parent_id / n
            y_min = 1 - (sample.parent_id + 1) / n
            axs[-1].axvspan(
                debut,
                fin,
                ymin=y_min,
                ymax=y_max,
                color=sample.color,
                alpha=0.5,
            )

        # plt.figure(figsize=(10, 2))
        # for i, (_, debut, fin) in enumerate(self.elements):
        #     plt.plot([debut, fin], [i, i], linewidth=5)
        # plt.yticks([])
        # plt.xlabel("Temps (s)")
        # plt.title(f"Partition (Tempo: {self.tempo} BPM)")
