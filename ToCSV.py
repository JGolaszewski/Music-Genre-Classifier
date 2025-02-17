import os
import numpy as np
import librosa
import pandas as pd
from scipy.stats import kurtosis


def main():
    corrupted_wav = "data/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/versions/1/Data/genres_original/jazz/jazz.00054.wav"
    corrupted_wav_spec = "data/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/versions/1/Data/images_original/jazz/jazz.00054.png"

    if os.path.exists(corrupted_wav):
        os.remove(corrupted_wav)
    if os.path.exists(corrupted_wav_spec):
        os.remove(corrupted_wav_spec)

    class MelFeatureExtractor:
        def __init__(self, sr=22050, num_parts=10):
            self.sr = sr
            self.num_parts = num_parts

        def extract_features(self, file_path):
            """
            Ekstrahuje cechy z pliku audio:
            - Chromagram
            - Tempo (BPM)
            - MFCC
            - F0 (ton podstawowy)
            - Beat tracking
            - Tempogram
            - Zmiana tempa w czasie
            - Spectral Centroid
            - Spectral Flatness
            - Zero Crossing Rate (ZCR)
            - RMS
            - Harmonic-to-Noise Ratio (HNR)
            - Spectral Rolloff
            - Standard Deviation (SD) dla różnych cech
            - Kurtosis (wypukłość) dla różnych cech
            """


            # Wczytaj plik audio
            y, sr = librosa.load(file_path, sr=self.sr)

            # Podziel na części (jeśli potrzebne)
            y_parts = np.array_split(y, self.num_parts)

            features = []

            epsilon = 1e-10  # Small constant to prevent catastrophic cancellation

            for part in y_parts:
                # 1. Chromagram
                chroma = librosa.feature.chroma_stft(y=part, sr=sr, n_fft=min(len(part), 1024))
                chroma_mean = np.mean(chroma, axis=1)  # Średnie wartości dla każdego z 12 półtonów
                chroma_std = np.std(chroma, axis=1)    # Standard deviation for chroma
                chroma_kurtosis = kurtosis(chroma+ epsilon, axis=1, fisher=True)  # Kurtosis for chroma
                features.extend(chroma_mean)
                features.extend(chroma_std)
                features.extend(chroma_kurtosis)

                # 2. Tonacja i ton podstawowy (średnia z tonnetz)
                tonnetz = librosa.feature.tonnetz(y=part, sr=sr)
                tonnetz_mean = np.mean(tonnetz, axis=1)  # Średnia wartości dla tonnetz
                tonnetz_std = np.std(tonnetz, axis=1)    # Standard deviation for tonnetz
                tonnetz_kurtosis = kurtosis(tonnetz+ epsilon, axis=1, fisher=True)  # Kurtosis for tonnetz
                features.extend(tonnetz_mean)
                features.extend(tonnetz_std)
                features.extend(tonnetz_kurtosis)

                # 3. Harmoniczno-perkusyjne rozdzielenie sygnału
                y_harmonic, y_percussive = librosa.effects.hpss(part)

                # 4. Tempo i beat tracking (jedna wartość dla tempa)
                tempo, beats = librosa.beat.beat_track(y=y_percussive, sr=sr)
                tempo = float(tempo)  # Upewniamy się, że tempo jest skalarem
                features.append(tempo)

                # 5. Tempogram (średnia z tempogramu)
                tempogram = librosa.feature.tempogram(y=y_percussive, sr=sr)
                tempogram_mean = np.mean(tempogram, axis=1)  # Jedna wartość średnia dla tempogramu
                tempogram_std = np.std(tempogram, axis=1)    # Standard deviation for tempogram
                tempogram_kurtosis = kurtosis(tempogram+ epsilon, axis=1, fisher=True)  # Kurtosis for tempogram
                features.extend(tempogram_mean)
                features.extend(tempogram_std)
                features.extend(tempogram_kurtosis)

                # 6. Zmiana tempa w czasie
                tempo_changes = np.diff(tempogram, axis=1)  # Różnice w tempogramie
                tempo_change_mean = np.mean(tempo_changes) if len(tempo_changes) > 0 else 0
                tempo_change_std = np.std(tempo_changes) if len(tempo_changes) > 0 else 0
                features.append(tempo_change_mean)
                features.append(tempo_change_std)

                # 7. MFCC (średnia z MFCC)
                mfcc = librosa.feature.mfcc(y=part, sr=sr, n_mfcc=13)
                mfcc_mean = np.mean(mfcc, axis=1)  # Średnia wartości dla każdego MFCC
                mfcc_std = np.std(mfcc, axis=1)    # Standard deviation for MFCC
                mfcc_kurtosis = kurtosis(mfcc+ epsilon, axis=1, fisher=True)  # Kurtosis for MFCC
                features.extend(mfcc_mean)
                features.extend(mfcc_std)
                features.extend(mfcc_kurtosis)

                # 8. Spectral Centroid
                spectral_centroid = librosa.feature.spectral_centroid(y=part, sr=sr)
                spectral_centroid_mean = np.mean(spectral_centroid)  # Średnia wartość
                spectral_centroid_std = np.std(spectral_centroid)    # Standard deviation for centroid
                spectral_centroid_kurtosis = kurtosis(spectral_centroid+ epsilon, fisher=True)  # Kurtosis for centroid
                features.append(spectral_centroid_mean)
                features.append(spectral_centroid_std)
                features.append(spectral_centroid_kurtosis)

                # 9. Spectral Flatness
                spectral_flatness = librosa.feature.spectral_flatness(y=part)
                features.append(np.mean(spectral_flatness))  # Średnia wartość

                # 10. Zero Crossing Rate (ZCR)
                zcr = librosa.feature.zero_crossing_rate(y=part)
                features.append(np.mean(zcr))  # Średnia wartość

                # 11. RMS (Root Mean Square Energy)
                rms = librosa.feature.rms(y=part)
                features.append(np.mean(rms))  # Średnia wartość

                # 12. Harmonic-to-Noise Ratio (HNR)
                hnr = librosa.effects.harmonic(y=part)
                hnr_mean = np.mean(hnr)
                features.append(hnr_mean)  # Średnia wartość

                # 13. Spectral Rolloff
                spectral_rolloff = librosa.feature.spectral_rolloff(y=part, sr=sr, roll_percent=0.85)
                features.append(np.mean(spectral_rolloff))  # Średnia wartość

            return features

    def extract_features_to_csv(root_dir, output_csv, num_parts=10):
        """
        Wczytuje pliki audio z folderów, ekstrahuje cechy i zapisuje je do pliku CSV.
        - root_dir: Folder główny z podfolderami, gdzie każdy podfolder reprezentuje klasę.
        - output_csv: Ścieżka do pliku CSV, do którego zostaną zapisane cechy.
        """
        extractor = MelFeatureExtractor(num_parts=num_parts)
        data = []
        labels = []

        # Iteracja po podfolderach
        for label, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for file_name in os.listdir(class_dir):
                    file_path = os.path.join(class_dir, file_name)
                    if file_path.endswith('.wav'):  # Obsługujemy tylko pliki .wav
                        print(f"Przetwarzanie: {file_path}")
                        try:
                            # Ekstrakcja cech
                            features = extractor.extract_features(file_path)
                            data.append(features)
                            labels.append(class_name)
                        except Exception as e:
                            print(f"Błąd podczas przetwarzania {file_path}: {e}")

        # Tworzenie DataFrame z odpowiednią liczbą kolumn
        if data:
            num_features = len(data[0])  # Określenie liczby cech na podstawie pierwszego przetworzonego pliku
            feature_columns = [f"feature_{i}" for i in range(num_features)]
        else:
            feature_columns = []  # Jeśli nie ma danych, nie twórz kolumn

        df = pd.DataFrame(data, columns=feature_columns)
        df["label"] = labels  # Dodanie etykiety klasy

        # Zapis do pliku CSV
        df.to_csv(output_csv, index=False)
        print(f"Cechy zostały zapisane do {output_csv}")

    # Ścieżka do folderu z plikami audio i ścieżka do pliku CSV
    dataset_dir = r'C:\Users\Komputer\Desktop\studia-Informatyka\Koła\Golem\projekt-rekrutacja\Music-Genre-Classifier\data\datasets\andradaolteanu\gtzan-dataset-music-genre-classification\versions\1\Data\genres_original'
    output_csv = r'C:\Users\Komputer\Desktop\studia-Informatyka\Koła\Golem\projekt-rekrutacja\Music-Genre-Classifier\data\datasets\andradaolteanu\gtzan-dataset-music-genre-classification\versions\1\Data\features.csv'  # Podmień na ścieżkę wyjściową

    # Wywołanie funkcji do ekstrakcji cech i zapisu do CSV
    extract_features_to_csv(dataset_dir, output_csv)


if __name__ == "__main__":
    main()
