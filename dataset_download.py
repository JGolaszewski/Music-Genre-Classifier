import os
import kagglehub
import librosa
import librosa.feature
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path

def wav_to_features(wav_path, n_samples=5):
    # Wczytanie pliku audio
    y, sr = sf.read(wav_path, always_2d=True)  # Gwarantuje dwuwymiarową tablicę
    y = y.mean(axis=1) if y.ndim > 1 else y  # Konwersja stereo do mono (jeśli potrzebne)

    # Ekstrakcja ścieżki i gatunku
    genre = os.path.basename(os.path.dirname(wav_path))
    name = os.path.splitext(os.path.basename(wav_path))[0]

    # Wybór n punktów w czasie
    duration = librosa.get_duration(y=y, sr=sr)

    # Przygotowanie listy cech
    feature_names = []
    feature_values = []

    samples = np.array_split(y, n_samples)

    for t, sample in enumerate(samples):
        y_segment = sample
        # Cechy chroma (średnia z chroma)
        chroma = librosa.feature.chroma_stft(y=y_segment, sr=sr, n_fft=512)
        chroma_mean = np.mean(chroma, axis=1)  # Średnia z chroma dla każdego kanału
        chroma_diff = np.diff(chroma, axis=1)  # Różnice chroma
        chroma_std = np.std(chroma, axis=1)  # Odchylenie standardowe chroma

        # Tonacja i ton podstawowy (średnia z tonnetz)
        tonnetz = librosa.feature.tonnetz(y=y_segment, sr=sr)
        tonnetz_mean = np.mean(tonnetz, axis=1)
        tonnetz_std = np.std(tonnetz, axis=1)
        tonnetz_diff = np.diff(tonnetz, axis=1)

        # Tempogram (średnia z tempogramu)
        tempogram = librosa.feature.tempogram(y=y_segment, sr=sr)
        tempogram_mean = np.mean(tempogram)  # Jedna wartość średnia dla tempogramu

        # Zmiana tempa w czasie
        tempo_changes = np.diff(tempogram, axis=1)  # Różnice w tempogramie
        tempo_change_mean = np.mean(tempo_changes)
        tempo_change_std = np.std(tempo_changes)

        # MFCC (średnia z MFCC)
        mfcc = librosa.feature.mfcc(y=y_segment, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        mfcc_diff = np.diff(mfcc, axis=1)

        # Dodawanie cech do listy
        feature_names += (
            [f'chroma_diff_t{int(t)}'] +
            [f'chroma_mean_t{int(t)}'] +
            [f'chroma_std_t{int(t)}'] +
            [f'tonnetz_diff_t{int(t)}'] +
            [f'tonnetz_mean_t{int(t)}'] +
            [f'tonnetz_std_t{int(t)}'] +
            [f'mfcc_diff_t{int(t)}'] +
            [f'mfcc_mean_t{int(t)}'] +
            [f'mfcc_std_t{int(t)}'] +
            [f'tempo_mean_t{int(t)}'] +
            [f'tempo_diff_t{int(t)}'] +
            [f'tempo_diff_mean_t{int(t)}'] +
            [f'tempo_diff_std_t{int(t)}']
        )

        feature_values += (
            [chroma_diff.mean()] +
            [chroma_mean.mean()] +
            [chroma_std.mean()] +
            [tonnetz_diff.mean()] +
            [tonnetz_mean.mean()] +
            [tonnetz_std.mean()] +
            [mfcc_diff.mean()] +
            [mfcc_mean.mean()] +
            [mfcc_std.mean()] +
            [tempogram_mean] +
            [np.mean(tempo_changes)] +
            [tempo_change_mean] +
            [tempo_change_std]
        )

    # Tworzenie DataFrame
    df = pd.DataFrame([feature_values], columns=feature_names)
    df['genre'] = genre
    df['name'] = name

    return df

def process_directory(directory_path, n_samples=5):
    all_features = []

    # Rekursywne przejście przez wszystkie pliki w folderze
    for wav_file in Path(directory_path).rglob('*.wav'):
        print(f"Processing {wav_file}")
        features_df = wav_to_features(wav_file, n_samples)
        all_features.append(features_df)

    # Łączenie wszystkich DataFrame w jeden
    final_df = pd.concat(all_features, ignore_index=True)

    # Zapis do CSV
    output_csv = os.path.join(directory_path, 'all_features.csv')
    final_df.to_csv(output_csv, index=False)

    return output_csv

def main():
    os.environ["KAGGLEHUB_CACHE"] = "data"
    path = kagglehub.dataset_download("andradaolteanu/gtzan-dataset-music-genre-classification")

    corrupted_wav = "data/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/versions/1/Data/genres_original/jazz/jazz.00054.wav"
    corrupted_wav_spec = "data/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/versions/1/Data/images_original/jazz/jazz.00054.png"

    if os.path.exists(corrupted_wav):
        os.remove(corrupted_wav)
    if os.path.exists(corrupted_wav_spec):
        os.remove(corrupted_wav_spec)

    print("Dataset downloaded to: ", path)
    process_directory("data/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/versions/1/Data/genres_original", n_samples=30)

if __name__ == "__main__":
    main()