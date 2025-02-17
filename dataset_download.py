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

        # CHROMA
        chroma = librosa.feature.chroma_stft(y=y_segment, sr=sr, n_fft=512, n_chroma=12).mean(axis=1)

        feature_names.extend([f'chroma_{i+1}_t{t}' for i in range(12)])
        feature_values.extend(chroma)

        # TONNETZ
        tonnetz = librosa.feature.tonnetz(y=y_segment, sr=sr).mean(axis=1)
        feature_names.extend([f'tonnetz_{i+1}_t{t}' for i in range(6)])
        feature_values.extend(tonnetz)

        # BPS
        tempo, _ = librosa.beat.beat_track(y=y_segment, sr=sr)
        feature_names.append(f'tempo_t{t}')
        feature_values.append(tempo)

        # RMS
        rms = librosa.feature.rms(y=y_segment)
        mean_rms = np.mean(rms)
        max_rms = np.max(rms)
        min_rms = np.min(rms)
        std_rms = np.std(rms)
        feature_names.extend([f'mean_rms_t{t}', f'max_rms_t{t}', f'min_rms_t{t}',
                              f'std_rms_t{t}'])
        feature_values.extend([mean_rms, max_rms, min_rms, std_rms])

        #ROLL-OFF FREQUENCY
        rolloff = librosa.feature.spectral_rolloff(y=y_segment, sr=sr)
        mean_rolloff = np.mean(rolloff)
        max_rolloff = np.max(rolloff)
        min_rolloff = np.min(rolloff)
        std_rolloff = np.std(rolloff)
        feature_names.extend([f'mean_rolloff_t{t}', f'max_rolloff_t{t}', f'min_rolloff_t{t}',
                              f'std_rolloff_t{t}'])
        feature_values.extend([mean_rolloff, max_rolloff, min_rolloff, std_rolloff])

        # ZERO CROSSING RATE
        zcr = librosa.feature.zero_crossing_rate(y_segment)
        mean_zcr = np.mean(zcr)
        max_zcr = np.max(zcr)
        min_zcr = np.min(zcr)
        std_zcr = np.std(zcr)
        feature_names.extend([f'mean_zcr_t{t}', f'max_zcr_t{t}', f'min_zcr_t{t}',
                              f'std_zcr_t{t}'])
        feature_values.extend([mean_zcr, max_zcr, min_zcr, std_zcr])

        # MFCC
        mfcc = librosa.feature.mfcc(y=y_segment, sr=sr, n_mfcc=11).mean(axis=1)
        feature_names.extend([f'mfcc_{i+1}_t{t}' for i in range(11)])
        feature_values.extend(mfcc)

    # Tworzenie DataFrame
    df = pd.DataFrame([feature_values], columns=feature_names)
    df['genre'] = genre
    df['name'] = name
    return df

def process_directory(directory_path, n_samples=5):
    all_features = []

    for wav_file in Path(directory_path).rglob('*.wav'):
        print(f"Processing {wav_file}")
        features_df = wav_to_features(wav_file, n_samples)
        all_features.append(features_df)

    final_df = pd.concat(all_features, ignore_index=True)

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