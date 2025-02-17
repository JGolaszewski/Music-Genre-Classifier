import torch
import datetime

from pyparsing import Combine

from model import SpectCNN, TemporalRNN, CombRNNCNN
from torchvision import datasets

def main():
    # Wykrywanie dostępnego urządzenia
    if torch.xpu.is_available():
        device = torch.device("xpu")  # Intel XPU
    elif torch.cuda.is_available():
        device = torch.device("cuda")  # NVIDIA GPU
    else:
        device = torch.device("cpu")  # CPU

    print(f"Device: {device}")

    spect_path = "data/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/versions/1/Data/images_original"
    csv_path = "data/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/versions/1/Data/genres_original/all_features.csv"

    combine = CombRNNCNN()
    combine.load_data(csv_path, spect_path)
    combine.train_whole()

if __name__ == "__main__":
    main()