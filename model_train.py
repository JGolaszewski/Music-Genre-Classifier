import torch
import datetime

from model import SpectCNN
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

    cnn = SpectCNN()
    cnn.load_data(img_folder_root="data/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/versions/1/Data/images_original",
                  batch_size=4, num_workers=2)

    print("Training...")
    cnn.data_train()
    cnn.data_test()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"save/model_{timestamp}.pth"
    torch.save(cnn.state_dict(), model_path)
    print(f"Model saved to '{model_path}'")

if __name__ == "__main__":
    main()