import os
import kagglehub

def main():
    os.environ["KAGGLEHUB_CACHE"] = "data"
    path = kagglehub.dataset_download("andradaolteanu/gtzan-dataset-music-genre-classification")
    print("Dataset downloaded to: ", path)

if __name__ == "__main__":
    main()