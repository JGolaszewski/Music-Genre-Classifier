from model import *

def main():
    # Inicjalizacja modeli i optymalizatorów
    cnn_model = SpectCNN(num_classes=10)
    rnn_model = TemporalRNN(input_size=42, hidden_size=128, num_layers=2, rnn_output_size=128)
    fusion_model = FusionModel(input_size=138, hidden_size=128, num_classes=10)

    # Ładowanie danych dla każdego modelu
    cnn_model.load_data(
        spect_dir=r'data\datasets\andradaolteanu\gtzan-dataset-music-genre-classification\versions\1\Data\images_original',
        batch_size=32)
    rnn_model.load_data(
        csv_file=r'data\datasets\andradaolteanu\gtzan-dataset-music-genre-classification\versions\1\Data\all_features.csv',
        batch_size=32)

    # Kryterium i optymalizatory
    criterion = nn.CrossEntropyLoss()
    optimizer_cnn = torch.optim.SGD(cnn_model.parameters(), lr=0.0001, momentum=0.9)
    optimizer_rnn = torch.optim.SGD(rnn_model.parameters(), lr=0.0001, momentum=0.9)
    optimizer_fusion = torch.optim.SGD(filter(lambda p: p.requires_grad, fusion_model.parameters()), lr=0.001)

    # Trenowanie indywidualnych modeli
    train_model(cnn_model, cnn_model.train_loader, cnn_model.test_loader, criterion, optimizer_cnn, num_epochs=5)
    # train_model(rnn_model, rnn_model.train_loader, rnn_model.test_loader, criterion, optimizer_rnn, num_epochs=200)

    # Testowanie indywidualnych modeli
    print("\nTesting CNN model\n")
    test_model(cnn_model, cnn_model.test_loader, criterion)
    print("\nTesting RNN model\n")
    test_model(rnn_model, rnn_model.test_loader, criterion)


    return

    # Trenowanie modelu fusion
    train_fusion_model(fusion_model, cnn_model, rnn_model, cnn_model.train_loader, rnn_model.train_loader, criterion,
                       optimizer_fusion, num_epochs=5)

    # Testowanie modelu fusion
    print("\nTesting Fusion model\n")
    test_fusion_model(fusion_model, cnn_model, rnn_model, cnn_model.test_loader, rnn_model.test_loader, criterion)


if __name__ == "__main__":
    main()