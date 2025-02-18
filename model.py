import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torchvision import transforms, datasets
from tqdm import tqdm
import pandas as pd
import numpy as np

def main():
    # Spectrogram Convolutional Neural Network Model
    class SpectCNN(nn.Module):
        def __init__(self, num_classes=10):
            super(SpectCNN, self).__init__()

            # Warstwy konwolucyjne
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # Wejście 432x288x3, wyjście 432x288x32
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # Zmniejszy wymiary o połowę

            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Zmniejszy wymiar
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

            # Warstwy w pełni połączone
            self.fc1 = nn.Linear(128 * 54 * 36, 512)  # Wymiary po konwolucjach i poolingach
            self.fc2 = nn.Linear(512, num_classes)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))  # Przechodzi przez warstwę konwolucyjną + pooling
            x = self.pool(F.relu(self.conv2(x)))  # Przechodzi przez drugą warstwę
            x = self.pool(F.relu(self.conv3(x)))  # Przechodzi przez trzecią warstwę

            x = x.view(-1, 128 * 54 * 36)  # Spłaszczenie przed warstwą FC
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

        def load_data(self, spect_dir, batch_size=4):
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

            # Load the dataset
            trainset = datasets.ImageFolder(root=spect_dir, transform=transform)
            self.classes = trainset.classes

            # Extract the labels from the dataset
            labels = [label for _, label in trainset]

            # Perform stratified split: 80% training, 20% testing
            train_indices, test_indices = train_test_split(
                np.arange(len(trainset)),  # Indices of all images
                test_size=0.2,  # 20% data for testing
                stratify=labels  # Stratify according to class distribution
            )

            # Create subsets for training and testing using the indices
            train_subset = torch.utils.data.Subset(trainset, train_indices)
            test_subset = torch.utils.data.Subset(trainset, test_indices)

            # DataLoaders
            self.train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                                                            num_workers=2)
            self.test_loader = torch.utils.data.DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=2)

    # TemporalRNN Model
    class TemporalRNN(nn.Module):
        def __init__(self, input_size=13, hidden_size=128, num_layers=2, rnn_output_size=128):
            super(TemporalRNN, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
            self.fc = nn.Linear(hidden_size, rnn_output_size)  # Zmniejszenie wymiaru do 128 cech

        def forward(self, x):
            # Inicjalizacja ukrytych stanów (z obsługą GPU)
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)

            # Przejście przez RNN
            out, _ = self.rnn(x, (h0, c0))

            # Wybierz ostatni ukryty stan i przekaż go przez warstwę w pełni połączoną
            out = self.fc(out[:, -1, :])  # Zwróć ostatnią próbkę w sekwencji
            return out

        def load_data(self, csv_file, batch_size=32):
            # Load CSV data
            df = pd.read_csv(csv_file)
            print(f"Rozmiar danych: {df.shape}")

            time_points = [str(i) for i in range(30)]
            labels = []
            sequences = []

            # Extract sequences and labels
            for index, row in df.iterrows():
                sequence = []
                for t in sorted(time_points):
                    cols = [col for col in df.columns if col.endswith(f'_t{t}')]
                    sequence.append(row[cols].values.tolist())

                labels.append(row["genre"])
                sequences.append(sequence)

            # Map labels to numeric values
            unique_labels = sorted(set(labels))
            label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
            numeric_labels = [label_mapping[label] for label in labels]

            # Convert sequences and labels to tensors
            sequences = np.array(sequences)
            sequences_tensor = torch.tensor(sequences, dtype=torch.float32)
            labels_tensor = torch.tensor(numeric_labels, dtype=torch.int64)

            # Stratified train-test split
            train_indices, test_indices = train_test_split(
                np.arange(len(sequences_tensor)),  # Indices of all sequences
                test_size=0.2,  # 20% data for testing
                stratify=numeric_labels  # Stratify based on the labels
            )

            # Create a Dataset class for PyTorch
            class TemporalDataset(torch.utils.data.Dataset):
                def __init__(self, sequences_in, labels_in):
                    self.m_sequences = sequences_in
                    self.m_labels = labels_in

                def __len__(self):
                    return len(self.m_sequences)

                def __getitem__(self, idx):
                    return self.m_sequences[idx], self.m_labels[idx]

            # Create datasets for training and testing
            train_subset = torch.utils.data.Subset(TemporalDataset(sequences_tensor, labels_tensor), train_indices)
            test_subset = torch.utils.data.Subset(TemporalDataset(sequences_tensor, labels_tensor), test_indices)

            # DataLoaders
            self.train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            self.test_loader = torch.utils.data.DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    # FusionModel
    class FusionModel(nn.Module):
        def __init__(self, input_size=138, hidden_size=128, num_classes=10):
            super(FusionModel, self).__init__()

            print(input_size, hidden_size, num_classes)
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, num_classes)
            self.dropout = nn.Dropout(0.3)

        def forward(self, cnn_output, rnn_output):
            # Łączenie wyjść z obu modeli
            fused_input = torch.cat((cnn_output, rnn_output), dim=1)  # dim=1 łączy po wymiarze cech

            # Przetwarzanie przez w pełni połączone warstwy
            x = F.relu(self.fc1(fused_input))
            x = self.dropout(x)
            x = self.fc2(x)

            return x

        def evaluate_model(self, cnn_model, rnn_model, cnn_loader, rnn_loader):
            self.eval()  # Ustawienie modelu na tryb ewaluacji

            all_preds = []
            all_labels = []

            with torch.no_grad():
                for (cnn_inputs, _), (rnn_inputs, labels) in zip(cnn_loader, rnn_loader):
                    cnn_output = cnn_model(cnn_inputs)
                    rnn_output = rnn_model(rnn_inputs)

                    outputs = self(cnn_output, rnn_output)
                    _, predicted = torch.max(outputs.data, 1)

                    all_preds.append(predicted)
                    all_labels.append(labels)

            # Łączenie wszystkich wyników
            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)

            # Obliczanie dokładności
            accuracy = accuracy_score(all_labels.numpy(), all_preds.numpy())
            print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Function to train the model
    def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10):
        model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_accuracy = 100 * correct / total
            print(f"Train Loss: {running_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%")

            # Validation step
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in tqdm(test_loader, desc="Validation"):
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_accuracy = 100 * correct / total
            print(f"Validation Accuracy: {val_accuracy:.2f}%")
            model.train()  # Switch back to training mode after validation

    # Function to train the fusion model
    def train_fusion_model(self, cnn_model, rnn_model, cnn_loader, rnn_loader, criterion, optimizer, num_epochs=10):
        # Włączenie trybu treningowego dla FusionModel
        self.train()
        cnn_model.eval()  # Ustawienie modelu CNN na tryb ewaluacji (brak trenowania)
        rnn_model.eval()  # Ustawienie modelu RNN na tryb ewaluacji (brak trenowania)

        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for (cnn_inputs, _), (rnn_inputs, labels) in zip(cnn_loader, rnn_loader):
                optimizer.zero_grad()

                # Uzyskiwanie wyników z modelu CNN i RNN
                cnn_output = cnn_model(cnn_inputs)
                rnn_output = rnn_model(rnn_inputs)

                # Przepuszczenie przez model Fusion
                outputs = self(cnn_output, rnn_output)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_accuracy = 100 * correct / total
            print(f"Train Loss: {running_loss / len(cnn_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%")

            # Kroki walidacyjne
            self.eval()  # Przełączamy model na tryb ewaluacji
            correct = 0
            total = 0
            with torch.no_grad():
                for (cnn_inputs, _), (rnn_inputs, labels) in zip(cnn_loader, rnn_loader):
                    cnn_output = cnn_model(cnn_inputs)
                    rnn_output = rnn_model(rnn_inputs)

                    outputs = self(cnn_output, rnn_output)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_accuracy = 100 * correct / total
            print(f"Validation Accuracy: {val_accuracy:.2f}%")
            self.train()  # Powrót do trybu treningowego po walidacji

        # Po zakończeniu treningu wykonaj ewaluację
        self.evaluate_model(cnn_model, rnn_model, cnn_loader, rnn_loader)





    def test_model(model, test_loader, criterion):
        model.eval()  # Przełączamy model na tryb ewaluacji

        correct = 0
        total = 0
        running_loss = 0.0
        all_preds = []
        all_labels = []

        # Słownik do przechowywania trafień dla każdego gatunku
        class_correct = {i: 0 for i in range(10)}  # Zakładamy, że mamy 10 gatunków
        class_total = {i: 0 for i in range(10)}

        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Testing"):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Obliczanie trafień dla każdej klasy
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    class_total[label] += 1
                    if predicted[i].item() == label:
                        class_correct[label] += 1

                all_preds.append(predicted)
                all_labels.append(labels)

        accuracy = 100 * correct / total
        print(f"Test Loss: {running_loss / len(test_loader):.4f}, Test Accuracy: {accuracy:.2f}%")

        # Wyświetlanie procentowych trafień dla każdego gatunku
        print("\nClass-wise accuracy:")
        for i in range(10):
            class_accuracy = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
            print(f"Class {i}: {class_accuracy:.2f}%")

    def test_fusion_model(fusion_model, cnn_model, rnn_model, cnn_test_loader, rnn_test_loader, criterion):
        fusion_model.eval()  # Przełączamy model na tryb ewaluacji
        cnn_model.eval()  # Model CNN w trybie ewaluacji
        rnn_model.eval()  # Model RNN w trybie ewaluacji

        correct = 0
        total = 0
        running_loss = 0.0
        all_preds = []
        all_labels = []

        # Słownik do przechowywania trafień dla każdego gatunku
        class_correct = {i: 0 for i in range(10)}  # Zakładamy, że mamy 10 gatunków
        class_total = {i: 0 for i in range(10)}

        with torch.no_grad():
            for (cnn_inputs, _), (rnn_inputs, labels) in zip(cnn_test_loader, rnn_test_loader):
                cnn_output = cnn_model(cnn_inputs)
                rnn_output = rnn_model(rnn_inputs)

                # Forward pass przez model fusion
                outputs = fusion_model(cnn_output, rnn_output)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Obliczanie trafień dla każdej klasy
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    class_total[label] += 1
                    if predicted[i].item() == label:
                        class_correct[label] += 1

                all_preds.append(predicted)
                all_labels.append(labels)

        accuracy = 100 * correct / total
        print(f"Test Loss: {running_loss / len(cnn_test_loader):.4f}, Test Accuracy: {accuracy:.2f}%")

        # Wyświetlanie procentowych trafień dla każdego gatunku
        print("\nClass-wise accuracy:")
        for i in range(10):
            class_accuracy = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
            print(f"Class {i}: {class_accuracy:.2f}%")

    # Inicjalizacja modeli i optymalizatorów
    cnn_model = SpectCNN(num_classes=10)
    rnn_model = TemporalRNN(input_size=42, hidden_size=128, num_layers=2, rnn_output_size=128)
    fusion_model = FusionModel(input_size=138, hidden_size=128, num_classes=10)

    # Ładowanie danych dla każdego modelu
    cnn_model.load_data(
        spect_dir=r'data\datasets\andradaolteanu\gtzan-dataset-music-genre-classification\versions\1\Data\images_original',
        batch_size=32)
    rnn_model.load_data(
        csv_file=r'data\datasets\andradaolteanu\gtzan-dataset-music-genre-classification\versions\1\Data\all_features_nowe.csv',
        batch_size=32)

    # Kryterium i optymalizatory
    criterion = nn.CrossEntropyLoss()
    optimizer_cnn = torch.optim.SGD(cnn_model.parameters(), lr=0.001)
    optimizer_rnn = torch.optim.SGD(rnn_model.parameters(), lr=0.001)
    optimizer_fusion = torch.optim.SGD(filter(lambda p: p.requires_grad, fusion_model.parameters()), lr=0.001)

    # Trenowanie indywidualnych modeli
    train_model(cnn_model, cnn_model.train_loader, cnn_model.test_loader, criterion, optimizer_cnn, num_epochs=1)
    train_model(rnn_model, rnn_model.train_loader, rnn_model.test_loader, criterion, optimizer_rnn, num_epochs=10)

    # Testowanie indywidualnych modeli
    print("\nTesting CNN model\n")
    test_model(cnn_model, cnn_model.test_loader, criterion)
    print("\nTesting RNN model\n")
    test_model(rnn_model, rnn_model.test_loader, criterion)

    # Trenowanie modelu fusion
    train_fusion_model(fusion_model, cnn_model, rnn_model, cnn_model.train_loader, rnn_model.train_loader, criterion,
                       optimizer_fusion, num_epochs=1)

    # Testowanie modelu fusion
    print("\nTesting Fusion model\n")
    test_fusion_model(fusion_model, cnn_model, rnn_model, cnn_model.test_loader, rnn_model.test_loader, criterion)
    print("Chyba działa XD")
if __name__ == '__main__':
    main()