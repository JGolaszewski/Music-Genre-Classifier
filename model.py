import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torchvision import transforms, datasets
from tqdm import tqdm
import pandas as pd
import numpy as np


"""
    MODEL DEFINITIONS
"""

class SpectCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SpectCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Warstwy FC z Dropout
        self.fc1 = nn.Linear(128 * 54 * 36, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

        # Inicjalizacja wag He (lepsza dla ReLU)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Przechodzi przez warstwę konwolucyjną + pooling
        x = self.pool(F.relu(self.conv2(x)))  # Przechodzi przez drugą warstwę
        x = self.pool(F.relu(self.conv3(x)))  # Przechodzi przez trzecią warstwę

        x = x.view(-1, 128 * 54 * 36)  # Spłaszczenie przed warstwą FC
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
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

class TemporalRNN(nn.Module):
    def __init__(self, input_size=13, hidden_size=128, num_layers=2, rnn_output_size=128):
        super(TemporalRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, rnn_output_size)
        self.dropout = nn.Dropout(0.5)
        self.bn = nn.BatchNorm1d(rnn_output_size)

        nn.init.xavier_normal_(self.fc.weight)

        # Inicjalizacja wag
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)

        out, _ = self.rnn(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        out = self.bn(out)
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

        scaler = StandardScaler()
        sequences_reshaped = sequences.reshape(-1, sequences.shape[-1])
        sequences_reshaped = scaler.fit_transform(sequences_reshaped)
        sequences = sequences_reshaped.reshape(sequences.shape)

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

class FusionModel(nn.Module):
        def __init__(self, input_size=138, hidden_size=128, num_classes=10):
            super(FusionModel, self).__init__()

            self.fc1 = nn.Linear(input_size, hidden_size)
            self.bn1 = nn.BatchNorm1d(hidden_size)
            self.dropout = nn.Dropout(0.3)
            self.fc2 = nn.Linear(hidden_size, num_classes)

            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        def forward(self, cnn_output, rnn_output):
            # Łączenie wyjść z obu modeli
            fused_input = torch.cat((cnn_output, rnn_output), dim=1)  # Konkatenacja po wymiarze cech

            # Przejście przez warstwy w pełni połączone
            x = self.fc1(fused_input)  # Linear(input_size, hidden_size)
            x = self.bn1(x)  # BatchNorm1d(hidden_size)
            x = F.relu(x)  # ReLU()
            x = self.dropout(x)  # Dropout(0.3)
            x = self.fc2(x)  # Linear(hidden_size, num_classes)

            return x

        def evaluate_model(self, cnn_model, rnn_model, cnn_loader, rnn_loader):
            self.eval()

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

"""
    HELPER FUNCTIONS
"""

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10):
    model.train()

    train_losses = []
    train_accuracies = []

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

        # Oblicz średnią stratę i dokładność dla epoki
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total

        # Dodaj wyniki do list
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        print(f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.2f}%")

    results_df = pd.DataFrame({
        'Epoch': range(1, num_epochs + 1),
        'Train Loss': train_losses,
        'Train Accuracy': train_accuracies
    })

    return results_df

class FusionDataset(torch.utils.data.Dataset):
    def __init__(self, cnn_loader, rnn_loader):
        self.cnn_loader = cnn_loader
        self.rnn_loader = rnn_loader

    def __len__(self):
        return min(len(self.cnn_loader.dataset), len(self.rnn_loader.dataset))

    def __getitem__(self, idx):
        # Pobierz dane z CNN i RNN z tego samego indeksu
        cnn_data = self.cnn_loader.dataset[idx]
        rnn_data = self.rnn_loader.dataset[idx]

        cnn_inputs, _ = cnn_data
        rnn_inputs, labels = rnn_data

        return (cnn_inputs, rnn_inputs), labels

# W modyfikowanej funkcji `train_fusion_model` używamy tego nowego datasetu
def train_fusion_model(self, cnn_model, rnn_model, cnn_loader, rnn_loader, criterion, optimizer, num_epochs=10):
    # Tworzymy nowy dataset Fusion
    fusion_dataset = FusionDataset(cnn_loader, rnn_loader)
    fusion_loader = torch.utils.data.DataLoader(fusion_dataset, batch_size=32, shuffle=False)

    self.train()  # Włączenie trybu treningowego dla FusionModel
    cnn_model.eval()  # Model CNN w trybie ewaluacji
    rnn_model.eval()  # Model RNN w trybie ewaluacji

    train_losses = []
    train_accuracies = []
    validation_accuracies = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for (cnn_inputs, rnn_inputs), labels in tqdm(fusion_loader,
                                                     desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
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

        epoch_loss = running_loss / len(rnn_loader)
        epoch_accuracy = 100 * correct / total

        # Dodaj wyniki do list
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        print(f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.2f}%")

        # Kroki walidacyjne
        self.eval()  # Przełączamy model na tryb ewaluacji
        correct = 0
        total = 0
        with torch.no_grad():
            for (cnn_inputs, rnn_inputs), labels in tqdm(fusion_loader, desc="Validation"):
                cnn_output = cnn_model(cnn_inputs)
                rnn_output = rnn_model(rnn_inputs)

                outputs = self(cnn_output, rnn_output)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        validation_accuracies.append(val_accuracy)
        print(f"Validation Accuracy: {val_accuracy:.2f}%")
        self.train()  # Powrót do trybu treningowego po walidacji

    results_df = pd.DataFrame({
        'Epoch': range(1, num_epochs + 1),
        'Train Loss': train_losses,
        'Train Accuracy': train_accuracies,
        'Validation Accuracy': validation_accuracies
    })

    # Po zakończeniu treningu wykonaj ewaluację
    self.evaluate_model(cnn_model, rnn_model, cnn_loader, rnn_loader)

    return results_df

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
    results = []
    for i in range(10):
        class_accuracy = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        results.append({'Class': i, 'Accuracy': class_accuracy})
        print(f"Class {i}: {class_accuracy:.2f}%")

    results_df = pd.DataFrame(results)
    return results_df

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
    results = []
    for i in range(10):
        class_accuracy = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        results.append({'Class': i, 'Accuracy': class_accuracy})
        print(f"Class {i}: {class_accuracy:.2f}%")

    results_df = pd.DataFrame(results)
    return results_df