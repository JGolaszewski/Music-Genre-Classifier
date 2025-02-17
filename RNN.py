import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split  # Importujemy funkcję train_test_split
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np


# Klasa Dataset dla danych sekwencyjnych
class TemporalDataset(Dataset):
    def __init__(self, sequences_in, labels_in):
        self.m_sequences = sequences_in
        self.m_labels = labels_in

    def __len__(self):
        return len(self.m_sequences)

    def __getitem__(self, idx):
        return self.m_sequences[idx], self.m_labels[idx]

# 1. Wczytanie danych
df = pd.read_csv(
    r'data\datasets\andradaolteanu\gtzan-dataset-music-genre-classification\versions\1\Data\all_features.csv')

# Usuwanie wierszy z NaN oraz 'name' (niepotrzebne)
df = df.dropna()
df = df.loc[:, df.columns != 'name']

# Lista punktów czasowych (np. t0, t1, ..., t29)
time_points = [str(i) for i in range(30)]
labels = []
sequences = []

# 2. Przetwarzanie danych na sekwencje czasowe
for index, row in df.iterrows():
    sequence = []  # Lista, która przechowuje sekwencję cech dla danego utworu

    # Iterujemy po punktach czasowych w kolejności
    for t in time_points:
        cols = [col for col in df.columns if col.endswith(f'_t{t}')]  # Wybieramy odpowiednie kolumny
        sequence.append(row[cols].values.tolist())  # Dodajemy cechy dla tego punktu czasowego

    labels.append(row["genre"])  # Dodajemy etykietę (gatunek)
    sequences.append(sequence)  # Dodajemy sekwencję cech

# 3. Konwersja etykiet `genre` na liczby
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)

# 4. Konwersja danych do tablicy NumPy
sequences = np.array(sequences)
sequences_tensor = torch.tensor(sequences, dtype=torch.float32)
labels_tensor = torch.tensor(y_encoded, dtype=torch.long)

# 5. Podział danych na zbiór uczący i testowy (z zachowaniem proporcji etykiet)
X_train, X_test, y_train, y_test = train_test_split(sequences_tensor, labels_tensor, test_size=0.2, stratify=y_encoded,
                                                    random_state=42)

# 6. Utworzenie Dataset i DataLoader dla zbioru uczącego
train_dataset = TemporalDataset(X_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 7. Utworzenie Dataset i DataLoader dla zbioru testowego
test_dataset = TemporalDataset(X_test, y_test)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# 9. Definicja LSTM
class MusicGenreLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MusicGenreLSTM, self).__init__()
        self.hidden_size = hidden_size

        # Zmiana z RNN na LSTM oraz dodanie więcej warstw
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=2, dropout=0.3)

        # Dropout dla regularizacji
        self.dropout = nn.Dropout(0.4)

        # Batch Normalization
        self.batch_norm = nn.BatchNorm1d(hidden_size)

        # Aktywacja LeakyReLU
        self.activation = nn.LeakyReLU(negative_slope=0.01)

        # Warstwa w pełni połączona
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Inicjalizacja stanów ukrytych dla LSTM (h0, c0)
        h0 = torch.randn(2, x.size(0), self.hidden_size).to(x.device)  # Inicjalizacja z losowymi wartościami
        c0 = torch.randn(2, x.size(0), self.hidden_size).to(x.device)  # Inicjalizacja komórki w LSTM

        # Przechodzimy przez LSTM
        out, _ = self.rnn(x, (h0, c0))

        # Używamy tylko ostatniego ukrytego stanu (ostatnia sekwencja)
        out = out[:, -1, :]

        # Normalizacja BatchNorm
        out = self.batch_norm(out)

        # Aktywacja
        out = self.activation(out)

        # Dropout dla regularizacji
        out = self.dropout(out)

        # Klasyfikacja końcowa
        out = self.fc(out)

        return out


# 10. Parametry sieci
input_size = sequences_tensor.shape[2]  # Liczba cech w danych sekwencyjnych
hidden_size = 128  # Można dostosować
num_classes = len(label_encoder.classes_)

# 11. Inicjalizacja sieci, funkcji kosztu i optymalizatora
model = MusicGenreLSTM(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

# 12. Trening modelu
num_epochs = 300  # Można dostosować
for epoch in range(num_epochs):
    for X_batch, y_batch in train_dataloader:
        # Nie dodajemy wymiaru sekwencji, ponieważ dane mają już odpowiedni wymiar (batch_size, num_time_steps, input_size)
        # X_batch = X_batch.unsqueeze(1)  # Dodajemy wymiar sekwencji (1) – niepotrzebne

        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward pass i optymalizacja
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# 13. Testowanie modelu z dokładnością dla każdego gatunku
model.eval()  # Ustawienie modelu w tryb ewaluacji


with torch.no_grad():  # Wyłączanie gradientów, bo nie trenujemy teraz modelu
    genre_accuracy = {genre: {'correct': 0, 'total': 0} for genre in label_encoder.classes_}
    total = 0
    correct = 0
    for X_batch, y_batch in test_dataloader:
        # Forward pass
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)

        # Obliczanie dokładności dla każdego gatunku
        for true, pred in zip(y_batch, predicted):
            genre = label_encoder.classes_[true.item()]
            genre_accuracy[genre]['total'] += 1
            if true == pred:
                genre_accuracy[genre]['correct'] += 1

        # Obliczanie całkowitej dokładności
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

# Obliczanie dokładności dla każdego gatunku
for genre in label_encoder.classes_:
    correct = genre_accuracy[genre]['correct']
    total = genre_accuracy[genre]['total']
    accuracy = 100 * correct / total if total > 0 else 0  # Zapewnia, że nie dzielimy przez zero
    print(f"Accuracy for genre '{genre}': {accuracy:.2f}%")

# Obliczanie całkowitej dokładności
accuracy = 100 * correct / total
print(f'Accuracy on test set: {accuracy:.2f}%')
