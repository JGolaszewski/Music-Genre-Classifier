import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy.cluster.hierarchy import single
from torchvision import transforms, datasets
from tqdm import tqdm
import pandas as pd
import numpy as np

# Spectrogram Convolutional Neural Network Model
class SpectCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SpectCNN, self).__init__()

        self.test_loader = None
        self.train_loader = None

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, 5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten_dim = 108 * 72 * 64

        self.fc1 = nn.Linear(self.flatten_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool2(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def load_data(self, spect_dir, batch_size=4):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        trainset = datasets.ImageFolder(root=spect_dir, transform=transform)
        testset = trainset

        self.classes = trainset.classes
        self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                                        num_workers=2)
        self.test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,
                                                       num_workers=2)

class TemporalRNN(nn.Module):
    def __init__(self, input_size=13, hidden_size=64, num_layers=2, rnn_output_size=128):
        super(TemporalRNN, self).__init__()

        self.train_loader = None
        self.test_loader = None

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, rnn_output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.rnn(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

    def load_data(self, csv_file, batch_size=32):
        df = pd.read_csv(csv_file)
        print(f"Rozmiar danych: {df.shape}")

        time_points = [str(i) for i in range(30)]
        labels = []
        sequences = []

        for index, row in df.iterrows():  # Iterujemy po utworach (wierszach)
            sequence = []  # Przechowuje sekwencję dla danego utworu

            for t in sorted(time_points):  # Iterujemy po krokach czasowych w kolejności
                cols = [col for col in df.columns if col.endswith(f'_t{t}')]
                sequence.append(row[cols].values.tolist())  # Dodajemy wartości dla danego kroku t

            labels.append(row["genre"])
            sequences.append(sequence)

        unique_labels = sorted(set(labels))
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

        numeric_labels = [label_mapping[label] for label in labels]

        sequences = np.array(sequences)
        sequences_tensor = torch.tensor(sequences, dtype=torch.float32)
        labels_tensor = torch.tensor(numeric_labels, dtype=torch.int64)

        # Tworzymy Dataset dla PyTorch
        class TemporalDataset(torch.utils.data.Dataset):
            def __init__(self, sequences_in, labels_in):
                self.m_sequences = sequences_in
                self.m_labels = labels_in

            def __len__(self):
                return len(self.m_sequences)

            def __getitem__(self, idx):
                return self.m_sequences[idx], self.m_labels[idx]

        dataset = TemporalDataset(sequences_tensor, labels_tensor)

        self.train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


class CombRNNCNN(nn.Module):
    def __init__(self):
        super(CombRNNCNN, self).__init__()

        self.cnn = SpectCNN()
        self.rnn = TemporalRNN()
    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.rnn(x)
        return x

    def _criterion(self):
        return nn.CrossEntropyLoss()
    def _optimizer(self):
        return optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

    def load_data(self, csv_file, spect_dir, csv_batch_size=32, spect_batch_size=4):
        self.rnn.load_data(csv_file, batch_size=csv_batch_size)
        self.cnn.load_data(spect_dir, batch_size=spect_batch_size)

    def train_whole(self):
        self.single_train(self.cnn, 2)
        self.single_test(self.cnn)
        self.single_train(self.rnn, 300)
        self.single_test(self.rnn)

        self.single_train(self, 2)
        self.single_test(self)


    def single_train(self, model, num_epochs=10):
        assert model.train_loader is not None
        assert model.test_loader is not None

        criterion = self._criterion()
        optimizer = self._optimizer()

        for epoch in range(num_epochs):
            running_loss = 0.0
            epoch_loss = 0.0
            num_batches = len(model.train_loader)

            progress_bar = tqdm(model.train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True)

            for i, (inputs, labels) in enumerate(progress_bar, 1):
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                epoch_loss += loss.item()

                progress_bar.set_postfix(loss=(running_loss / i))

            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch + 1} finished. Avg Loss: {avg_loss:.4f}")
    def single_test(self, model):
        assert model.train_loader is not None
        assert model.test_loader is not None

        correct = 0
        total = 0

        with torch.no_grad():
            for data in model.test_loader:
                inputs, labels = data
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy: {100 * correct / total:.2f} %')