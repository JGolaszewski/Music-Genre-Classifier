import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets
from tqdm import tqdm

# Spectrogram Convolutional Neural Network Model
class SpectCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SpectCNN, self).__init__()

        # Warstwy konwolucyjne
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

    def _criterion(self):
        return nn.CrossEntropyLoss()

    def _optimizer(self):
        return optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

    def load_data(self, img_folder_root, batch_size=4, shuffle=True, num_workers=2):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        trainset = datasets.ImageFolder(root=img_folder_root, transform=transform)
        testset = trainset

        self.classes = trainset.classes
        self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle,
                                                        num_workers=num_workers)
        self.test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,
                                                       num_workers=num_workers)

    def data_train(self):
        assert self.train_loader is not None
        assert self.test_loader is not None

        criterion = self._criterion()
        optimizer = self._optimizer()

        num_epochs = 10

        for epoch in range(num_epochs):
            running_loss = 0.0
            epoch_loss = 0.0
            num_batches = len(self.train_loader)

            # Pasek postępu dla epoki
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True)

            for i, (inputs, labels) in enumerate(progress_bar, 1):
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                epoch_loss += loss.item()

                progress_bar.set_postfix(loss=(running_loss / i))

            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch + 1} finished. Avg Loss: {avg_loss:.4f}")

    def data_test(self):
        assert self.train_loader is not None
        assert self.test_loader is not None

        correct = [0] * len(self.classes)  # Tworzymy listę poprawnych przewidywań dla każdej klasy
        total = [0] * len(self.classes)    # Tworzymy listę liczby próbek w każdej klasie

        with torch.no_grad():
            for data in self.test_loader:
                images, labels = data
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)

                # Liczymy poprawne klasyfikacje dla każdej klasy
                for i in range(labels.size(0)):
                    label = labels[i]
                    correct[label] += (predicted[i] == label).item()
                    total[label] += 1

        # Wyświetlamy dokładność dla każdej klasy
        for i, class_name in enumerate(self.classes):
            accuracy = 100 * correct[i] / total[i] if total[i] > 0 else 0
            print(f'Accuracy for class {class_name}: {accuracy:.2f} %')

        # Możesz także obliczyć ogólną dokładność:
        overall_accuracy = 100 * sum(correct) / sum(total)
        print (f'Overall accuracy: {overall_accuracy:.2f} %')

class TemporalRNN(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, rnn_output_size=128):
        super(TemporalRNN, self).__init__()

        self.test_loader = None
        self.train_loader = None

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

    def _criterion(self):
        return nn.CrossEntropyLoss()

    def _optimizer(self):
        return optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

    def load_data(self, data, batch_size=4, shuffle=True, num_workers=2):
        self.train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle,
                                                        num_workers=num_workers)
        self.test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False,
                                                        num_workers=num_workers)

    def data_train(self):
        assert self.train_loader is not None
        assert self.test_loader is not None

        criterion = self._criterion()
        optimizer = self._optimizer()

        num_epochs = 10

        for epoch in range(num_epochs):
            running_loss = 0.0
            epoch_loss = 0.0
            num_batches = len(self.train_loader)

            # Pasek postępu dla epoki
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True)

            for i, (inputs, labels) in enumerate(progress_bar, 1):
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                epoch_loss += loss.item()

                progress_bar.set_postfix(loss=(running_loss / i))

            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch + 1} finished. Avg Loss: {avg_loss:.4f}")

    def data_test(self):
        assert self.train_loader is not None
        assert self.test_loader is not None

        correct = 0
        total = 0

        with torch.no_grad():
            for data in self.test_loader:
                inputs, labels = data
                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy: {100 * correct / total:.2f} %')

class CombRNNCNN(nn):
    def __init__(self, cnn, rnn):
        super(CombRNNCNN, self).__init__()

        self.cnn = cnn
        self.rnn = rnn

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.rnn(x)
        return x

    def _criterion(self):
        return nn.CrossEntropyLoss()

    def _optimizer(self):
        return optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

    def load_data(self, img_folder_root, batch_size=4, shuffle=True, num_workers=2):
        self.cnn.load_data(img_folder_root, batch_size, shuffle, num_workers)

    def data_train(self):
        self.cnn.data_train()
        self.rnn.load_data(self.cnn.train_loader.dataset)
        self.rnn.data_train()

    def data_test(self):
        self.cnn.data_test()
        self.rnn.load_data(self.cnn.test_loader.dataset)
        self.rnn.data_test()