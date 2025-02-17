import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import os


def main():
    global accuracy
    data_dir = r'C:\Users\Komputer\Desktop\studia-Informatyka\Koła\Golem\projekt-rekrutacja\Music-Genre-Classifier\data\datasets\andradaolteanu\gtzan-dataset-music-genre-classification\versions\1\Data\images_original'

    # Transformacje danych
    transform = transforms.Compose([
        transforms.ToTensor(),  # Konwersja obrazu na tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalizacja do zakresu [-1, 1]
    ])

    # Wczytanie danych treningowych i testowych z podfolderów
    trainset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)

    # Pobieramy etykiety (labels) dla każdego obrazu
    labels = np.array(trainset.targets)  # 'targets' zawiera etykiety klas dla ImageFolder

    # Używamy stratified split, aby podzielić dane na zbiór treningowy i testowy
    train_indices, test_indices = train_test_split(
        np.arange(len(trainset)),  # indeksy wszystkich obrazów
        test_size=0.2,  # 20% danych do testów
        stratify=labels  # Stratifikacja, czyli podział proporcjonalny do klas
    )

    # Tworzymy SubsetDataset na podstawie indeksów
    train_dataset = Subset(trainset, train_indices)
    test_dataset = Subset(trainset, test_indices)

    # Ustawienia DataLoadera
    batch_size = 3

    # DataLoader dla treningu i testów
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Definicja klas (opcjonalnie, można użyć `trainset.classes`)
    classes = ('blues', 'classical', 'country', 'disco',
               'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock')

    # Sprawdzenie poprawności danych
    dataiter = iter(trainloader)
    images, labels = next(dataiter)


#UCZENIE
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 8, 5)
            self.pool = nn.MaxPool2d(2, 1)
            self.conv2 = nn.Conv2d(8, 16, 5)

            # Obliczanie dynamicznego wymiaru dla fc1
            self._calculate_fc1_input_dim()

            self.fc1 = nn.Linear(self.fc1_input_dim, 120)  # Zmieniono wymiar
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def _calculate_fc1_input_dim(self):
            # Symulowanie przejścia przez warstwy konwolucyjne i pooling
            sample_input = torch.zeros(1, 3, 432, 288)  # Przykładowy obraz wejściowy o rozmiarze 224x224
            sample_output = self.pool(F.relu(self.conv1(sample_input)))
            sample_output = self.pool(F.relu(self.conv2(sample_output)))

            # Oblicz wymiar wyjściowy i zapisz go jako fc1_input_dim
            self.fc1_input_dim = sample_output.numel()

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1)  # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(10):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

             #print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

    print('Finished Training')

    # Zapis modelu
    #torch.save(net, 'model.pth')

    #net = torch.load('model.pth')
    #net.eval()  # Ustawienie modelu w tryb ewaluacji (nie aktualizuje wag)
    # print images
    #imshow(torchvision.utils.make_grid(images))
    # Wyświetlanie prawdziwych etykiet (GroundTruth)
    #print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(min(len(labels), 3))))

    # Przewidywania sieci
    #outputs = net(images)
   # _, predicted = torch.max(outputs, 1)

    # Wyświetlanie przewidywanych etykiet
   # print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(min(len(predicted), 3))))

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model_accuracy = 100 * correct // total
    print(f'Accuracy of the network on the test images: {model_accuracy} %')

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.3f} %')

    save_dir = 'save'

    # Sprawdzenie, czy folder istnieje, jeśli nie to tworzymy go
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Zdefiniowanie nazwy pliku z dokładnością w nazwie
    model_filename = f'model_accuracy_{model_accuracy:.3f}.pth'

    # Pełna ścieżka do pliku, który chcemy zapisać
    save_path = os.path.join(save_dir, model_filename)

    # Zapisanie modelu
    torch.save(net.state_dict(), save_path)
    print(f'Model saved as: {save_path}')

if __name__ == "__main__":
    main()