import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Definir um modelo simples
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Camada totalmente conectada
        self.fc2 = nn.Linear(128, 10)  # Camada de saída para 10 classes (MNIST)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Achatar a imagem 28x28 em um vetor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Carregar o dataset MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Inicializar o modelo, critério de perda e otimizador
model = SimpleNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Selecionar uma única batch de dados
data, targets = next(iter(train_loader))  # Pega a primeira batch
data, targets = data.cuda(), targets.cuda()  # Enviar para GPU (se disponível)

# Loop de treinamento para uma única batch
num_epochs = 100  # Número de vezes que a batch será treinada

for epoch in range(num_epochs):
    model.train()  # Modo de treinamento
    optimizer.zero_grad()  # Zera os gradientes antes do backpropagation

    # Forward pass (passar os dados pela rede)
    outputs = model(data)

    # Calcular a perda
    loss = criterion(outputs, targets)

    # Backward pass (calcular gradientes)
    loss.backward()

    # Atualizar os pesos
    optimizer.step()

    # Imprimir a perda a cada 10 épocas
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# O modelo deve ser capaz de memorizar a batch rapidamente, e a perda deve diminuir consideravelmente.
