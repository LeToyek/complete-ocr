import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import le_net as mo
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

def show_images(images,labels):
    pass

# Function to visualize the outputs of each layer
def main():
    batch_size = 64
    num_classes = 10
    learning_rate = 0.001
    num_epochs = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = torchvision.datasets.MNIST(
        root="../data",
        train=True,
        transform=transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.30081,)),
            ]
        ),
        download=True,
    )

    test_dataset = torchvision.datasets.MNIST(
        root="../data",
        train=False,
        transform=transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.1325,), (0.3105,)),
            ]
        ),
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=True
    )

    num_classes: int = 10
    model = mo.LeNet5(10).to(device)
    cost = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = cost(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 400 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                        epoch + 1, num_epochs, i + 1, total_step, loss.item()
                    )
                )
                
    with torch.no_grad():
        correct = 0
        total = 0
        for images,labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _,predicted = torch.max(outputs.data,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print("Accuracy of the model on the 10000 test images: {} %".format(100 * correct / total))
    

if __name__ == "__main__":
    main()