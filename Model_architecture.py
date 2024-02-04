import torch
import torch.nn as nn # ALL nerual Nerwork architecture
import torch.nn.functional as F # All funcional
import torchvision.datasets as datasets
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import gradio 

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperarmeter
batch_size = 100
learning_rate = 0.001
num_class = 10 # 0 - 9 digit
num_epochs = 5

train_data = datasets.MNIST("MNIST_DATA", train = True, download = True, transform = transforms.ToTensor())
train_loader = DataLoader(train_data,batch_size = batch_size, shuffle = True)
test_data = datasets.MNIST("MNIST_DATA", train = False, download= True, transform = transforms.ToTensor())
test_loader = DataLoader(test_data, batch_size = batch_size, shuffle=False)

# CNN Architecture
class CNN(nn.Module):
    def __init__(self, num_class = 10):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size = 5, stride = 1, padding = 2),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2)
    )
        self.conv2 = nn.Sequential(
        nn.Conv2d(16, 32, kernel_size = 5, stride = 1, padding = 2),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2)
    )
        self.fc = nn.Linear(7 * 7 * 32,num_class) # 28 x 28 -> 14 x 14 -> 7 x 7

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

model = CNN(num_class).to(device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate) # using Adam Algorithm to optimizer parameter

total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backpropation and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
        
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        hehe, predicted = torch.max(outputs.data, 1) # find max value in each row
        print(hehe)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Model Accuracy on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
# torch.save(model.state_dict(), 'testModel.ckpt')