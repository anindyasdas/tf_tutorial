import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys


from torch.utils.tensorboard import SummaryWriter
# Device configuration
writer= SummaryWriter("runs/mnist2") #default is ./runs folder
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 784  # 28x28
hidden_size = 500
num_classes = 10
num_epochs = 2
batch_size = 100
#learning_rate = 0.001
learning_rate = 0.01

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

examples = iter(test_loader)
example_data, example_targets = examples.next()
#for i in range(6):
#    plt.subplot(2, 3, i + 1)
#    plt.imshow(example_data[i][0], cmap='gray')
#plt.show()
img_grid= torchvision.utils.make_grid(example_data)
writer.add_image("mnist_images", img_grid)
#writer.close()
#sys.exit()

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.tanh=nn.Tanh()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        feature=self.tanh(out)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out, feature


model = NeuralNet(input_size, hidden_size, num_classes).to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
writer.add_graph(model, example_data.reshape(-1,28*28))
writer.close()
#sys.exit()



# Train the model
n_total_steps = len(train_loader)
print("n_total_steps", n_total_steps)
running_loss=0.0
running_correct=0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [100, 1, 28, 28]
        # resized: [100, 784]
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs, _ = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
        _, predicted = torch.max(outputs.data, 1)
        running_correct=(predicted==labels).sum().item()


        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')
            writer.add_scalar("training loss", running_loss/100, epoch  * n_total_steps + i)
            writer.add_scalar("accuracy", running_correct/ 100, epoch * n_total_steps + i)
            running_loss=0.0
            running_correct=0

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
tot_lables=[]
preds=[]
features=[]
imgs=[]
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        imgs.append(images)
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        outputs, feature = model(images)
        features.append(feature)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        tot_lables.append(labels)
        outputs_prob= F.softmax(outputs, dim=1)
        preds.append(outputs_prob)

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 10000 test images: {acc} %')
    labels=torch.cat(tot_lables)
    print("labels.shape", labels.shape)
    predictions=torch.cat(preds)
    n_classes=10
    ####plotting as binary classification
    for class_i in range(n_classes):
        labels_i =  labels==class_i #binary labels
        predicts_i= predictions[:, class_i]
        #print("labels",labels_i)
        writer.add_pr_curve('class_'+str(class_i), labels_i, predicts_i, global_step=0) #precesion recall
        writer.close()

    #####Projector
    imgs=torch.cat(imgs) #torch.Size([10000, 1, 28, 28])
    features=torch.cat(features) #torch.Size([10000, 500])
    #print(imgs.shape, features.shape, labels.shape)
    ####taking first n entries
    n=500
    imgs=imgs[:n,]
    features=features[:n,]
    labels=labels[:n,]
    writer.add_embedding(features, metadata=labels.tolist(), label_img=imgs, global_step=1)
    writer.close()



