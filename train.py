import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import util as ut
from torch.autograd import Variable
import numpy as np
from dataset import MovieDataset
from model import ConvolutionalNeuralNet

CRITERION = nn.CrossEntropyLoss()
N_EPOCHS = 100
BATCH_SIZE = 25
ADAM_ALPHA = 0.001
ADAM_BETA = (0.9, 0.999)
PRINT_INTERVAL = 5

cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def main():

    train_data = MovieDataset("data/temp/dataset10000/traindata.pkl")
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    # Instantiate the models
    CNN = ConvolutionalNeuralNet().to(device)
    Optimizer = torch.optim.Adam(CNN.parameters(), lr=ADAM_ALPHA, betas=ADAM_BETA)
    print('Training.')
    for epoch_index in range(N_EPOCHS):  # loop over the dataset multiple times
        for batch_index, batch_data in enumerate(train_loader):
            # get the inputs
            imgs, labels = batch_data

            imgs = Variable(imgs.type(FloatTensor)).to(device)
            labels = Variable(labels.type(LongTensor)).to(device)

            # zero the parameter gradients
            Optimizer.zero_grad()

            # forward + backward + optimize
            outputs = CNN(imgs)
            loss = CRITERION(outputs, labels)
            loss.backward()
            Optimizer.step()

            # Print Loss
            if epoch_index % PRINT_INTERVAL == 0 and not batch_index:    # print every 2000 mini-batches
                print('Epoch: %d \tLoss: %.3f' % (epoch_index, loss))

    print('Finished Training. Testing on Validation Set.')


    #Validation Set
    val_data = MovieDataset("data/temp/dataset10000/valdata.pkl")
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

    correct_predicted = 0
    total_predicted = 0
    with torch.no_grad():
        for data in val_loader:
            imgs, labels = data
            imgs = Variable(imgs.type(FloatTensor)).to(device)
            labels = Variable(labels.type(LongTensor)).to(device)
            outputs = CNN(imgs)
            _, predicted = torch.max(outputs.data, 1)
            total_predicted += labels.size(0)
            correct_predicted += (predicted == labels).sum().item()

    print('Accuracy of the CNN on Validation Set: %d %%' % (
        100 * correct_predicted / total_predicted))

    classes = ut.unpickle("data/class_indeces.pkl")
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    with torch.no_grad():
        for data in val_loader:
            imgs, labels = data
            imgs = Variable(imgs.type(FloatTensor)).to(device)
            labels = Variable(labels.type(LongTensor)).to(device)
            outputs = CNN(imgs)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
                
    for i in range(len(classes)):
        if class_total[i] == 0:
            print('Accuracy of %5s : N/A' % (classes[i]))
        else:
            print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
        
    
if __name__ == '__main__':
    main()
