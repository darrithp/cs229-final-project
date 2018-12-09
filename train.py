import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import util as ut
import os
import sys
from torch.autograd import Variable
import numpy as np
from dataset import MovieDataset
from model import MultiClassifier

CRITERION = nn.CrossEntropyLoss()
N_EPOCHS = 100
BATCH_SIZE = 25
ADAM_ALPHA = 0.001
ADAM_BETA = (0.9, 0.999)
PRINT_INTERVAL = 5
DATASET_RAW_PATH = os.path.join("data","temp")
TRAIN_DATA = "traindata.pkl"
VAL_DATA = "valdata.pkl"
TEST_DATA = "testdata.pkl"
CLASS_INDECES_RAW_PATH = os.path.join("data","class_indeces.pkl")

cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def train_multiclassifier(dataset_path):

    train_path = os.path.join(DATASET_RAW_PATH, dataset_path, TRAIN_DATA)
    train_data = MovieDataset(train_path)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    # Instantiate the models
    MC = MultiClassifier().to(device)
    Optimizer = torch.optim.Adam(MC.parameters(), lr=ADAM_ALPHA, betas=ADAM_BETA)
    print('Training.')
    for epoch_index in range(N_EPOCHS):  # loop over the dataset multiple times
        for batch_index, batch_data in enumerate(train_loader):
            # get the inputs
            imgs, labels = batch_data
            imgs = Variable(imgs.type(FloatTensor)).to(device)
            labels = torch.stack(labels)
            labels = torch.transpose(labels, 0, 1)
            labels = Variable(labels).to(device)
            #labels = Variable(labels.to(device))
            # zero the parameter gradients
            Optimizer.zero_grad()

            # forward + backward + optimize
            outputs = MC(imgs)
            loss = CRITERION(outputs, torch.max(labels, 1)[1])
            loss.backward()
            Optimizer.step()

            # Print Loss
            if epoch_index % PRINT_INTERVAL == 0 and not batch_index:    # print every 2000 mini-batches
                print('Epoch: %d \tLoss: %.3f' % (epoch_index, loss))

    print('Finished Training. Testing on Validation Set.')


    #Validation Set
    val_path = os.path.join(DATASET_RAW_PATH, dataset_path, VAL_DATA)
    val_data = MovieDataset(val_path)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

    correct_predicted = 0
    total_predicted = 0
    with torch.no_grad():
        for data in val_loader:
            imgs, labels = data
            imgs = Variable(imgs.type(FloatTensor)).to(device)
            labels = torch.stack(labels)
            labels = torch.transpose(labels, 0, 1)
            labels = Variable(labels).to(device)
            outputs = MC(imgs)
            _, predicted = torch.max(outputs.data, 1)
            #total_predicted += labels.size(0)
            #correct_predicted += (labels[predicted] == 1).sum().item()
            for batch_i in range (predicted.size(0)):
                total_predicted += 1
                correct_predicted += (labels[batch_i][predicted[batch_i]].item() == 1)

    print('Accuracy of the CNN on Validation Set: %d %%' % (
        100 * correct_predicted / total_predicted))

    classes = ut.unpickle(CLASS_INDECES_RAW_PATH)
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    with torch.no_grad():
        for data in val_loader:
            imgs, labels = data
            imgs = Variable(imgs.type(FloatTensor)).to(device)
            labels = torch.stack(labels)
            labels = torch.transpose(labels, 0, 1)
            labels = Variable(labels).to(device)
            outputs = MC(imgs)
            _, predicted = torch.max(outputs, 1)
            '''
            c = (labels[predicted] == 1).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
            '''
            for batch_i in range (predicted.size(0)):
                label = np.argmax(labels[batch_i])
                class_total[label] += 1
                class_correct[label] += (labels[batch_i][predicted[batch_i]].item() == 1)
                
    for i in range(len(classes)):
        if class_total[i] == 0:
            print('Accuracy of %5s : N/A' % (classes[i]))
        else:
            print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

def train_multilabelclassifier(dataset_path):
    print('TODO')


def main():
    if len(sys.argv) < 3:
        print('Usage: python3 train.py [MC|ML] [dataset_name]')
        exit()
    train_type = sys.argv[1] 
    if train_type == "MC":
        train_multiclassifier(sys.argv[2])
    elif train_type == "ML":
        train_multilabelclassifier(sys.argv[2])
    else:
        print('Usage: python3 train.py [MC|ML] [dataset_name]')
        exit()
    
if __name__ == '__main__':
    main()
