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
from model import MultiClassifier, MultiLabelClassifier

#CE_CRITERION = nn.CrossEntropyLoss()
#BCE_CRITERION = nn.BCEWithLogitsLoss()
N_EPOCHS = 30
BATCH_SIZE = 25
ADAM_ALPHA = 0.0000001
ADAM_BETA = (0.9, 0.999)
PRINT_INTERVAL = 5
DATASET_RAW_PATH = os.path.join("data","temp")
TRAIN_DATA = "traindata.pkl"
VAL_DATA = "valdata.pkl"
TEST_DATA = "testdata.pkl"
CLASS_INDECES_RAW_PATH = os.path.join("data","class_indeces.pkl")
REGULARIZATION = 0.01

cuda = torch.cuda.is_available()
device = 'cuda' if cuda else 'cpu'
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def train_multiclassifier(dataset_path, weights):
    CE_CRITERION = nn.CrossEntropyLoss(weight=weights)
    
    train_path = os.path.join(DATASET_RAW_PATH, dataset_path, TRAIN_DATA)
    train_data = MovieDataset(train_path)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    val_path = os.path.join(DATASET_RAW_PATH, dataset_path, VAL_DATA)
    val_data = MovieDataset(val_path)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)
    # Instantiate the models
    MC = MultiClassifier().to(device)
    Optimizer = torch.optim.Adam(MC.parameters(), lr=ADAM_ALPHA, betas=ADAM_BETA, weight_decay=REGULARIZATION)
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
            loss = CE_CRITERION(outputs, torch.max(labels, 1)[1])
            loss.backward()
            Optimizer.step()

            # Print Loss
            if epoch_index % PRINT_INTERVAL == 0 and not batch_index:    # print every 2000 mini-batches
                print('Epoch: %d \tTraining Loss: %.3f' % (epoch_index, loss))
                for val_batch_i, val_batch_data in enumerate(val_loader):
                    val_imgs, val_labels = val_batch_data
                    val_imgs = Variable(val_imgs.type(FloatTensor)).to(device)
                    val_labels = torch.stack(val_labels)
                    val_labels = torch.transpose(val_labels, 0, 1)
                    val_labels = Variable(val_labels).to(device)
                    val_outputs = MC(val_imgs)
                    val_loss = CE_CRITERION(val_outputs, torch.max(val_labels, 1)[1])
                    print('Epoch: %d \tValidation Loss: %.3f' % (epoch_index, val_loss))
                    break
    print('Finished Training. Testing on Validation Set.')


    #Validation Set
    #val_path = os.path.join(DATASET_RAW_PATH, dataset_path, VAL_DATA)
    #val_data = MovieDataset(val_path)
    #val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

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
            outputs = F.softmax(outputs.data, dim=1)
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
    BCE_CRITERION = nn.MultiLabelSoftMarginLoss()

    train_path = os.path.join(DATASET_RAW_PATH, dataset_path, TRAIN_DATA)
    train_data = MovieDataset(train_path)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    # Instantiate the models
    ML = MultiLabelClassifier().to(device)
    Optimizer = torch.optim.Adam(ML.parameters(), lr=ADAM_ALPHA, betas=ADAM_BETA)
    print('Training.')
    for epoch_index in range(N_EPOCHS):  # loop over the dataset multiple times
        for batch_index, batch_data in enumerate(train_loader):
            # get the inputs
            imgs, labels = batch_data
            imgs = Variable(imgs.type(FloatTensor)).to(device)
            labels = torch.stack(labels)
            labels = torch.transpose(labels, 0, 1)
            labels = Variable(labels.type(FloatTensor)).to(device)
            print("labels")
            print(labels)
            #labels = Variable(labels.to(device))
            # zero the parameter gradients
            Optimizer.zero_grad()

            # forward + backward + optimize
            outputs = ML(imgs)
            print("outputs")
            print(torch.sigmoid(outputs.data))
            loss = BCE_CRITERION(outputs, labels) 
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

    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    with torch.no_grad():
        for data in val_loader:
            imgs, labels = data
            imgs = Variable(imgs.type(FloatTensor)).to(device)
            labels = torch.stack(labels)
            labels = torch.transpose(labels, 0, 1)
            labels = Variable(labels.type(FloatTensor)).to(device)
            outputs = ML(imgs)
            outputs = torch.sigmoid(outputs.data)
            predicted = outputs
            true = torch.ones_like(predicted, device=device)
            false = torch.zeros_like(predicted, device=device)
            print(predicted)
            predicted = torch.where(predicted >= 0.5, true, false)
            #total_predicted += labels.size(0)
            #correct_predicted += (labels[predicted] == 1).sum().item()
            '''
            for batch_i in range (predicted.size(0)):
                total_predicted += 1
                correct_predicted += (labels[batch_i][predicted[batch_i]].item() == 1)
            '''
            true_negative_batch = ((predicted-1)*(labels-1)).sum()
            true_positive_batch = (predicted*labels).sum()
            false_positive_batch = predicted.sum() - true_positive_batch
            false_negative_batch = labels.sum() - true_positive_batch
            true_positives += true_positive_batch
            false_positives += false_positive_batch
            false_negatives += false_negative_batch
            true_negatives += true_negative_batch

    print('Accuracy of the CNN on Validation Set: %d %%' % (
        100 * (true_negatives+true_positives) / (true_negatives+true_positives+false_negatives+false_positives)))
    print('Precision: %d %%' % (
        100 * true_positives / (true_positives+false_positives)))
    print('Recall: %d %%' % (
        100 * true_positives / (true_positives+false_negatives)))

def get_class_weights():
    class_counts = ut.get_distribution()
    total = sum(class_counts)
    weights = [0]*len(class_counts)
    for i in range(len(class_counts)):
        x = class_counts[i]
        weights[i] = total/(1.0*x) if x > 0 else 0
    tensor_weights =  torch.tensor(weights, device=device)
    tensor_weights = tensor_weights / torch.sum(tensor_weights)
    print("weights:")
    print(tensor_weights)
    return tensor_weights

    
def main():
    if len(sys.argv) < 3:
        print('Usage: python3 train.py [MC|ML] [dataset_name]')
        exit()
    train_type = sys.argv[1]
    weights = get_class_weights()
    if train_type == "MC":
        train_multiclassifier(sys.argv[2], weights)
    elif train_type == "ML":
        train_multilabelclassifier(sys.argv[2])
    else:
        print('Usage: python3 train.py [MC|ML] [dataset_name]')
        exit()
    
if __name__ == '__main__':
    main()
