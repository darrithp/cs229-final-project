
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import numpy as np
from dataset import MovieDataset


CRITERION = nn.CrossEntropyLoss()
N_EPOCHS = 10
ADAM_ALPHA = 0.001
ADAM_BETA = (0.9, 0.999)
PRINT_INTERVAL = 5

def main():

    train_data = MovieDataSet("data/temp/dataset1000/traindata.pkl")
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    # Instantiate the models
    CNN = ConvolutionalNeuralNet()
    Optimizer = torch.optim.Adam(CNN.parameters(), lr=ADAM_ALPHA, betas=ADAM_BETA)
    print('Training.')
    for epoch_index in range(N_EPOCHS):  # loop over the dataset multiple times
        for batch_index, batch_data in enumerate(train_loader):
            # get the inputs
            imgs, labels = batch_data

            # zero the parameter gradients
            Optimizer.zero_grad()

            # forward + backward + optimize
            outputs = CNN(imgs)
            loss = CRITERION(outputs, labels)
            loss.backward()
            Optimizer.step()

            # Print Loss
            if epoch % PRINT_INTERVAL == 0 and not batch_index:    # print every 2000 mini-batches
                print('Epoch: %d, \tLoss: %.3f' % (epoch_index, loss))

    print('Finished Training. Testing on Validation Set.')


    #Validation Set
    val_data = MovieDataSet("data/temp/dataset1000/valdata.pkl")
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

    correct_predicted = 0
    total_predicted = 0
    with torch.no_grad():
        for data in val_loader:
            imgs, labels = data
            outputs = CNN(imgs)
            _, predicted = torch.max(outputs.data, 1)
            total_predicted += labels.size(0)
            correct_predicted += (predicted == labels).sum().item()

    print('Accuracy of the CNN on Validation Set: %d %%' % (
        100 * correct / total))

if __name__ == '__main__':
    main()
