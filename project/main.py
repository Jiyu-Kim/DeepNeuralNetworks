import numpy as np
import pandas as pd

from dataset import SSL_Dataset
from model import CNN
from utils import save_prediction, plot_accuracy

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models 
import torch.nn as nn  
from time import time 

"""
Please use this code as a guideline. 
Feel free to create your own code for training, testing, ... etc.
But for creating "submission.csv" file, utilizing this code is highly recommended.
"""

class Trainer:
    def __init__(self, model, device, weight_path, model_name, patience, momentum, weight_decay, learning_rate, num_epoch, print_every, train_batch, test_batch):

        self.patience = patience
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lr = learning_rate
        self.num_epoch = num_epoch
        self.print_every = print_every
        self.train_batch = train_batch
        self.test_batch = test_batch
    
        self.best_acc = 0
        self.best_epoch = 0
        self.crnt_epoch = 0 
        self.endure = 0 
        self.stop_flag = False
        self.num_class = 10
        self.device = device
        self.weight_path = weight_path
        self.model_name = model_name

        self.model = model
        self.loss_fn = nn.CrossEntropyLoss() 
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)

        self.train_acc = []
        self.valid_acc = []
        
    # test
    def _test(self, mode, data_loader):
       
        test_preds = []
        self.model.eval()
        correct = 0
        total = 0
        
        if mode == 'Valid':
            with torch.no_grad():
                for batch_data in data_loader: 
                    batch_x, batch_y = batch_data 
                    inputs, targets = batch_x.to(self.device), batch_y.to(self.device)

                    outputs = self.model(inputs)
                    _, predicted = torch.max(outputs, 1) 

                    total += targets.size(0)
                    correct += predicted.eq(targets).cpu().sum().item()
                    if self.device == 'cuda':
                        test_preds += predicted.detach().cpu().numpy().tolist()
                    else:
                        test_preds += predicted.detach().numpy().tolist()

            total_acc = correct / total

            print("| \033[31m%s Epoch #%d\t Accuracy: %.2f%%\033[0m" %(mode, self.crnt_epoch+1, 100.*total_acc))
            if self.crnt_epoch % self.print_every == 0 : 
                self.valid_acc.append(total_acc)
            if self.best_acc < total_acc:
                print('| \033[32mBest Accuracy updated (%.2f => %.2f)\033[0m\n' % (100.*self.best_acc, 100.*total_acc))
                self.best_acc = total_acc
                self.best_epoch = self.crnt_epoch
                self.endure = 0
                # Save best model
                torch.save(self.model.state_dict(), self.weight_path+self.model_name)
            else:
                self.endure += 1
                print(f'| Endure {self.endure} out of {self.patience}\n')
                if self.endure >= self.patience:
                    print('Early stop triggered...!')
                    self.stop_flag = True

        if mode == 'Test':
            print('Predicting Starts...')
            with torch.no_grad():
                for batch_data in data_loader: 
                    batch_x = batch_data 
                    inputs = batch_x.to(self.device)

                    outputs = self.model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    if self.device == 'cuda':
                        test_preds += predicted.detach().cpu().numpy().tolist()
                    else:
                        test_preds += predicted.detach().numpy().tolist()

            return test_preds, self.crnt_epoch, self.train_acc, self.valid_acc

    def _train(self, labeled_trainloader, labeled_validloader, unlabeled_trainloader=None):
        self.model.train()
        print('Training Starts...')
        total = 0
        correct = 0
        whole_data = []
        whole_targets = []

        #training with labeled train datset
        for epoch in range(self.num_epoch):
            self.crnt_epoch = epoch 
            for batch_data in labeled_trainloader:
                batch_x, batch_y = batch_data
      
                for data in batch_x:
                  whole_data.append(data.numpy())
                for target in batch_y:
                  whole_targets.append(int(target))

                batch_size = batch_x.size(0) 
                batch_y = torch.zeros(batch_size, self.num_class).scatter_(1, batch_y.view(-1,1), 1) 
                inputs_l, targets_l = batch_x.to(self.device), batch_y.long().to(self.device) 
            
                outputs = self.model(inputs_l)
                loss = self.loss_fn(outputs,torch.argmax(targets_l, dim=-1))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print(f"Epoch: {epoch+1}, Loss:{loss:.4f}")

                _, predicted = torch.max(outputs, 1)

                total += targets_l.size(0)
                correct += predicted.eq(torch.argmax(targets_l, dim=-1)).cpu().sum().item()

            total_acc = correct / total
            if self.crnt_epoch % self.print_every == 0 : 
                self.train_acc.append(total_acc)
            print("\n| \033[31mTrain Epoch #%d\t Accuracy: %.2f%%\033[0m" %(self.crnt_epoch+1, 100.*total_acc))
            pred = self._test("Valid", labeled_validloader)
            if self.stop_flag : break
        print('Training Finished...!!')

        #if train_data_mode is 'semi_supervised, procceed pseudo labeling
        if unlabeled_trainloader != None:
            print('\n\nPseudo labeling Starts...')
            with torch.no_grad():
                for batch_data in unlabeled_trainloader: 
                    batch_x, batch_y = batch_data 
                    inputs, targets = batch_x.to(self.device), batch_y.to(self.device)

                    threshold = 0.5
                    outputs = self.model(inputs)
                    for idx, output in enumerate(outputs):
                      output = F.softmax(output, dim=0)
                      max_value, predicted_y = torch.max(output, 0)
                      if max_value >= threshold:
                        whole_data.append(batch_x[idx].numpy())
                        whole_targets.append(int(predicted_y))
                    
            print("Pseudo labeling Finished...!!")
        
        
            #training with labeled train dataset and pseudo labeled train dataset
            print("\n\nTraining2 Starts...")
            print("# Tran Data: ", len(whole_data))
            for epoch in range(10):
              self.crnt_epoch = epoch
              for i in range(len(whole_data)//self.train_batch):
                start = i*self.train_batch
                end = start + self.train_batch
                batch_x = torch.tensor(whole_data[start:end])
                batch_y = torch.tensor(whole_targets[start:end])

                batch_size = len(batch_x)
                batch_y = torch.zeros(batch_size, self.num_class).scatter_(1, batch_y.view(-1,1), 1) 
                inputs_l, targets_l = batch_x.to(self.device), batch_y.long().to(self.device) 
                
                outputs = self.model(inputs_l)
                loss = self.loss_fn(outputs,torch.argmax(targets_l, dim=-1))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print(f"Epoch: {epoch+1}, Loss:{loss:.4f}")

                _, predicted = torch.max(outputs, 1)

                total += targets_l.size(0)
                correct += predicted.eq(torch.argmax(targets_l, dim=-1)).cpu().sum().item()

              total_acc = correct / total
              if self.crnt_epoch % self.print_every == 0 : 
                self.train_acc.append(total_acc)
              print("\n| \033[31mTrain Epoch #%d\t Accuracy: %.2f%%\033[0m" %(self.crnt_epoch+1, 100.*total_acc))
              
              
              pred = self._test("Valid", labeled_validloader)
              if self.stop_flag : break
              print('Training2 Finished...!!')


def main():

    #################### EDIT HERE ####################
    """
    You can change any values of hyper-parameter below.
    *test_only: If this parameter is True, you can just test with a model that already exists without training step. 
    (for your time saving..!) 
    """
    random_seed=0

    patience = 7
    momentum = 0.9
    weight_decay = 5e-4
    learning_rate = 0.001

    num_epoch = 10
    print_every = 1
    train_batch = 64
    test_batch = 1000
    valid_ratio = 0.1

    model_name = 'my_model.p'
    #train_data_mode = 'labeled_train'   -> supervised learning
    #train_data_mode = 'semi_supervised' -> semi-supervised learning
    train_data_mode = 'semi_supervised'

    test_only = False
    ###################################################

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    if torch.cuda.is_available():
        device = 'cuda' 
        torch.cuda.manual_seed_all(random_seed) 
        torch.backends.cudnn.deterministic = True
    else :
        device = 'cpu'
    
    weight_path = './best_model/'
    
    #transform image data by using Resize, ToTensor, Normalize
    transform0 = transforms.Compose(
                [transforms.Resize(size=(32,32)), 
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    #transform image data by using Resize, RandomHorizontalFlip, ToTensor, Normalize
    transform1 = transforms.Compose(
                [transforms.Resize(size=(32, 32)),
                transforms.RandomHorizontalFlip(), 
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    #transform image data by using Resize, ColorJitter, ToTensor, Normalize
    transform2 = transforms.Compose(
                [transforms.Resize(size=(32,32)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    #transform image data by using Resize, RandomRotation, ToTensor, Normalize
    transform3 = transforms.Compose(
              [transforms.Resize(size=(32,32)), 
              transforms.transforms.RandomRotation(10),
              transforms.ToTensor(),
              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    #transform image data by using Resize, RandomGrayscale, ToTensor, Normalize
    transform4 = transforms.Compose(
                [transforms.Resize(size=(32, 32)),
                transforms.RandomGrayscale(p=1), 
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    train_examples0 = SSL_Dataset(root='../', transform=transform0, mode='labeled_train') #original
    train_examples1 = SSL_Dataset(root='../', transform=transform1, mode='labeled_train') #flip
    train_examples2 = SSL_Dataset(root='../', transform=transform2, mode='labeled_train') #jitter color
    train_examples3 = SSL_Dataset(root='../', transform=transform3, mode='labeled_train') #rotate
    train_examples4 = SSL_Dataset(root='../', transform=transform4, mode='labeled_train') #grayscale
    
    #data augmentation(merge all transformed data)
    train_examples = train_examples0 + train_examples1 + train_examples2 + train_examples3 + train_examples4

    num_train = len(train_examples)
    num_valid = int(num_train * valid_ratio)

    train_labeled_dataset, valid_labeled_dataset = torch.utils.data.random_split(train_examples, [num_train-num_valid, num_valid])
    
    test_labeled_dataset = SSL_Dataset(root='../',transform=transform0, mode="test") 
    test_idx = test_labeled_dataset.idx

    labeled_trainloader = DataLoader(train_labeled_dataset, batch_size=train_batch, shuffle=True)
    labeled_validloader = DataLoader(valid_labeled_dataset, batch_size=test_batch, shuffle=False)
    labeled_testloader = DataLoader(test_labeled_dataset, batch_size=test_batch, shuffle=False)

    #Load unlabeled train dataset
    train_unlabeled_dataset0 = SSL_Dataset(root='../',transform=transform0, mode="unlabeled_train") #original
    train_unlabeled_dataset1 = SSL_Dataset(root='../',transform=transform1, mode="unlabeled_train") #flip
    train_unlabeled_dataset2 = SSL_Dataset(root='../',transform=transform2, mode="unlabeled_train") #jitter color
    train_unlabeled_dataset3 = SSL_Dataset(root='../',transform=transform3, mode="unlabeled_train") #rotate
    train_unlabeled_dataset4 = SSL_Dataset(root='../',transform=transform4, mode="unlabeled_train") #grayscale
    
    #data augmentation(merge all transformed data)
    train_unlabeled_dataset = train_unlabeled_dataset0 + train_unlabeled_dataset1 + train_unlabeled_dataset2 + train_unlabeled_dataset3 + train_unlabeled_dataset4

    unlabeled_trainloader = DataLoader(train_unlabeled_dataset, batch_size=train_batch, shuffle=False)

    #################### EDIT HERE ####################
    """
    If you want to try with ResNet18 model :
    """
    #model = models.resnet18().to(device)
    #model.fc = nn.Linear(model.fc.in_features, 10).to(device)
    
    """
    If you want to try with custom model (CNN in model.py) :
    """
    model = CNN().to(device)

    """
    Or you can try any model! :
    """
    ###################################################
    
    trainer = Trainer(model, device, weight_path, model_name, patience, momentum, weight_decay, learning_rate, num_epoch, print_every, train_batch, test_batch)

    if test_only == False:
        if train_data_mode == 'labeled_train': # supervised training
            print(f"# Train data: {len(train_labeled_dataset)}, # Valid data: {len(valid_labeled_dataset)}")
            train_start = time()
            trainer._train(labeled_trainloader, labeled_validloader)
            train_elapsed = time() - train_start
            print('Train Time: %.4f\n' % train_elapsed)
        else: # semi-supervised training
            print(f"# Train data: {len(train_labeled_dataset)}, # Valid data: {len(valid_labeled_dataset)}")
            train_start = time()
            trainer._train(labeled_trainloader, labeled_validloader, unlabeled_trainloader)
            train_elapsed = time() - train_start
            print('Train Time: %.4f\n' % train_elapsed)

    print(f"# Test data: {len(test_labeled_dataset)}")
    model.load_state_dict(torch.load(weight_path+model_name))
    pred, num_epochs, train_acc, valid_acc = trainer._test("Test", labeled_testloader)
    save_prediction(weight_path, pred, test_idx)
    plot_accuracy(print_every, weight_path, num_epochs, train_acc, valid_acc)

if __name__ == "__main__":
    main()