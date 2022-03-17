import torch
import torch.nn
from model import CNN_class
import model
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
from typing import Tuple
from tqdm import tqdm
import model_wrapper

def load_data(batch_size = 500, data_workers = 1, oracle_train_size=10000, train_size=1000, noise=0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.Compose([transforms.ToTensor()])

    #training_set = datasets.MNIST(root="./data", split="byclass", train=True,  download=True, transform=transform)
    #test_set = datasets.MNIST(root="./data", split="byclass", train=False,  download=True, transform=transform)
    training_set = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
    test_set = datasets.MNIST(root="./data", train=False,  download=True, transform=transform)
    total_train = len(training_set) 
    
    oracle_train_percent = oracle_train_size/total_train
    train_percent = train_size/total_train 

    train_indices, _ = train_test_split(np.arange(total_train), train_size=oracle_train_percent+train_percent, 
                                        stratify=training_set.targets)
    training_set = data_utils.Subset(training_set, train_indices)
    oracle_indices, train_indices = train_test_split(np.arange(len(training_set)), 
                                             train_size=oracle_train_percent/(oracle_train_percent+train_percent), 
                                             stratify=training_set.targets)
    oracle_training_set = data_utils.Subset(training_set, oracle_indices)
    training_set = data_utils.Subset(training_set, train_indices)

    # Make stupid =) 
    
    train_dl = DataLoader(training_set, batch_size=batch_size, num_workers=data_workers)
    oracle_dl = DataLoader(oracle_training_set, batch_size=batch_size, num_workers=data_workers)
    test_dl = DataLoader(test_set, batch_size=batch_size, num_workers=data_workers)

    return train_dl, oracle_dl, test_dl


#training_dl, calibration_dl, validation_dl = load_data(batch_size=1000)
#transform = transforms.Compose([transforms.ToTensor()])
training_set = datasets.MNIST(root="./data", train=True,  download=True)
ds = torch.utils.data.Subset(training_set, range(1000))
X = np.array([np.array(i[0]) for i in ds])
y = np.array([np.array(i[1]) for i in ds])

# Best hypers from Active ML proj 1 
validation_accuracies = np.load('bayesian_optimization_accuracies_val.npy')
hyperparameters = np.load('bayesian_optimization_hyperparameters.npy')
hyperparameter = hyperparameters[np.argmax(validation_accuracies)]


test_model = model.CNN_class(*hyperparameter[2:], n_classes=10)

#X, y = next(iter(training_dl))

wrapped_model = model_wrapper.ModelWrapper(test_model, lr=10**(-hyperparameter[0]), weight_decay=10**(-hyperparameter[1]))

wrapped_model.fit(X, y)

wrapped_model(X)


"""
#for later use

oracle = CNN_class(*hyperparameter[2:])

model.train(oracle, training_dl, hyperparameter[0], hyperparameter[1], n_epochs=100)

oracle_preds = []
for X, _ in calibration_dl: 
    pred = oracle(X)
    oracle_preds.append(torch.argmax(pred, dim=1))
    
oracle_preds = torch.cat(torch.tensor(oracle_preds), dim=0)
""" 



    