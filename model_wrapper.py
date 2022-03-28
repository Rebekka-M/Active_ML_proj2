from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
import torch
from vars import *
from lauges_tqdm import tqdm
import numpy as np
from model import CNN_class


class ModelWrapper(BaseEstimator, ClassifierMixin):
    """
    ModelWrapper that prepares the model to be used in the active learning framework
    (If change to EMNIST use the lr=10**(-2.37391888) from our previous project
    """

    def __init__(self, seed=1, lr=0.015, weight_decay=10**(-5.41253564), epochs=35):
        self.model = CNN_class(4, 4, n_classes=10)
        self.lr = lr 
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def fit(self, X, y, printing=False):
        """
        Fits the model to the known data set, runs 35 epochs and assumes that one batch is enough
        TODO train until convergence instead?

        Parameters
        ----------
        X : (batch_size, 28, 28) array that contains images
        y : (batch_size, label_space(10)) array of probabilities of each class
        """
        # Reset model 
        self.model.__init__(4, 4, n_classes=10)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        # Takes inputs and outputs to tensors and enables training
        X = X.reshape((-1, 1, 28, 28))
        X = torch.tensor(X)
        y = torch.tensor(y)
        self.model.train()

        X = X.to(DEVICE)
        y = y.to(DEVICE)
        """
        for epoch in range(self.epochs):
            #Training loop
            self.optimizer.zero_grad()

            preds = self.model(X)
            # loss = torch.sum(-torch.log(torch.nn.functional.softmax(preds))*y)
            loss = self.criterion(preds, y)
            loss.backward()

            if epoch % 10 == 5:#and printing:
                print(f"Accuracy on {epoch} : {np.mean(torch.argmax(y, dim=1).detach().numpy() == torch.argmax(preds, dim=1).detach().numpy())}")

            self.optimizer.step()
            
        """
        acc_prev = 0 
        acc_curr = 0
        epoch = 0
        while epoch <= self.epochs or abs(acc_curr - acc_prev)/acc_prev >= 0.001: #for epoch in range(self.epochs):
            epoch += 1
            acc_prev = acc_curr
            
            #Training loop
            self.optimizer.zero_grad()

            preds = self.model(X)
            # loss = torch.sum(-torch.log(torch.nn.functional.softmax(preds))*y)
            loss = self.criterion(preds, y)
            loss.backward()
            self.optimizer.step()

            acc_curr = np.mean(torch.argmax(y, dim=1).detach().numpy() == torch.argmax(preds, dim=1).detach().numpy())
            if epoch % 10 == 5 and printing:
                print(f"Accuracy on {epoch} : {acc_curr}, {acc_prev}, {abs(acc_curr - acc_prev)/acc_prev}")

            
        
        #print(acc_curr, acc_prev)
        


    def predict(self, X):
        """
        Predicts which classes a list of images are from

        Parameters
        ----------
        X : (batch_size, 28, 28) array that contains images

        Returns
        -------
        List of shape (batch_size) of labels
        """
        X = torch.tensor(X).unsqueeze(dim=1)
        preds = self.model(X)
        
        preds = torch.argmax(preds, dim=1)
        return preds.detach().numpy()
    
    def predict_proba(self, X):
        """
        Predict with which pseudo probabilities the model assigns to each input image

        Parameters
        ----------
        X : (batch_size, 28, 28) array that contains images

        Returns
        -------
        List of shape (batch_size, class_size) of probabilities of each image in the batch.
        """
        X = torch.tensor(X).unsqueeze(dim=1)
        preds = self.model(X)
        
        preds = torch.nn.functional.softmax(preds, dim=1)
        return preds.detach().numpy()
