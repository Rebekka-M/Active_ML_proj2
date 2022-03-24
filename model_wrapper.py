from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
import torch
from vars import *
from lauges_tqdm import tqdm
import numpy as np


class ModelWrapper(BaseEstimator, ClassifierMixin):
    
    def __init__(self, model, lr=10**(-2.37391888), weight_decay=10**(-5.41253564), epochs=100):
        self.model = model
        self.lr = lr 
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.015, weight_decay=self.weight_decay)
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def fit(self, X, y): 
        # Training the model in one batch
        X = X.reshape((-1,1,28,28))
        print(X.shape)
        X = torch.tensor(X)
        y = torch.tensor(y)
        # TODO until convergence (min max epocs?)
        self.model.train()

        X = X.to(DEVICE)
        y = y.to(DEVICE)
        """
        # This is a test
        X = torch.ones((1, 1, 28, 28))
        y = torch.zeros((1, 10))
        y[0, 0] = 1

        self.optimizer.zero_grad()

        preds = self.model(X)
        probs = torch.nn.functional.softmax(preds, dim=1)
        loss = torch.sum(- torch.log(probs) * y)
        loss.backward()
        self.optimizer.step()
        print(loss, self.criterion(preds, torch.tensor([0])))
        """

        for epoch in range(self.epochs):#tqdm(range(self.epochs), freq=0.01):
            self.optimizer.zero_grad()

            preds = self.model(X)
            loss = self.criterion(preds, y)
            #probs = torch.nn.functional.softmax(preds, dim=1)
            #loss = torch.mean(- torch.log(probs) * y)
            loss.backward()
            #if epoch % 10 == 5:
            #    hej = [X[i, 0].detach().numpy() for i in range(10)]
            #    print(np.mean(torch.argmax(y, dim=1).detach().numpy() == torch.argmax(preds, dim=1).detach().numpy()))

            self.optimizer.step()


    def predict(self, X): 
        X = torch.tensor(X).unsqueeze(dim=1)
        preds = self.model(X)
        
        preds = torch.argmax(preds, dim=1)
        return preds.detach().numpy()
    
    def predict_proba(self, X): 
        X = torch.tensor(X).unsqueeze(dim=1)
        preds = self.model(X)
        
        preds = torch.nn.functional.softmax(preds)
        return preds.detach().numpy()
