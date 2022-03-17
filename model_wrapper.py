from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
import torch
from vars import *


class ModelWrapper(BaseEstimator, ClassifierMixin):
    
    def __init__(self, model, lr=10**(-2.37391888), weight_decay=10**(-5.41253564), epochs=10):
        self.model = model
        self.lr = lr 
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def fit(self, X, y): 
        # Training the model in one batch
        X = X.reshape((-1,1,28,28))
        print(X.shape)
        X = torch.tensor(X)
        y = torch.tensor(y)
        # TODO until convergence (min max epocs?)
        self.model.train()
        for epoch in range(self.epochs):
            X = torch.permute(X, (0, 1, 3, 2))
            self.optimizer.zero_grad()
            X = X.to(DEVICE)
            y = y.to(DEVICE)


            preds = self.model(X)
            loss = self.criterion(preds, y)
            loss.backward()
            self.optimizer.step()
            
    def predict(self, X): 
        X = torch.tensor(X)
        preds = self.model(X)
        
        preds = torch.argmax(preds, dim=1)
        return preds.detach().numpy()
    
    def predict_proba(self, X): 
        X = torch.tensor(X)
        preds = self.model(X)
        
        preds = torch.nn.functional.softmax(preds)
        return preds.detach().numpy()
 
    