import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, roc_auc_score
import matplotlib.pyplot as plt
import pickle

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from transformers import BertTokenizer, BertConfig, BertModel, AdamW, get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification

from sklearn.base import BaseEstimator

class p2_estimator(BaseEstimator): 
    model: BertForSequenceClassification

    def __init__(self, hidden_size=12, num_hidden_layers=3):
        # create model 

        config = BertConfig.from_pretrained('bert-base-uncased',
                                    num_labels=2,
                                    hidden_size=hidden_size,
                                    num_hidden_layers=num_hidden_layers)
        self.model = BertForSequenceClassification(config)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def fit(self, reviews, tr_y):
        tokens = self.tokenizer(reviews, padding=True, truncation=True, return_tensors="pt")

        self.classes_ = np.unique(tr_y)  # Determine unique classes and set the classes_ attribute
   
        # just load in train dataloader for now 
        dataset = TensorDataset(tokens['input_ids'], tokens['attention_mask'], torch.tensor(tr_y))
        train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        ## train the model 
        optimizer = AdamW(self.model.parameters(), lr=1e-5)

        num_loops = 1 # TO DO: experiment with number of loops 
        for i in range(num_loops):
            self.model.train()
            for batch in train_dataloader:
                input_ids, attention_mask, labels = batch
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

    def predict_proba(self, data_list):
        reviews = [val[1].lower() for val in data_list]

        tokens = self.tokenizer(reviews, padding=True, truncation=True, return_tensors="pt")

        dataset = TensorDataset(tokens['input_ids'], tokens['attention_mask'])
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

        self.model.eval()

        probabilities = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids, attention_mask = batch 
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits 
                probabilities.extend(torch.nn.functional.softmax(logits, dim=1).cpu().numpy().tolist())
        
        return np.array(probabilities)
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        config = BertConfig.from_pretrained('bert-base-uncased',
                            num_labels=2,
                            hidden_size=self.hidden_size,
                            num_hidden_layers=self.num_hidden_layers)
        self.model = BertForSequenceClassification(config)
        return self

    def get_params(self, deep: bool = True) -> dict:
        return super().get_params(deep)
    