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
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification

class p2_model:
    model: BertForSequenceClassification

    def __init__(self):
        ## read in data 
        x_train_df = pd.read_csv(os.path.join('data_reviews', 'x_train.csv'))
        y_train_df = pd.read_csv(os.path.join('data_reviews', 'y_train.csv'))

        tr_text_list = x_train_df.values.tolist()
        tr_y_list = y_train_df.values.tolist()

        tr_y = np.hstack(np.array(tr_y_list))

        reviews_list = [val[1].lower() for val in tr_text_list]

        ## create tokens 
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokens = tokenizer(reviews_list, padding=True, truncation=True, return_tensors="pt")

        ## create model 
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
 
        ## TO DO: Right now validation doesn't do anything, need to figure out cross validation and what that means
        # dataset = TensorDataset(tokens['input_ids'], tokens['attention_mask'], torch.tensor(tr_y))
        # train_size = int(0.8 * len(dataset))
        # val_size = len(dataset) - train_size
        # train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        # val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False)

        # just load in train dataloader for now 
        dataset = TensorDataset(tokens['input_ids'], tokens['attention_mask'], torch.tensor(tr_y))
        train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        ## train the model 
        optimizer = AdamW(self.model.parameters(), lr=1e-5)

        num_loops = 3 # TO DO: experiment with number of loops 
        for i in range(num_loops):
            self.model.train()
            for batch in train_dataloader:
                input_ids, attention_mask, labels = batch
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

    def predict_proba(self, tokens):
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
        return probabilities
