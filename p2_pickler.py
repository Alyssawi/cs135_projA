

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

from p2_model import p2_model 

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# with open('tokenizer.pkl','wb') as f:
#     pickle.dump(tokenizer,f)

classifier = 'bert2.pkl'

with open(classifier, 'rb') as fin:
    model = pickle.load(fin).model
    # classifier2 = pickle.load(f)
# model = p2_model()
    
# with open('classifier2.pkl','wb') as f:
    # pickle.dump(model,f)

    model_state_dict = model.state_dict()

    # Convert the state dict to a list of tuples for easier chunking
    items = list(model_state_dict.items())

    # Define the number of parts you want to split your model into
    num_parts = 8

    # Calculate the size of each chunk
    chunk_size = len(items) // num_parts

    for i in range(num_parts):
        # Determine the start and end of the current chunk
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_parts - 1 else len(items)
        
        # Extract the chunk from the items list
        chunk = items[start:end]
        
        # Convert the chunk back to a dict
        chunk_dict = dict(chunk)
        
        # Save the chunk using pickle
        with open(f'{classifier}_part_{i}.pkl', 'wb') as f:
            pickle.dump(chunk_dict, f)