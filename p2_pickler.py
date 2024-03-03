

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

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
with open('tokenizer.pkl','wb') as f:
    pickle.dump(tokenizer,f)

model = p2_model()
with open('classifier2.pkl','wb') as f:
    pickle.dump(model,f)