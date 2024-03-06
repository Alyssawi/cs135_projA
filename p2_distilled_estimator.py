import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertTokenizer, DistilBertConfig, DistilBertForSequenceClassification, AdamW
from sklearn.base import BaseEstimator

class p2_estimator(BaseEstimator):
    model: DistilBertForSequenceClassification

    def __init__(self, hidden_size=768, num_hidden_layers=6, model=None):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        if model:
            self.model = model
            return
        # Create model with DistilBert configuration
        config = DistilBertConfig.from_pretrained('distilbert-base-uncased',
                                                  num_labels=2,
                                                  dim=hidden_size,
                                                  n_layers=num_hidden_layers)
        self.model = DistilBertForSequenceClassification(config)



    def fit(self, reviews, tr_y):
        tokens = self.tokenizer(reviews, padding=True, truncation=True, return_tensors="pt")
        
        self.classes_ = np.unique(tr_y)  # Set the classes_ attribute for sklearn compatibility

        dataset = TensorDataset(tokens['input_ids'], tokens['attention_mask'], torch.tensor(tr_y, dtype=torch.long))
        train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        optimizer = AdamW(self.model.parameters(), lr=1e-5)

        num_epochs = 3  # Experiment with the number of epochs
        for _ in range(num_epochs):
            self.model.train()
            for batch in train_dataloader:
                input_ids, attention_mask, labels = batch
                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
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
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probabilities.extend(torch.nn.functional.softmax(logits, dim=1).cpu().numpy().tolist())

        return np.array(probabilities)

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        config = DistilBertConfig.from_pretrained('distilbert-base-uncased',
                                                  num_labels=2,
                                                  dim=self.hidden_size,
                                                  n_layers=self.num_hidden_layers)
        self.model = DistilBertForSequenceClassification(config)
        return self

    def get_params(self, deep=True) -> dict:
        return {"hidden_size": self.hidden_size, "num_hidden_layers": self.num_hidden_layers}