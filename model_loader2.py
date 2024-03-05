# This file is suppose to provide an interface between your implementation and the autograder. 
# In reality, the autograder should be a production system. This file provide an interface for 
# the system to call your classifier. 

# Ideally you could bundle your feature extractor and classifier in a single python function, 
# which takes a raw instance (a list of two strings) and predict a probability. 

# Here we use a simpler interface and provide the feature extractor and the classifer separately. 
# For Problem 2, you are supposed to provide
# * a feature extraction function `extract_awesome_features`, and  
# * a sklearn classifier, `classifier2`, whose `predict_proba` will be called.
# * your team name

# These two python objects will be imported by the `test_classifier_before_submission` autograder.

import pickle

# TODO: please replace the line below with your implementations. The line below is just an 
# example. 
from problem2_extract import extract_awesome_features

from p2_model import p2_model
from transformers import BertForSequenceClassification


# TODO: please load your own trained models. Please check train_and_save_classifier.py to find 
# an example of training and saving a classiifer. 

# with open('classifier2.pkl', 'rb') as f:
#     classifier2 = pickle.load(f)

reconstructed_state_dict = {}

for i in range(15):
    # Load the chunk
    with open(f'model_part_{i}.pkl', 'rb') as f:
        chunk_dict = pickle.load(f)
    
    # Update the reconstructed state dictionary with the chunk
    reconstructed_state_dict.update(chunk_dict)

# Now, load the reconstructed state dict back into the model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.load_state_dict(reconstructed_state_dict)
classifier2 = p2_model(model)


# TODO: please provide your team name -- 20 chars maximum and no spaces please.  
teamname = "byte-sized"


