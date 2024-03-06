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


# TODO: please load your own trained models. Please check train_and_save_classifier.py to find 
# an example of training and saving a classiifer. 

# with open('bert2.pkl', 'rb') as f:
#     classifier2 = pickle.load(f)

classifier = 'bert2.pkl'

reconstructed_state_dict = {}

for i in range(8):
    # Load the chunk
    with open(f'{classifier}_part_{i}.pkl', 'rb') as f:
        chunk_dict = pickle.load(f)
    
    # Update the reconstructed state dictionary with the chunk
    reconstructed_state_dict.update(chunk_dict)



# from transformers import BertForSequenceClassification
from transformers import DistilBertForSequenceClassification

# Now, load the reconstructed state dict back into the model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
model.load_state_dict(reconstructed_state_dict)


from p2_distilled_estimator import p2_estimator
classifier2 = p2_estimator(model=model)


# TODO: please provide your team name -- 20 chars maximum and no spaces please.  
teamname = "byte-sized"


