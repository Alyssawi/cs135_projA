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
from stub.feature_extraction import dumb_feature_extractor2
extract_awesome_features = dumb_feature_extractor2

# TODO: please load your own trained models. Please check train_and_save_classifier.py to find 
# an example of training and saving a classiifer. 

with open('stub/classifier2.pkl', 'rb') as f:
    classifier2 = pickle.load(f)

# TODO: please provide your team name -- 20 chars maximum and no spaces please.  
teamname = "awesome-team"


