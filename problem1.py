import pickle


def extract_BoW_features(data_list):
    with open('vectorizer1.pkl','rb') as f:
        vectorizer = pickle.load(f)
    reviews_list = [val[1] for val in data_list]
    features_count = vectorizer.transform(reviews_list)
    return features_count

