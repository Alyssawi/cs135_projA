import pickle

def extract_awesome_features(data_list):
    reviews_list = [val[1].lower() for val in data_list]
    with open('tokenizer.pkl','rb') as f:
        tokenizer = pickle.load(f)
    tokens = tokenizer(reviews_list, padding=True, truncation=True, return_tensors="pt")
    return tokens