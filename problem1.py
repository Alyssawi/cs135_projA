import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, roc_auc_score
import pickle


# overview the training data
x_train_df = pd.read_csv(os.path.join('data_reviews', 'x_train.csv'))
y_train_df = pd.read_csv(os.path.join('data_reviews', 'y_train.csv'))

tr_text_list = x_train_df.values.tolist()
tr_y_list = y_train_df.values.tolist()
# TODO: add max_df, min_df
vectorizer = CountVectorizer(strip_accents='ascii', lowercase=True, stop_words='english')
reviews_list = [val[1] for val in tr_text_list]
vectorizer = vectorizer.fit(reviews_list)

def extract_BoW_features(data_list):
    reviews_list = [val[1] for val in data_list]
    features_count = vectorizer.transform(reviews_list)
    return features_count

def main(data_dir='data_reviews'):

    tr_y = np.hstack(np.array(tr_y_list))
    features_count = extract_BoW_features(tr_text_list)
    # features = vectorizer.get_feature_names_out()
    # print(features)
    # print(features_count.toarray())

    param_grid = {
        'C' : np.logspace(-9, 6, 31)
    }
    
    lr = sklearn.linear_model.LogisticRegression(solver='liblinear')
    # Setup GridSearchCV with AUROC scoring
    auroc_scorer = make_scorer(roc_auc_score, needs_proba=True, greater_is_better=True)
    grid_search = GridSearchCV(lr, param_grid, scoring=auroc_scorer, cv=5)
    # grid_search = GridSearchCV(lr, param_grid, scoring=auroc_scorer, cv=5, verbose=1)


    # Fit GridSearchCV
    grid_search.fit(features_count, tr_y)

    # Print the best parameters and AUROC score
    # print("Best parameters:", grid_search.best_params_)
    # print("Best AUROC score:", grid_search.best_score_)

    # Optional: Use the best model for further predictions or analysis
    best_model = grid_search.best_estimator_

    with open('classifier1.pkl','wb') as f:
        pickle.dump(best_model,f)




    
if __name__ == '__main__':
    main()
