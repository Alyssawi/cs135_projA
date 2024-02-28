import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, roc_auc_score


def main(data_dir='data_reviews'):

    # overview the training data
    x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
    y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))

    tr_text_list = x_train_df.values.tolist()
    tr_y_list = y_train_df.values.tolist()

    reviews_list = [val[1] for val in tr_text_list]
    tr_y = np.hstack(np.array(tr_y_list))
    # TODO: add max_df, min_df
    vectorizer = CountVectorizer(strip_accents='ascii', lowercase=True, stop_words='english', max_df=0.95, min_df=2)
    features_count = vectorizer.fit_transform(reviews_list)
    # features = vectorizer.get_feature_names_out()
    # print(features)
    # print(features_count.toarray())


    param_grid = {
        'C' :  np.logspace(-9, 6, 31)
    }
    
    lr = sklearn.linear_model.LogisticRegression(solver='lbfgs')
    # Setup GridSearchCV with AUROC scoring
    auroc_scorer = make_scorer(roc_auc_score, needs_proba=True, greater_is_better=True)
    grid_search = GridSearchCV(lr, param_grid, scoring=auroc_scorer, cv=5, verbose=1)


    # Fit GridSearchCV
    grid_search.fit(features_count, tr_y)

    # Print the best parameters and AUROC score
    print("Best parameters:", grid_search.best_params_)
    print("Best AUROC score:", grid_search.best_score_)

    # Optional: Use the best model for further predictions or analysis
    best_model = grid_search.best_estimator_
    # Example: best_model.predict(new_data)





    
if __name__ == '__main__':
    main()
