import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.linear_model


def main(data_dir='data_reviews'):

    # overview the training data
    x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
    y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))

    tr_text_list = x_train_df.values.tolist()
    tr_y_list = y_train_df.values.tolist()

    reviews_list = [val[1] for val in tr_text_list]

    # TODO: add max_df, min_df
    vectorizer = CountVectorizer(strip_accents='ascii', lowercase=True, stop_words='english')
    features_count = vectorizer.fit_transform(reviews_list)
    features = vectorizer.get_feature_names_out()
    # print(features)
    # print(feature_count.toarray())

    C_grid = np.logspace(-9, 6, 31)

    for C in C_grid:

        # create the logictic regressor

        # calc cross validation error

        # store cross validation error

    

    # pick mean CV error
    


    





    
if __name__ == '__main__':
    main()
