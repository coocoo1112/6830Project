from sklearn.linear_model import LinearRegression
import numpy as np
import random
from data_utils import dataset_iter, save_results, get_size_vocab, get_shape_vector, get_word_vector_sentence, get_offset
import math
import argparse
from sklearn.metrics import mean_squared_error 
from query_encoding import histogram_encoding, join_matrix
from gensim.models import Word2Vec
from sklearn.model_selection import KFold


def load_data(csv_name, query_encoding = False, word=False):
    data = []
    queries = []
    

    for row in dataset_iter(csv_name):
        offset = get_offset(row["query"])
        x = float(row["plan"]["Plan"]["Total Cost"]) if not query_encoding else np.concatenate([histogram_encoding(row["plan"]), join_matrix(row["plan"])]) if not word else [i.strip().replace("'","") for i in row["query"][offset+1:].split()[:-1]]
        if word:
            x = [i.strip() for i in x]
        pair = x, float(row["execution_time (ms)"]) 
        data.append(pair)
    return data


def train(data, query_encoding=False, word=False):
    """
    :data list of x, y where x is the encoding of the query and y is the run time
    :offset is our train/test split float
    :query_encoding if we are doing neo 
    :word if we are doing word2vec query representation. This flag being true implies query_encoding is also true
    """
    model = LinearRegression(fit_intercept=True, normalize=True)
    random.shuffle(data)
    
    X = [np.array([i]) if not query_encoding else i for i, _ in data]
        
    if word:
        size = math.floor(.1*get_size_vocab(X))
        # print(X)
        nlp = Word2Vec(X, window=20, workers=16, vector_size = size)
        shape = get_shape_vector(X, nlp)
        X_ = [get_word_vector_sentence(sentence, nlp, shape) for sentence in X]
        X = X_
       
    Y = [np.array([i]) for _, i in data]

    kf = KFold(n_splits=args.k, shuffle=True)
    results = []
    i = 1
    for train_idx, test_idx in kf.split(X):
        print(len(train_idx), len(test_idx))
        trainX = np.array([X[i] for i in train_idx])
        testX = np.array([X[i] for i in test_idx])
        trainY = np.array([Y[i] for i in train_idx])
        trueY = np.array([Y[i] for i in test_idx])
        model.fit(trainX, trainY)
        predictions = [model.predict(np.array([test])) for test in testX]
        predY = [pred.flatten()[0] for pred in predictions]
        rmse = mean_squared_error(trueY, predY, squared=False)
        print(f"Trial {i} RMSE: {rmse}")
        i += 1
        results.append(rmse)
    save_results(args.c, results, args.v, args.s, args.e, args.w, True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # version of data to run regression on
    parser.add_argument("--v",type=str, default="data_v26.csv", required=False)
    # query encoding or no
    parser.add_argument("--e",type=lambda x: True if x=="True" else False, default=False, required=False)
    # word2vec encoding or no
    parser.add_argument("--w",type=lambda x: True if x=="True" else False, default=False, required=False)
    # how many kfolds to do
    parser.add_argument("--k",type=lambda x: int, default=15, required=False)
    # csv name to save results to
    parser.add_argument("--c",type=str, default="result_v0.csv", required=False)
    # instance size
    parser.add_argument("--s",type=int, default=2, required=False)

    args = parser.parse_args()
    train(load_data(args.v, args.e, args.w), args.e, args.w)
