from sklearn.linear_model import LinearRegression
import numpy as np
import random
from data_utils import dataset_iter
import math
import argparse
from sklearn.metrics import mean_squared_error 
from query_encoding import histogram_encoding, join_matrix
from gensim.models import Word2Vec, FastText


def load_data(csv_name, query_encoding = False, word=False):
    print(query_encoding)
    data = []
    queries = []
    for row in dataset_iter(csv_name):
        x = float(row["plan"]["Plan"]["Total Cost"]) if not query_encoding else np.concatenate([histogram_encoding(row["plan"]), join_matrix(row["plan"])]) if not word else [i.strip().replace("'","") for i in row["query"].split()[:-1]][7:]
        if word:
            x = [i.strip() for i in x]
        pair = x, float(row["execution_time (ms)"]) 
        data.append(pair)
    return data


def train(data, offset, query_encoding=False, word=False):
    model = LinearRegression(fit_intercept=True, normalize=True)
    random.shuffle(data)
    
    X = [np.array([i]) if not query_encoding else i for i, _ in data]

    if word:
        # print("YES")
        # print(X)
        nlp = Word2Vec(X, window=20, workers=16)
        shape = None
        for sentence in X:
            for word in sentence:
                if word in sentence:
                    shape = nlp.wv[word].shape
                    break
            break
        X_ = []
        for sentence in X:
            vector = np.copy(nlp.wv[sentence[0]]) if sentence[0] in nlp.wv else np.zeros(shape)
            for word in sentence[1:]:
                if word not in nlp.wv:
                    vector += np.zeros(shape)
                else:
                    vector += np.copy(nlp.wv[word])
            X_.append(vector)
            X = X_
        
    Y = [np.array([i]) for _, i in data]
    trainOffset = math.floor(offset*len(X))

    trainX = np.array(X[:trainOffset])
    trainY = np.array(Y[:trainOffset])
    testX = np.array(X[trainOffset:])
    trueY = np.array(Y[trainOffset:])
    model.fit(trainX, trainY)
    predictions = [model.predict(np.array([test])) for test in testX]
    predY = [pred.flatten()[0] for pred in predictions]
    
    print(f"RMSE: {mean_squared_error(trueY, predY, squared=False)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--o",type=float, default=0.75, required=False)
    parser.add_argument("--v",type=str, default="data_v26.csv", required=False)
    parser.add_argument("--e",type=lambda x: True if x=="True" else False, default=False, required=False)
    parser.add_argument("--w",type=lambda x: True if x=="True" else False, default=False, required=False)

    args = parser.parse_args()
    train(load_data(args.v, args.e, args.w), args.o, args.e, args.w)
