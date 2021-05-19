from csv import DictReader
import model
import random
from data_utils import dataset_iter, save_results
from sklearn.metrics import mean_squared_error 
import numpy as np
import sys
from gensim.models import Word2Vec
from sklearn.model_selection import KFold
import argparse

class BaoTrainingException(Exception):
    pass


def train(verbose=True):
    query_encoding = True
    x = []
    y = []
    pairs = []
    tx = []
    ty =[]
    for row in dataset_iter(args.v):
        if args.e:
            pairs.append((row["query"], row["plan"], row["execution_time (ms)"]))
        else:
            pairs.append((row["plan"], row["execution_time (ms)"]))

            
    random.shuffle(pairs)
    train_percent = 0.75
    train_amount = int(len(pairs) * train_percent)
    print(train_amount)
    if args.e:
        x = [(p, q) for q, p, r in pairs[:train_amount]]
        y = [float(r) for q, p, r in pairs[:train_amount]]
        tx = [(p, q) for q, p, r in pairs[train_amount:]]
        ty = [float(r) for q, p, r in pairs[train_amount:]]
    else:
        x = [i for i, _ in pairs[:train_amount]]
        y = [float(i) for _, i in pairs[:train_amount]]
        tx = [i for i, _ in pairs[train_amount:]]
        ty = [float(i) for _, i in pairs[train_amount:]]  
   
    reg = model.BaoRegression(have_cache_data=False, verbose=verbose, neo=args.e, word2vec=args.w)
    #print(y)
    # print("1")
    # print(np.mean(ty))
    # print(np.std(ty))
    # print("2")
    kf = KFold(n_splits=args.k, shuffle=True)
    results = []
    i = 1
    for train_idx, test_idx in kf.split(x):
        print(len(train_idx), len(test_idx))
        trainX = np.array([x[i] for i in train_idx])
        testX = np.array([x[i] for i in test_idx])
        trainY = np.array([y[i] for i in train_idx])
        trueY = np.array([y[i] for i in test_idx])
        reg.fit(trainX, trainY)

        predictions = model.predict(testX)
        predY = [pred[0] for pred in predictions]
        rmse = mean_squared_error(trueY, predY, squared=False)
        print(f"Trial {i} RMSE: {rmse}")
        i += 1
        results.append(rmse)
    save_results(args.c, results, args.v, args.s, args.e, args.w, False)
    # result = reg.predict(tx)
    # ty = np.array(ty).astype(np.float32)
    # # print(ty)
    # # print(result)

    # res = np.array([])
    # for i in range(len(result)):
    #     # print("test")
    #     res = np.append(res, result[i])
    #print(res)
    # for i in range(len(result)):
    #     print(ty[i], res[i])
    #     print(type(ty[i]), type(res[i]))
    #flat_result = result.flatten()
    #sub = np.subtract(flat_result, ty)
    #print(np.sort(sub))


    # print(f"RMSE: {mean_squared_error(ty, res, squared=False)}")
    # reg.save(fn)
    return reg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # what version of the data set to run on
    parser.add_argument("--v",type=str, default="data_v26.csv", required=False)
    # --e for neo and --w for word2vec
    parser.add_argument("--e",type=lambda x: True if x=="True" else False, default=False, required=False)
    parser.add_argument("--w",type=lambda x: True if x=="True" else False, default=False, required=False)
    # how many kfold splits to do
    parser.add_argument("--k",type=int, default=15, required=False)
    parser.add_argument("--c", type=str, default="result_v1.csv", required=False)
    parser.add_argument("--s",type=int, default=20, required=False)

    args = parser.parse_args()

    train()

