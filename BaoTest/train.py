from csv import DictReader
import model
import random
from data_utils import dataset_iter, save_results, get_size_vocab, get_shape_vector, get_word_vector_sentence, get_offset
from sklearn.metrics import mean_squared_error 
import numpy as np
import sys
from gensim.models import Word2Vec
from sklearn.model_selection import KFold
import argparse
import math
import torch

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
    if args.e:
        x = [(p, q) for q, p, r in pairs]
        y = [float(r) for q, p, r in pairs]
        
    else:
        x = [i for i, _ in pairs]
        y = [float(i) for _, i in pairs]

    shape=None
    w2v=None 
    if args.w:
        offset = get_offset(x[0][1])
        sentences = [[i.strip() for i in data[1][get_offset(x[0][1])+1:][:-1].split()] for data in x]       
        size = math.floor(.1*get_size_vocab(sentences))
        w2v = Word2Vec(sentences, window=20, workers=16, vector_size = size)
        shape = get_shape_vector(sentences, w2v)  
    reg = model.BaoRegression(have_cache_data=False, verbose=verbose, neo=args.e, word2vec=w2v, shape=shape)
    #print(y)
    # print("1")
    # print(np.mean(ty))
    # print(np.std(ty))
    # print("2")
    kf = KFold(n_splits=args.k, shuffle=True)
    results = []
    i = 1
    for train_idx, test_idx in kf.split(x):
        trainX =[x[i] for i in train_idx]
        testX = [x[i] for i in test_idx]
        trainY = [y[i] for i in train_idx]
        trueY = [y[i] for i in test_idx]
        reg.fit(trainX, trainY)


        predictions = reg.predict(testX)
        predY = [pred[0] for pred in predictions]
        rmse = mean_squared_error(trueY, predY, squared=False)
        print(f"Trial {i} RMSE: {rmse}")
        i += 1
        results.append(rmse)
    save_results(args.c, results, args.v, args.s, args.e, args.w, False)
    torch.save(reg.stat_dict(), args.m)
    return reg

def progress_bar():
    reg = model.BaoRegression(have_cache_data=False, verbose=verbose, neo=args.e, word2vec=w2v, shape=shape)

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
    #path to save model and path to load
    parser.add_argument("--m", type=str, required=True)
    #if we are training or doing a progress bar
    parser.add_argument("--p", type=lambda x: True if x=='True' else False, default=False, required=False)
    
    args = parser.parse_args()
    if not args.p:
        train()
    else:
        pass


