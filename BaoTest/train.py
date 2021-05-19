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
from RDS_query import run_query
import time
from threading import Thread
from tqdm import tqdm
from time import sleep
import queue


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
        
         
        w2v.save(args.save_word2vec)
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
    # torch.save(reg.stat_dict(), args.m)
    reg.save(args.m)
    return reg


def get_explain_output(query):
    """
    :query str the SQL query
    :return the query and the explain analyze output
    """
    try:
        return query, run_query(query)[0][0][0]
    except:
        return query, None

def target(query, queue):
    start= int(round(time.time() * 1000))
    try:
        run_query(query)
    except:
        raise RuntimeError("error running the query")

    queue.put(f"{int(round(time.time() * 1000))-start} ms total for the query")
    return
def progress_bar():
    shape = None
    w2v = None
    if args.w:
        w2v = Word2Vec.load(args.save_word2vec)
        for word in w2v.wv:
            shape = word.shape
            break

    reg = model.BaoRegression(have_cache_data=False, verbose=True, neo=args.e, word2vec=w2v, shape=shape)
    reg.load(args.m)
    query = f"EXPLAIN (COSTS true, FORMAT json, BUFFERS true) {args.q}"
    q, output = get_explain_output(query)
    predicted = reg.predict([output["Plan"]])
    predY = [pred[0] for pred in predicted][0]
    q = queue.Queue()
    i = 1
    started = False
    start = int(round(time.time() * 1000))
    t = Thread(target=target, args=(1,q))
    t.start()

    over_predict = False
    total = math.ceil(predY/1000)
    with tqdm(total=total, position=0, leave=True) as pbar:
        for i in tqdm(range(total), position=0, leave=True):         
            elapsed = int(round(time.time() * 1000))-start
            if not t.is_alive():
               over_predict = True
               pbar.close()
               break
            if elapsed >= 5000:
                pbar.close()
                break
            sleep(0.5)
  
    if over_predict:
        print(f"over prediction by {10000-elapsed} ms")
        t.join()
        print(q.get())
    else:
        print(f"under prediction by {elapsed - 10000} ms")  
        t.join()
        print(q.get()) 

    return


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
    #for convolutional tree
    parser.add_argument("--m", type=str, required=True)
    #if we are training or doing a progress bar
    parser.add_argument("--p", type=lambda x: True if x=='True' else False, default=False, required=False)
    parser.add_argument("--save_word2vec", type=str, default="", required=False)
    # q for query
    parser.add_argument("--q", type=str, default='', required=False)
    args = parser.parse_args()
    if not args.p:
        train()
    else:
        progress_bar()
    

    

