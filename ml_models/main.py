import os
import json
import argparse
import constant
import pickle
import numpy as np
import transformers
import glob

import torch
from tqdm import tqdm

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import StandardScaler
from transformers import pipeline, AutoTokenizer

def _commandline_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_filepath', type=str, default='data/data.v3.csv.pkl')
  parser.add_argument('--seed', type=int, default=199)
  parser.add_argument('--reg', type=float, default=0.01)
  parser.add_argument('--lr', type=float, default=0.001)
  parser.add_argument('--model_name', type=str, default='mlp', choices=['mlp', 'lasso'])
  parser.add_argument('--query_feature', type=str, default='bow', choices=['bow', 'tfidf', 'bert'])
  parser.add_argument('--plan_feature', type=str, default='none', choices=['none', 'bow', 'tfidf'])
  parser.add_argument('--table_feature', type=str, default='none', choices=['none', 'onehot', 'stats'])

  return parser

def prepare_table(table_feature):
  table_filenames = glob.glob(os.path.join(constant.table_data_path, '*.json'))
  table_names = [os.path.basename(item) for item in table_filenames]
  table_keys = [os.path.basename(item).replace('_table_stats.json', '') for item in table_filenames]

  D = len(table_filenames) + 1
  table_feat_dict = {}
  table_feat_dict[''] = np.zeros([1, D])
  for id, table_name in enumerate(table_names):
    if table_feature == 'none':
      table_feat_dict[table_keys[id]] = np.zeros([1, D])
    elif table_feature == 'onehot':
      table_feat_dict[table_keys[id]] = np.zeros([1, D])
      table_feat_dict[table_keys[id]][:, id] = 1
    elif table_feature == 'stats':
      table_feat_dict[table_keys[id]] = np.zeros([1, D])
      table_feat_dict[table_keys[id]][:, id] = 1
      numrows = json.load( open(table_filenames[id], 'r') )['rows']
      table_feat_dict[table_keys[id]][:, -1] = numrows

  return table_feat_dict

def prepare_bert_features(train_sents, eval_sents):
  print('Feature encoding....')
  feature_extraction = pipeline('feature-extraction', model="bert-base-uncased", tokenizer="bert-base-uncased", device=1)
  train_features = []
  for sent in tqdm(train_sents):
    train_features.append(np.array(feature_extraction(sent)).max(1))
  train_features = np.concatenate(train_features)

  eval_features = []
  for sent in tqdm(eval_sents):
    eval_features.append(np.array(feature_extraction(sent)).max(1))
  eval_features = np.concatenate(eval_features)
  return train_features, eval_features

def prepare_label(Xtrain, Xtest):
  y_train = [item[1] for item in Xtrain]
  y_test = [item[1] for item in Xtest]

  return y_train, y_test

def featurize_data(Xtrain, Xtest, query_feature, plan_feature, table_feature):
  text_train = [item[0] for item in Xtrain]
  text_test = [item[0] for item in Xtest]

  Xhat_train = None
  Xhat_test = None
  if query_feature in {'bow'}:
    featurizer = CountVectorizer()
  elif query_feature in {'tfidf'}:
    featurizer = TfidfVectorizer()

  if query_feature in {'bert'}:
    Xhat_train, Xhat_test = prepare_bert_features(text_train, text_test)
  else:
    Xhat_train = featurizer.fit_transform(text_train).toarray()
    Xhat_test = featurizer.transform(text_test).toarray()

  plan_train = [item[2] for item in Xtrain]
  plan_test  = [item[2] for item in Xtest]

  if plan_feature in {'bow'}:
    plan_featurizer = CountVectorizer()
  elif plan_feature in {'tfidf'}:
    plan_featurizer = TfidfVectorizer()
  else:
    plan_featurizer = None

  if plan_featurizer:
    plan_train = plan_featurizer.fit_transform(plan_train).toarray()
    plan_test = plan_featurizer.transform(plan_test).toarray()
  else:
    plan_train = np.zeros([len(plan_train), 16])
    plan_test = np.zeros([len(plan_test), 16])

  table_train = [json.loads(item[3]) for item in Xtrain]
  table_test  = [json.loads(item[3]) for item in Xtest]
  table_dict = prepare_table(table_feature)
  output_table_train = []
  output_table_test = []
  for row in table_train:
    if len(row) == 1:
      output_table_train.append(np.concatenate([table_dict[row[0]], table_dict['']], axis=-1))
    else:
      output_table_train.append(np.concatenate([table_dict[row[0]], table_dict[row[1]]], axis=-1))

  for row in table_test:
    if len(row) == 1:
      output_table_test.append(np.concatenate([table_dict[row[0]], table_dict['']], axis=-1))
    else:
      output_table_test.append(np.concatenate([table_dict[row[0]], table_dict[row[1]]], axis=-1))

  table_train = np.concatenate(output_table_train)
  table_test = np.concatenate(output_table_test)

  Xhat_train = np.hstack([Xhat_train, plan_train, table_train])
  Xhat_test = np.hstack([Xhat_test, plan_test, table_test])

  preprocessor = StandardScaler()
  preprocessor.fit(Xhat_train)
  return preprocessor.transform(Xhat_train), preprocessor.transform(Xhat_test)

def prepare_estimator(args):
  if args.model_name == 'lasso':
    estimator = Lasso(alpha=0.1)
  elif args.model_name == 'mlp':
    estimator = MLPRegressor(
        hidden_layer_sizes=(256,128),
        max_iter=50,
        early_stopping=True,
        random_state=args.seed,
        alpha=args.reg,
        verbose=True,
        learning_rate_init=args.lr)

  return estimator

def _evalMSE(labels, preds):
  error = labels - preds
  return (error ** 2).sum(axis=-1).mean()

def _evalAcc(labels, preds):
  match = labels == preds
  return match.mean(axis=0).mean()

def main(args):
  # Preparing data
  with open(args.input_filepath, 'rb') as fd:
    data = pickle.load(fd)

  column_names = data['column_names']
  X_train = data['train_data']
  X_test = data['test_data']

  y_train, y_test = prepare_label(X_train, X_test)
  Xhat_train, Xhat_test = featurize_data(X_train, X_test, args.query_feature, args.plan_feature, args.table_feature)
  # Imputation Model
  est = prepare_estimator(args)
  est.fit(Xhat_train, y_train)

  Pred_test = est.predict(Xhat_test)

  print('Global Mean Squared Error: {}'.format(_evalMSE(y_test, Pred_test)))

if __name__ == '__main__':
  parser = _commandline_parser()
  args = parser.parse_args()

  main(args)

