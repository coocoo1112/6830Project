import os
import json
import argparse
import constant
import pickle
import numpy as np
import transformers
import glob

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

def _commandline_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_filepath', type=str, default='data/data.v1.csv.pkl')
  parser.add_argument('--seed', type=int, default=199)
  parser.add_argument('--reg', type=float, default=0.01)
  parser.add_argument('--lr', type=float, default=0.01)
  parser.add_argument('--model_name', type=str, default='mlp', choices=['mlp', 'lasso'])
  parser.add_argument('--query_feature', type=str, default='bow', choices=['bow', 'tfidf', 'bert'])
  parser.add_argument('--table_feature', type=str, default='stats', choices=['none', 'onehot', 'stats'])

  return parser

def prepare_table(table_feature):
  table_filenames = glob.glob(os.path.join(constant.table_data_path, '*.json'))
  table_names = [os.path.basename(item) for item in table_filenames]

  D = len(table_filenames) + 1
  table_feat_dict = {}
  table_feat_dict[''] = np.zeros([1, D])
  for id, table_name in enumerate(table_names):
    if table_feature == 'none':
      table_feat_dict[table_name] = np.zeros([1, D])
    elif table_feature == 'onehot':
      table_feat_dict[table_name] = np.zeros([1, D])
      table_feat_dict[table_name][:, id] = 1
    elif table_feature == 'stats':
      table_feat_dict[table_name] = np.zeros([1, D])
      table_feat_dict[table_name][:, id] = 1
      numrows = json.load( open(table_filenames[id], 'r') )['rows']
      table_feat_dict[table_name][:, -1] = numrows

  return table_feat_dict

def prepare_label(Xtrain, Xtest):
  y_train = [item[1] for item in Xtrain]
  y_test = [item[1] for item in Xtest]

  return y_train, y_test

def featurize_data(Xtrain, Xtest, query_feature, table_feature):
  text_train = [item[0] for item in Xtrain]
  text_test = [item[0] for item in Xtest]

  Xhat_train = None
  Xhat_test = None
  if query_feature in {'bow'}:
    featurizer = CountVectorizer()
  elif query_feature in {'tfidf'}:
    featurizer = TfidfVectorizer()

  Xhat_train = featurizer.fit_transform(text_train)
  Xhat_test = featurizer.transform(text_test)

  table_dict = prepare_table(table_feature)
  table_train = np.vstack([np.hstack([table_dict[item[2]], table_dict[item[3]]]) for item in Xtrain])
  table_test  = np.vstack([np.hstack([table_dict[item[2]], table_dict[item[3]]]) for item in Xtest])

  Xhat_train = np.hstack([table_train, Xhat_train.toarray()])
  Xhat_test = np.hstack([table_test, Xhat_test.toarray()])

  return Xhat_train, Xhat_test

def prepare_estimator(args):
  if args.model_name == 'lasso':
    estimator = Lasso(alpha=0.1)
  elif args.model_name == 'mlp':
    estimator = MLPRegressor(
        hidden_layer_sizes=(256,128),
        max_iter=2000,
        early_stopping=True,
        random_state=args.seed,
        alpha=args.reg,
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
  Xhat_train, Xhat_test = featurize_data(X_train, X_test, args.query_feature, args.table_feature)
  # Imputation Model
  est = prepare_estimator(args)
  est.fit(Xhat_train, y_train)

  Pred_test = est.predict(Xhat_test)

  print('Global Mean Squared Error: {}'.format(_evalMSE(y_test, Pred_test)))

if __name__ == '__main__':
  parser = _commandline_parser()
  args = parser.parse_args()

  main(args)

