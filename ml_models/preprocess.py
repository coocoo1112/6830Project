import ast
import os
import csv
import argparse
import constant
import pickle
import numpy as np

def _commandline_parser():
  """Commandline parser."""
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_filename', type=str, default='data.v3.csv')
  parser.add_argument('--num_evals', type=int, default=2000)

  return parser

def write_csv(filepath, column_names, data):
  """Write CSV file."""
  with open(filepath, 'w') as fd:
    CSVwriter = csv.writer(fd, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    CSVwriter.writerow(column_names)
    for i in range(len(data)):
      row = data[i]
      CSVwriter.writerow(row)

def _curate_data(data):
  """Removing rows that contains NA."""
  new_data = []
  for row in data:
    execute_time = ast.literal_eval(row[2])

    new_data.append((row[0].lower(), execute_time, row[1].lower(), row[3].lower()))

  return new_data

if __name__ == '__main__':
  parser = _commandline_parser()
  args = parser.parse_args()

  input_filepath = os.path.join(constant.raw_data_path, args.input_filename)
  with open(input_filepath, 'r') as fd:
    raw_data = list(csv.reader(fd, delimiter=',',  quotechar='"', quoting=csv.QUOTE_ALL, skipinitialspace=True))

  column_names = raw_data[0][:-1]
  data = _curate_data(raw_data[1:])

  N = len(data)

  np.random.seed(199)
  data_ids = np.random.permutation(range(N))
  train_ids = data_ids[:-args.num_evals]
  test_ids = data_ids[-args.num_evals:]

  train_data = [data[idx] for idx in train_ids]
  test_data = [data[idx] for idx in test_ids]

  output_filepath = os.path.join(constant.processed_data_path, args.input_filename + '.pkl')
  with open(output_filepath, 'wb') as fd:
    pickle.dump(
        dict(
          column_names=column_names,
          train_data=train_data,
          test_data=test_data), fd)

  output_dirpath = os.path.join(constant.processed_data_path, args.input_filename.split('.')[0])
  if not os.path.exists(output_dirpath):
    os.makedirs(output_dirpath)

  output_data_loops = [
      (train_data, 'train_data.cvs'),
      (test_data, 'test_data.cvs'),
  ]
  for data, data_filepath in output_data_loops:
    write_csv(os.path.join(output_dirpath, data_filepath), column_names, data)
