# Usage

To use this Software for Query Execution Time Regression, please use the guidance below to make use of this software:

1. Preprocess all the query features by running the commandline below

```
python preprocess.py
```


2. Run the default training algorithms and collect results as the following

```
python main.py --query_feature tfidf --table_feature stats --plan_feature none
```

# Detailed Arguments

Below we list all the arguments for ''proprocess.py''.
```
usage: preprocess.py [-h] [--input_filename INPUT_FILENAME]
                     [--num_evals NUM_EVALS]

optional arguments:
  -h, --help            show this help message and exit
  --input_filename INPUT_FILENAME
  --num_evals NUM_EVALS
```

- ''--input_filename'' corresponds to the raw input csv file.
- ''--num_evals'' corresponds to the number of data used for evaluation (the default is set to 20%).

---

Below we list all the arguments for ''main.py''.
```
usage: main.py [-h] [--input_filepath INPUT_FILEPATH] [--seed SEED]
                  [--reg REG] [--lr LR] [--model_name {mlp,lasso}]
                  [--query_feature {bow,tfidf,bert}]
                  [--plan_feature {none,bow,tfidf}]
                  [--table_feature {none,onehot,stats}]

optional arguments:
  -h, --help            show this help message and exit
  --input_filepath INPUT_FILEPATH
  --seed SEED
  --reg REG
  --lr LR
  --model_name {mlp,lasso}
  --query_feature {bow,tfidf,bert}
  --plan_feature {none,bow,tfidf}
  --table_feature {none,onehot,stats}
```

- ''--input_filename'' corresponds to the processed input pkl file (by running ''preprocess.py'').
- ''--seed'' corresponds to the random state used for the model initialization and training.
- ''--reg'' corresponds to the coefficient set for the regularization term.
- ''--lr'' corresponds to the learning rate set for the learning process.
- ''--model_name'' corresponds to the type of classifier we are aiming to run, choices are {mlp,lasso}
- ''--query_feature'' corresponds to the type of features for the query input, which includes {bow,tfidf,bert}
- ''--plan_feature'' corresponds to the type of features processing for the plan input, which includes {none,bow,tfidf}
- ''--table_feature'' corresponds to the type of features processing for the table input, which includes {none,onehot,stats}
