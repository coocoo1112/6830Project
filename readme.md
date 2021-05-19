Query Performance predicter

for files in BaoTest folder:
-in order to run our code, you first must set the correct database credential in BaoTest/RDS_query.py this is important, if you don't do this, you cannot generate data or use the progress bar. Our BaoTest/generate_sql.py is tailored specifically to TPC-H schema and does not generalize.

-to generate a dataset you need to run cd in the BaoTest folder and run python generate_data.py --v [name of csv to save to]

-to get a plot of a dataset, cd into BaoTest folder and run python data_utils.py --v [dataset name to visualize] --q [if you want to filter out values less than a quantile for yes, type in True] 

-to make a dataset more uniform, cd into BaoTest folder and run python data_utils.py --v [dataset name to make uniform] --n [new dataset name to save to]

-to train a tree convolution model, cd into BaoTest and run python train.py --v [training data set name] --e [True if we want to run neo encoding False else] --w [True if we want to run word2vec encoding False else, if this param is True, --e MUST ALSO BE TRUE] --k [how many folds of cross validation we want to do] --c [csv name of where we want to save results] --s [size of database instance dataset came from] --save_word2vec [where to save word2vec model] --m [path to save tree convolution model] 

-to run the progress par, cd into BaoTest an run python train.py --m [path of model to load] --save_word2vec [path of word2vec model to load this is optional] --p [needs to be True, so type in True] --q [query to run]

-to run a regression, cd into BaoTest and run python baseline_regression.py --v [name of dataset to train on] --e [if we are doing query encdoing or no, for yes type in True] --w [if we are doing word2vec encoding, for yes type in True if this is True, then --e MUST ALSO BE TRUE] --k [how many folds of cross validation we want to do] --c [csv name of where we want to save results] --s [size of database instance dataset came from]
