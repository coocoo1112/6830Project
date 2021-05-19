from csv import writer, DictWriter, DictReader
import os
import json
# from featurize import get_all_relations
import numpy as np
import pickle
import matplotlib.pyplot as plot
import math
import argparse
# relevant csvs to union would be v3, v5, v6, v7, v8, v10


FIELDS = ["query", "plan", "execution_time (ms)", "tables"]


def get_table_stats(table_name):
    """
    Use this to get that table stats dict for tables
    :param table_name : str this is the table name
    """
    path = "../table_info/{}_table_stats.json".format(table_name)
    with open(path, "r") as f:
        stats = json.load(f)
        return stats


def dataset_iter(csv_name):
    """
    :csv_name is a string, path to csv dataset we want to load
    :return a generator yielding one row at a time in our dataset
    """
    if not os.path.exists(csv_name):
        print(f"{csv_name} does not exist")
        return
    else:
        with open(csv_name, "r") as f:
            reader = DictReader(f, FIELDS)
            for i, row in enumerate(reader):
                if i != 0:
                    yield {k: v if k not in ["plan", "tables"] else json.loads(v) for k,v in row.items()}


def get_stats_dict(vals):
    """
    :vals list of ints or floats
    : return dict of min, max, mean, median, std
    """
    return {
        "min": np.min(vals),
        "max": np.max(vals),
        "mean": np.mean(vals),
        "median": np.median(vals),
        "std" : np.std(vals)
    }


def summarize_dataset(csv_name):
    """
    :csv_name the csv name
    :return stats dict of min, max, mean, median, std wrt to execution time
    """

    vals = np.array([float(row["execution_time (ms)"]) for row in dataset_iter(csv_name)])
    counts = {}
    return get_stats_dict(vals)
    

def get_counts(csv_name):
    """
    :csv_name csv name
    :return dict of counts stats min, max, mean, median, std
    """
    total_ = get_unique_query_times(csv_name)
    counts = {}
    for _, time, _ in total_:
        counts.setdefault(time, 0)
        counts[time] += 1
    vals = np.array(list(counts.values()))
    return get_stats_dict(vals), total_

def dataset_stats(file, *args):
    """
    :*args any number of csv names
    :dump to json file table stats: min, max, mean, median, stdv 
    """
    if os.path.exists(file):
        print(f"{file} already exists")
        return

    stats = {}
    for csv_name in args:
        print(f"on file: {csv_name}")
        stats[csv_name] = summarize_dataset(csv_name)
    # with open(file, "wb") as f:
    #     pickle.dump(stats, f)
    with open(file, "w") as f:
        json.dump(json.dumps(stats, indent=4), f)


def get_uniform_params(csv_name, weighted_func):
    """
    :csv_name is the csv name
    :weighted func is a function, it is how much we care about num samples vs how wide our distribution is
    :return max number of samples we can get from a uniform distribution
    """
    counts_stats, total = get_counts(csv_name)
    print("stats", counts_stats)
    #total = get_unique_query_times(csv_name)
    counts = {}
    for _, time, _ in total:
        counts.setdefault(time, 0)
        counts[time] += 1
    print(sum(counts.values()))
    def find_candidate_val(epsilon):
        """
        :epsilon if a float between [0,1] that we can tune to see how close to uniform we can get 
        :return get best uniform sample
        """
        print(f"num iterations: {len(counts)}, epsilon: {epsilon}")
        max_ = -float('inf')
        value_ = None
        answer = []
        for k, value in counts.items():
            num_samples, width, epsilon = compute_sample_size(counts, value, epsilon)
            answer.append((num_samples, value, width, epsilon))
        #answer.sort(key=lambda x: 0.10*x[0] + 0.9*x[-1])
        candidates = []
        for i, j, k, _ in answer:
            # if i >= 8000:
            candidates.append((i,j,k, _))
        candidates.sort(key = lambda x: weighted_func(x))
        # print("candidates: ", candidates)
        if candidates:
            return candidates[-1]
        else:
            return None
    low = 0
    high = .5
    answer = None
    best_so_far = 0
    while low < high:
        print("----------")
        print(low, high)
        mid = (low+high)/2
        candidate = find_candidate_val(mid)
        if not candidate:
            high = mid - 0.001
        else:
            if weighted_func(candidate) >= best_so_far:
                best_so_far = weighted_func(candidate)
                answer = candidate
                low = mid + 0.001
            else:
                high = mid - 0.001
        print(low, high)
        print("-----------")
    print(answer)
    return answer


def make_uniform_dataset(new_csv, old_csv, weighted_func):
    params = get_uniform_params(old_csv, weighted_func)
    num_sample, value, width, epsilon = params
    width = set(width)
    print(width)
    counts = {}
    print(f"what i think {num_sample}")
    with open(new_csv, "w", newline='') as f:
        good = 0
        csv_writer = writer(f)
        csv_writer.writerow(FIELDS)
        dict_writer = DictWriter(f, fieldnames=FIELDS)
        for row in dataset_iter(old_csv):
            t = round_time(row)
            if t in width:
                counts.setdefault(t, 0)
                if counts[t] <= value:
                    counts[t] += 1
                    good += 1
                    _row = {k: v if k not in ["plan", "tables"] else json.dumps(v) for k,v in row.items()}
                    dict_writer.writerow(_row)
        print(f"dataset of size {good} made")
        


def compute_sample_size(counts, val, epsilon):
    """
    given a counts_dict: compute how many sample will come from truncating the sample by val
    :counts counts dict
    :val val to test on
    :epsilon: how strictly uniform we need to be
    """
    total = 0
    width = []
    for k, v in counts.items():
        if (1-epsilon)*val <= v:
            total += (1-epsilon)*val
            width.append(k)
    # for w in width:
    #     if w > 900:
    #         print(total, width, epsilon)
            
    return total, width, epsilon


def filter_outliers(new_csv, *args):
    """
    :new_csv new csv to make
    :*args any number of csv names
    :make a new csv with no outliers from previous datasets
    """
    if os.path.exists(new_csv):
        print(f"choose a different name, {new_csv} already exists")
        return

    total = get_unique_query_times(*args)
    counts = {}
    for _, time, true_time in total:
        counts.setdefault(time, 0)
        counts[time] += 1
    n = len(total)
    for x, y in counts.items():
        print(f"time: {x}, count: {y}")
    x = []
    y = []
    cutoff = np.quantile(np.array(list(counts.values())), .50)
    print(f"cutoff: {cutoff}")
    times = {k for k, v, in counts.items() if v > cutoff}
    queries = {q for q, t, _ in total if t in times}
    length = len(queries)

    with open(new_csv, "w", newline='') as f:
        csv_writer = writer(f)
        csv_writer.writerow(FIELDS)
        dict_writer = DictWriter(f, fieldnames=FIELDS)
        for csv_name in args:
            for row in dataset_iter(csv_name):
                if row["query"] in queries:
                    _row = {k: v if k not in ["plan", "tables"] else json.dumps(v) for k,v in row.items()}
                    dict_writer.writerow(_row)
                    queries.remove(row["query"])
        print(f"dataset of size {length} made")


def visualize_data(*args, quantile=True):
    """
    :*args any number of csv_names
    """
    # everything is in terms of milliseconds, so we can round to the nearest 100 milliseconds for better view of data
    
    total = get_unique_query_times(*args)
    counts = {}
    for _, time, _ in total:
        counts.setdefault(time, 0)
        counts[time] += 1
    n = len(total)
    x = []
    y = []
    cutoff = np.quantile(np.array(list(counts.values())), .85)
    print(f"cutoff: {cutoff}")
    if quantile:
        for _y, _x in counts.items():
                if _x > cutoff:
                    y.append(_y)
                    # empirical pdf
                    x.append(_x/n)
    else:
        for q, rounded_time, real_time in total:
            y.append(real_time)
            # empirical pdf
            x.append(counts[rounded_time])
            # print(f"value: {_y} frequency: {_x}")
    print(f"num samples: {len(y)}")
    
    if quantile:
        rem = {t: counts[t] for t in y}
        print(f"highest count: {max(rem.values())}")
        print(f"lowest count: {min(rem.values())}")
        print(f"total samples: {sum(rem.values())}")

        plot.scatter(y, x)
        plot.show()
    else:
        x_ = []
        y_ = []
        
        for _y, _x in counts.items():
            y_.append(_y)
            # empirical pdf
            x_.append(_x/n)
        other_counts = {}
        for real in y:
            other_counts.setdefault(real, 0)
            other_counts[real] += 1
        n = len(y)
        ry = []
        rx = []

        for k, v in other_counts.items():
            rx.append(k)
            ry.append(v/n)
        # print(x_)
        # print(y_)
        ry = np.array(ry)
        stats = get_stats_dict(ry)
        print(stats)
        plot.scatter(y_, x_)
        plot.title("Empirical PDF of query execution times")
        plot.xlabel("Query Execution Time (ms)")
        plot.ylabel("Empircal Probability")
        plot.show()

        y = np.array([float(i) for i in y])
        print(y)
        stats_ = get_stats_dict(y)
        print(stats_)
        


def get_unique_query_times(*args):
    """
    this function unions several csvs, so we need to make sure there are no duplicate queries, this is why it's called unique
    :*args any number of csv_names
    :return list of tuples : (query, rounded time, true time)
    """
    unique = set()
    total = []
    for csv_name in args:
        for row in dataset_iter(csv_name):
            if row["query"] not in unique:
                q = row["query"]
                t = round_time(row)
                total.append((q, t, row["execution_time (ms)"]))
                unique.add(row["query"])
    return total

def round_time(row):
    return int(math.ceil(float(row["execution_time (ms)"]) / 100.0)) * 100

def save_results(result_set_name, results, data_set_name, instance_size=2, neo=False, word2vec=False, regression=False):
    """
    :result_set_name is the name of the csv to save to, str
    :results is a list of rmses from trials
    :data_set_name is the name the model was trained on
    :neo, word2vec, bao are flags to see which model (encoding scheme was used)
    """
    mode = "a" if os.path.exists(result_set_name) else "w"
    first =  os.path.exists(result_set_name)
    fields_ = ["model name", "instance size", "dataset name", "num trials", "mean rmse", "min rmse", "max rmse", "std rmse", "median rmse"]
    name = "Bao"
    if neo and not word2vec:
        name = "Neo"
    elif word2vec:
        name = "word2vec"
    if not neo and not word2vec and regression:
        name = "postgres cost prediction"
    name = f"{name} regression" if regression else name
    # if neo and bao:
    #     name = "Hybrid"
    
    num_trials = len(results)
    rmse_stats = get_stats_dict(results)
    with open(result_set_name, mode, newline='') as r:
        if not first:
            csv_writer = writer(r)
            csv_writer.writerow(fields_)
        row_dict = {"model name": name, "num trials": num_trials, "instance size": instance_size, "dataset name": data_set_name}
        for i, field_name in enumerate(fields_):
            stat_name = field_name.split()[0]
            if stat_name in rmse_stats:
                row_dict[field_name] = rmse_stats[stat_name]
        dict_writer = DictWriter(r, fieldnames=fields_)
        dict_writer.writerow(row_dict)
        print(f"results saved to {result_set_name} with model type {name}")
    

def get_size_vocab(sentences):
    """
    get number of words our vocab
    """
    seen = set()
    for x in sentences:
        for word in x:
            try:
                if word not in seen:
                    seen.add(word)
            # hacky don't know why i needed this
            except TypeError:
                for actual_word in word:
                    if actual_word not in seen:
                        seen.add(actual_word)
    return len(seen)
    

def get_word_vector_sentence(sentence, nlp, shape):
    vector = np.copy(nlp.wv[sentence[0]]) if sentence[0] in nlp.wv else np.zeros(shape)
    for word in sentence[1:]:
        if word not in nlp.wv:
            vector += np.zeros(shape)
        else:
            vector += np.copy(nlp.wv[word])

    return vector

def get_shape_vector(sentences, nlp):
    for sentence in sentences:
        for word in sentence:
            if word in sentence:
                return nlp.wv[word].shape

def get_offset(query):
    return query.index(")")
    


if __name__ == "__main__":
    # these are the RAW datasets i.e no rounding
    # current_data = ["data_v3.csv", "data_v5.csv", "data_v6.csv", "data_v7.csv", "data_v8.csv", "data_v10.csv"]
    # current_data = ["data_v30.csv", "data_v31.csv"]
    #dataset_stats("stats_data_v0.json", *current_data)
    # visualize_data(*current_data)
    # filter_outliers("data_v38.csv", *current_data)
    # print(summarize_dataset("data_v11.csv"))
    #visualize_data("data_v11.csv")
    #print(get_counts("data_v11.csv")[0])
    # print(make_uniform_dataset("data_v42.csv", "data_v38.csv", lambda x: .975*x[0] + 0.025*max(x[-2])))
    parser = argparse.ArgumentParser()
    parser.add_argument("--v", type=str, required=True)
    parser.add_argument("--q", type=lambda x: True if x == "True" else False, required=False)
    parser.add_argument("--n", type=str, required=False)
    args = parser.parse_args()
    if not args.n:
        print(visualize_data(args.v, quantile=args.q))
    else:
        print(make_uniform_dataset(args.n, args.v, lambda x: .975*x[0] + 0.025*max(x[-2])))

    #26 is the other dataset
