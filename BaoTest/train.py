from csv import DictReader
import model
import random
from data_utils import dataset_iter
from sklearn.metrics import mean_squared_error 
import numpy as np

class BaoTrainingException(Exception):
    pass


def train_and_save_model(csv_file, verbose=True, neo=False):
    x = []
    y = []
    pairs = []
    tx = []
    ty =[]
    for row in dataset_iter(csv_file):
        if neo:
            pairs.append((row["query"], row["plan"], row["execution_time (ms)"]))
        else:
            pairs.append((row["plan"], row["execution_time (ms)"]))

            
    random.shuffle(pairs)
    train_percent = .8
    train_amount = int(len(pairs) * train_percent)
    print(train_amount)
    if neo:
        x = [(p, q) for q, p, r in pairs[:train_amount]]
        y = [float(r) for q, p, r in pairs[:train_amount]]
        tx = [(p, q) for q, p, r in pairs[train_amount:]]
        ty = [float(r) for q, p, r in pairs[train_amount:]]
    else:
        x = [i for i, _ in pairs[:train_amount]]
        y = [float(i) for _, i in pairs[:train_amount]]
        tx = [i for i, _ in pairs[train_amount:]]
        ty = [float(i) for _, i in pairs[train_amount:]]  
   

 
    # for _ in range(emphasize_experiments):
    #     all_experience.extend(storage.experiment_experience())
    # if not all_experience:
    #     raise BaoTrainingException("Cannot train a Bao model with no experience")
    
    # if len(all_experience) < 20:
    #     print("Warning: trying to train a Bao model with fewer than 20 datapoints.")
    

    reg = model.BaoRegression(have_cache_data=False, verbose=verbose, neo=neo)
    #print(y)
    print("1")
    print(np.mean(ty))
    print(np.std(ty))
    reg.fit(x, y)
    print("2")

    result = reg.predict(tx)
    ty = np.array(ty).astype(np.float32)
    # print(ty)
    # print(result)

    res = np.array([])
    for i in range(len(result)):
        # print("test")
        res = np.append(res, result[i])
    #print(res)
    # for i in range(len(result)):
    #     print(ty[i], res[i])
    #     print(type(ty[i]), type(res[i]))
    #flat_result = result.flatten()
    #sub = np.subtract(flat_result, ty)
    #print(np.sort(sub))


    print(f"RMSE: {mean_squared_error(ty, res, squared=False)}")
    # reg.save(fn)
    return reg

if __name__ == "__main__":
    train_and_save_model("data_v26.csv", neo=False)

