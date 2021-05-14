from csv import DictReader
import model
import random
from generate_data import dataset_iter
from sklearn.metrics import mean_squared_error 
import numpy as np

class BaoTrainingException(Exception):
    pass


def train_and_save_model(csv_file, verbose=True):
    x = []
    y = []
    pairs = []
    tx = []
    ty =[]
    for row in dataset_iter(csv_file):
        pairs.append((row["plan"], row["execution_time (ms)"]))

            
    random.shuffle(pairs)

    x = [i for i, _ in pairs[:20000]]
    y = [float(i) for _, i in pairs[:20000]]
    tx = [i for i, _ in pairs[20000:]]
    ty = [float(i) for _, i in pairs[20000:]]  

 
    # for _ in range(emphasize_experiments):
    #     all_experience.extend(storage.experiment_experience())
    # if not all_experience:
    #     raise BaoTrainingException("Cannot train a Bao model with no experience")
    
    # if len(all_experience) < 20:
    #     print("Warning: trying to train a Bao model with fewer than 20 datapoints.")

    reg = model.BaoRegression(have_cache_data=False, verbose=verbose)
    #print(y)
    reg.fit(x, y)

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
    train_and_save_model("data_v6.csv")

