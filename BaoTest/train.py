from csv import DictReader
import model
import random
class BaoTrainingException(Exception):
    pass


def train_and_save_model(csv_file, verbose=True):
    x = []
    y = []
    pairs = []
    tx = []
    ty =[]
    with open(csv_file, "r") as f:
        reader = DictReader(f)
        for i, row in enumerate(reader):
            pairs.append((row["plan"], row["execution_time (ms)"]))

            
    random.shuffle(pairs)

    x = [i for i, _ in pairs[:290]]
    y = [i for _, i in pairs[:290]]
    tx = [i for i, _ in pairs[290:]]
    ty = [i for _, i in pairs[290:]]       
    # for _ in range(emphasize_experiments):
    #     all_experience.extend(storage.experiment_experience())
    # if not all_experience:
    #     raise BaoTrainingException("Cannot train a Bao model with no experience")
    
    # if len(all_experience) < 20:
    #     print("Warning: trying to train a Bao model with fewer than 20 datapoints.")

    reg = model.BaoRegression(have_cache_data=False, verbose=verbose)
    reg.fit(x, y)

    result = reg.predict(tx)


    for i in range(len(result)):
        print(result[i], ty[i])
    # reg.save(fn)
    return reg

if __name__ == "__main__":
    train_and_save_model("trail_two_data.csv")

