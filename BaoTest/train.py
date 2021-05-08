from csv import DictReader
import model
class BaoTrainingException(Exception):
    pass


def train_and_save_model(fn, csv_file, verbose=True, emphasize_experiments=0):
    x = []
    y = []
    with open(csv_file, "r") as f:
        reader = DictReader(f)
        for i, row in enumerate(reader):
            x.append(row["plan"])
            y.append(row["execution_time (ms)"])
            if i > 23:
                break



    # for _ in range(emphasize_experiments):
    #     all_experience.extend(storage.experiment_experience())
    # if not all_experience:
    #     raise BaoTrainingException("Cannot train a Bao model with no experience")
    
    # if len(all_experience) < 20:
    #     print("Warning: trying to train a Bao model with fewer than 20 datapoints.")

    reg = model.BaoRegression(have_cache_data=True, verbose=verbose)
    reg.fit(x, y)
    # reg.save(fn)
    return reg