


class BaoTrainingException(Exception):
    pass


def train_and_save_model(fn, verbose=True, emphasize_experiments=0):
    all_experience = None#storage.experience()

    for _ in range(emphasize_experiments):
        all_experience.extend(storage.experiment_experience())
    
    x = [i[0] for i in all_experience]
    y = [i[1] for i in all_experience]        
    
    if not all_experience:
        raise BaoTrainingException("Cannot train a Bao model with no experience")
    
    if len(all_experience) < 20:
        print("Warning: trying to train a Bao model with fewer than 20 datapoints.")

    reg = model.BaoRegression(have_cache_data=True, verbose=verbose)
    reg.fit(x, y)
    reg.save(fn)
    return reg