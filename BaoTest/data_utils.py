from csv import writer, DictWriter, DictReader
import os
import json
from featurize import get_all_relations


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