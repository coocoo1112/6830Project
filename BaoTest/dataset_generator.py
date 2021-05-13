
from csv import writer, DictWriter, DictReader
import os
from RDS_query import run_query
import json
import re
import pickle
from featurize import get_all_relations


FIELDS = ["query", "plan", "execution_time (ms)", "tables"]

columns_query = "SELECT column_name FROM information_schema.columns\
    WHERE table_name = '{}';"

def create_data_set(query, csv_name):
    """
    add rows to a csv file
    :csv_name str - path to csv file
    :query str - SQL query to run
    """
    if not os.path.exists(csv_name):
        with open(csv_name, "w", newline='') as csv_file:
            csv_writer = writer(csv_file)
            csv_writer.writerow(FIELDS)
            dict_writer = DictWriter(csv_file, fieldnames=FIELDS)
            add_row(query, dict_writer)      
    else:
        with open(csv_name, "a", newline="") as csv_file:
            dict_writer = DictWriter(csv_file, fieldnames=FIELDS)
            add_row(query, dict_writer)


def add_row(query, csv_writer):
    output = run_query(query)[0][0][0]
    relations = list(get_all_relations([output]))
    values = [query, json.dumps(output), output["Execution Time"], json.dumps(relations)]
    row_dict = {k:v for k, v in zip(FIELDS, values)}
    csv_writer.writerow(row_dict)


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
    :return a generator yielding one row at a time
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



if __name__ == "__main__":
    #fix error of not dumping to json the explain outputs
    with open("data_v2.csv", "r") as old:
        reader = DictReader(old, FIELDS[:-1])
        with open("data_v3.csv", "w", newline='') as new:
            writer = DictWriter(new, fieldnames=FIELDS)
            for i, row in enumerate(reader):
                print(f"{i}")
                if i != 0:
                    new_row = {}
                    new_row["query"] = row["query"].replace("ANALYZE true", "ANALYZE false")
                    plan = run_query(new_row["query"])[0][0][0]
                    new_row["plan"] = json.dumps(plan)
                    new_row["tables"] = json.dumps(list(get_all_relations([plan])))
                    new_row["execution_time (ms)"] = row["execution_time (ms)"]
                else:
                    new_row = {f:f for f in FIELDS}
                writer.writerow(new_row)

    


    # for data in dataset_iter("data_v3.csv"):
    #     print(data)
    #     break

                






