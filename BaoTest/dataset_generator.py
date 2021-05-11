
from csv import writer, DictWriter, DictReader
import os
from RDS_query import run_query
import json
import re
import pickle
from featurize import get_all_relations


FIELDS = ["query", "plan", "execution_time (ms)", "tables"]


def create_data_set(query, csv_name):
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
    plan_dict = output["Plan"]
    relations = get_all_relations(plan_dict)
    values = [query, json.dumps(plan_dict, indent=4), output["Execution Time"], json.dumps(relations)]
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


if __name__ == "__main__":
    #fix error of not dumping to json the explain outputs
    with open("data_v2.csv", "r") as old:
        reader = DictReader(old, FIELDS[:-1])
        with open("data_v3.csv", "w") as new:
            writer = DictWriter(new, fieldnames=FIELDS)
            for row in reader:
                row["query"] = row["query"].replace("ANALYZE true", "ANALYZE false")
                plan = run_query(row["query"])[0][0][0]["Plan"]
                row["plan"] = json.dumps(plan)
                row["tables"] = json.dumps(get_all_relations(plan))
                writer.writerow(row)
                






