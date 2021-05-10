
from csv import writer, DictWriter, DictReader
import os
from RDS_query import run_query
import json
import re
import pickle


FIELDS = ["query", "plan", "execution_time (ms)"]

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
    values = [query, json.dumps(output["Plan"]), output["Execution Time"]]
    row_dict = {k:v for k, v in zip(FIELDS, values)}
    csv_writer.writerow(row_dict)
                     
if __name__ == "__main__":
    with open("data_v2.csv", "r") as f:
        r = DictReader(f, FIELDS)
        with open("data_v3.csv", "w") as d:
            w = DictWriter(d, fieldnames=FIELDS)
            csv_writer = writer(d)
            csv_writer.writerow(FIELDS)
            for i in r:
                print(json.dumps(i["plan"]))

                new_dict = i
                new_dict["plan"] = json.dumps(i["plan"])
                w.writerow(new_dict)




