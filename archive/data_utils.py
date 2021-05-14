
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

# def add_rows(query, csv_writer):
    
#     output = run_query(query)[0][0][0]
#     relations = list(get_all_relations([output]))
#     values = [query, json.dumps(output), output["Execution Time"], json.dumps(relations)]
#     row_dict = {k:v for k, v in zip(FIELDS, values)}
#     csv_writer.writerow(row_dict)



def add_row(query, csv_writer):
    output = run_query(query)[0][0][0]
    relations = list(get_all_relations([output]))
    values = [query, json.dumps(output), output["Execution Time"], json.dumps(relations)]
    row_dict = {k:v for k, v in zip(FIELDS, values)}
    csv_writer.writerow(row_dict)









                






