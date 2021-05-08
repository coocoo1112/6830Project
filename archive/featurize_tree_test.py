# from BaoForPostgreSQL-master.bao_server import 
from BaoForPostgreSQL.bao_server.featurize import TreeFeaturizer
import json
from csv import writer, DictWriter, DictReader
import os
from numpy import loadtxt, savetxt
import sqlite3

def get_json_plans_paths(path):
    return [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f == "aggregate_join.json"]

def get_json_plans(path):
    paths = get_json_plans_paths(path)
    plans = []
    for p in paths:
        with open(p, 'r') as f:
            json_plan = json.load(f)
            plans.append(json_plan[0][0][0])
    return plans


# def project_db():
#     conn = sqlite3.connect("bao.db")
#     c = conn.cursor()
#     c.execute("""
#     CREATE TABLE IF NOT EXISTS experience (
#         id INTEGER PRIMARY KEY,
#         pg_pid INTEGER,
#         plan TEXT, 
#         reward REAL
#     )""")

if __name__ == "__main__":
    
    plans = get_json_plans("json")
    
    tf = TreeFeaturizer()
    tf.fit(plans)

    output = tf.transform(plans)
    csv_name = "test_featurize_1.csv"
    
    # FIELDS = ["test"]
    # with open(csv_name, "w", newline='') as csv_file:

    #     csv_writer = writer(csv_file)
    #     csv_writer.writerow(FIELDS)
    #     dict_writer = DictWriter(csv_file, fieldnames=FIELDS)
    #     dict_writer.writerow({"test": json.dumps(output)})

    # with open(csv_name, "r") as f:
    #     csv_reader = DictReader(f)
    #     for row in csv_reader:
    #         hi = [row["test"]]
    #         print("HI: ", hi)
    #         X= []
    #         for x in hi:
    #             print("x: ", x)
    #             if isinstance(x, str):
    #                 print("x is string: ", x)
    #                 X.append(json.loads(x))
    #             else:
    #                 print("x is not string: ", x)
    #                 X.append(x)
    #         print(X)