
from csv import writer, DictWriter
import os
from RDS_query import run_query
import json


TABLES_QUERY = "SELECT tablename FROM pg_catalog.pg_tables \
    WHERE schemaname != 'pg_catalog' AND \
    schemaname != 'information_schema';"

TABLES = set([i[0] for i in run_query(TABLES_QUERY)])

TABLE_CACHE = {}

def open_table_stats(json_file):
    with open(json_file, "r") as json_output:
        return json.load(json_output)

TABLE_STATS = open_table_stats("table_stats.json")

def create_data_set(query, csv_name):
    
    query_explain_analyze = run_query(query)
    fields = ["query", "execution_time"]
    table1_fields = make_table_fields(1)
    fields.extend(table1_fields)
    table2_fields = make_table_fields(2)
    fields.extend(table2_fields)

    if not os.path.exists(csv_name):
        with open(csv_name, "w", newline='') as csv_file:
            csv_writer = writer(csv_file)
            csv_writer.writerow(fields)
            dict_writer = DictWriter(csv_file, fieldnames=fields)
            add_row(query, dict_writer)
    else:
        with open(csv_name, "a", newline="") as csv_file:
            dict_writer = DictWriter(csv_file, fieldnames=fields)
            add_row(query, dict_writer)

def make_table_fields(table_num):
    fields = ["table{}_name".format(table_num), "table{}_rows".format(table_num)]
    most_keys = get_max_num_columns("table_stats.json")
    col_stats = ["hist_bounds", "common_vals", "avg_width", "null_frac", "n_distinct", "correlation", "rows"]
    for i in range(most_keys):
        fields.append('table{}_col{}name'.format(table_num, i+1))
        for stat in col_stats:
            fields.append("table{}_col{}_{}".format(table_num, i+1, stat))
    return fields


def add_row(query, csv_writer):
    relations = [i for i in query.split() if i in TABLES]
    output = run_query(query)[0][0][0]
    row_dict = {}
    row_dict["query"] = query
    row_dict["execution_time"] = output["Execution Time"]
    rows = None

    for i, table in enumerate(relations):
        if table not in TABLE_CACHE:
            table_fields = make_table_fields(i+1)
            table_stat_dict = {"table{}_name".format(i+1) : table}
            for j, column in enumerate(TABLE_STATS[table]):
                table_stat_dict["table{}_col{}name".format(i+1, j+1)] = column
                for stat in TABLE_STATS[table][column]:
                    if stat == "rows":
                        rows = TABLE_STATS[table][column][stat]
                        continue
                    table_stat_dict["table{}_col{}_{}".format(i+1, j+1, stat)] = TABLE_STATS[table][column][stat]
            table_stat_dict["table{}_rows".format(i+1)] = rows
            TABLE_CACHE[table] = table_stat_dict
        row_dict.update(TABLE_CACHE[table])
    csv_writer.writerow(row_dict)
    




def get_max_num_columns(json_file_name):
    with open(json_file_name, "r") as json_file:
        json_text = json.load(json_file)
        most = 0
        for key in json_text:
            most = max(most, len(json_text[key]))
        return most

        
if __name__ == "__main__":
    # hi = run_query("EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select * from partsupp right join lineitem on partsupp.ps_partkey=lineitem.l_partkey;")

    # print(hi[0][0][0])

    query = "EXPLAIN (ANALYZE true, COSTS true, FORMAT json) select * from region join nation on region.r_regionkey=nation.n_regionkey;"

    create_data_set(query, "test4.csv")


