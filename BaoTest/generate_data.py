from generate_sql import generate_selects, generate_filters, generate_two_table_joins, table_columns
from RDS_query import run_query
from multiprocessing import Pool, Lock
from csv import writer, DictWriter, DictReader
import os
import json
from featurize import get_all_relations
import time


FIELDS = ["query", "plan", "execution_time (ms)", "tables"]

def test_gen_sql(queries):
    """
    run a bunch of queries to make sure there is no syntax errors
    """
    good = True
    for t in queries:
        print(f'query: {t}')
        run_query(t)
    return good

def test_create_data_set(queries, csv_name):
    """
    add to a csv and make sure once iterating thru the dataset, the plan is dictionary and the tables list is a list
    """
    for q in queries:
        create_data_set(q, csv_name)
    for data in dataset_iter(csv_name):
        if type(data["plan"]) != dict and type(data["tables"]) != list:
            return False
    return True


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


def get_queries(table_columns):
    queries = []
    queries.extend(generate_two_table_joins())
    queries.extend(generate_filters(table_columns))
    selects = generate_selects(table_columns)
    for table, query in selects.items():
        queries.extend(query)
    return queries


def get_explain_output(query):
    """
    :query str the SQL query
    :return the query and the explain analyze output
    """
    return query, run_query(query)[0][0][0]


def make_row_dict(query, output):
    """
    :query str the SQL query
    :output dict the explain output
    :return dict the row_dict to put into a csv
    """
    relations = list(get_all_relations([output]))
    values = [query, json.dumps(output), output["Execution Time"], json.dumps(relations)]
    row_dict = {k:v for k, v in zip(FIELDS, values)}
    return row_dict


def create_data_set(csv_name):
    """
    make a csv as a dataset
    """
    print("starting")
    start = time.time()
    queries = get_queries(table_columns)
    if os.path.exists(csv_name):
        print(f"{csv_name} already exists")
        return
    with open(csv_name, "w", newline='') as csv_file:
        csv_writer = writer(csv_file)
        csv_writer.writerow(FIELDS)
        dict_writer = DictWriter(csv_file, fieldnames=FIELDS)
        pool = Pool(processes=4)
        result = pool.imap_unordered(get_explain_output, queries)
        i = 1
        for query, output in result:
            print(f"{i}/{len(queries)} done so far")
            i += 1
            row_dict = make_row_dict(query, output)
            dict_writer.writerow(row_dict)
    print("DONE!")
    return f"{int(time.time()-start)} seconds elapsed"


if  __name__ == "__main__":
    pass

    #all queries ran: the point was to test all possible selects that could be made following the logic flow of each function
    test_select_1 = ['EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select n_name from nation ;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select MAX(n_name) from nation ;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select n_name from nation order by n_name ;']
    test_select_2 = ['EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select n_name, n_nationkey, n_comment from nation ;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select n_name,n_comment, MAX(n_nationkey) from nation GROUP BY n_name,n_comment;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select n_name, n_nationkey, n_comment from nation order by n_nationkey ;']  
    test_filter_1 = ["EXPLAIN (ANALYZE true, COSTS true, FORMAT json) select n_name from nation            where n_name > 'ALGERIA                  ' ;", "EXPLAIN (ANALYZE true, COSTS true, FORMAT json) select count(n_name) from nation            where n_name > 'ALGERIA                  ' ;", "EXPLAIN (ANALYZE true, COSTS true, FORMAT json) select n_name from nation            where n_name > 'ALGERIA                  ' order by n_name ;", "EXPLAIN (ANALYZE true, COSTS true, FORMAT json) select count(n_name) from nation            where n_name > 'ALGERIA                  ' group by n_name ;"]
    test_filter_2 = ['EXPLAIN (ANALYZE true, COSTS true, FORMAT json) select n_nationkey from nation            where n_nationkey > 0.0 ;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json) select count(n_nationkey) from nation            where n_nationkey > 0.0 ;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json) select n_nationkey from nation            where n_nationkey > 0.0 order by n_nationkey ;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json) select count(n_nationkey) from nation            where n_nationkey > 0.0 group by n_nationkey ;']
    test_join_1 = ['EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select * from part        right join partsupp on part.p_partkey=partsupp.ps_partkey        where part.p_partkey > 1.0 and partsupp.ps_partkey > 12.0 ;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select Count(*) from part    right join partsupp on part.p_partkey=partsupp.ps_partkey    where part.p_partkey > 1.0 and partsupp.ps_partkey > 12.0 ;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select * from part    right join partsupp on part.p_partkey=partsupp.ps_partkey    where part.p_partkey > 1.0 and partsupp.ps_partkey > 12.0 order by p_partkey ;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select * from part    right join partsupp on part.p_partkey=partsupp.ps_partkey    where part.p_partkey > 1.0 and partsupp.ps_partkey > 12.0 order by ps_partkey ;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select Max(ps_partkey) from part    right join partsupp on part.p_partkey=partsupp.ps_partkey    where part.p_partkey > 1.0 and partsupp.ps_partkey > 12.0 group by p_partkey ;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select Max(p_partkey) from part    right join partsupp on part.p_partkey=partsupp.ps_partkey    where part.p_partkey > 1.0 and partsupp.ps_partkey > 12.0 group by ps_partkey ;']
    test_join_2 = ['EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select * from part        right join partsupp on part.p_partkey=partsupp.ps_partkey ;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select Count(*) from part        right join partsupp on part.p_partkey=partsupp.ps_partkey ;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select * from part    right join partsupp on part.p_partkey=partsupp.ps_partkey    order by ps_partkey ;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select * from part    right join partsupp on part.p_partkey=partsupp.ps_partkey    order by p_partkey ;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select Max(ps_partkey) from part    right join partsupp on part.p_partkey=partsupp.ps_partkey    group by p_partkey ;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select Max(p_partkey) from part    right join partsupp on part.p_partkey=partsupp.ps_partkey    group by ps_partkey ;']

    # results = [test_gen_sql(i) for i in [test_select_1, test_select_2, test_filter_1, test_filter_2, test_join_1, test_join_2]]
    # print(f"Result of my tests: {all(results)}")
    
    create_data_set("data_v7.csv")


    
    

    
