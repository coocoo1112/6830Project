from generate_sql import generate_selects, generate_filters, generate_two_table_joins, table_columns
from RDS_query import run_query
from multiprocessing import Pool, Lock
from csv import writer, DictWriter, DictReader
import os
import json
from featurize import get_all_relations
import time
from data_utils import dataset_iter


FIELDS = ["query", "plan", "execution_time (ms)", "tables"]
FAILED = []


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


def get_queries():
    queries = []
    queries.extend(generate_two_table_joins())
    # queries.extend(generate_filters(table_columns))
    # selects = generate_selects(table_columns)
    # for table, query in selects.items():
    #     queries.extend(query)
    return queries


def get_explain_output(query):
    """
    :query str the SQL query
    :return the query and the explain analyze output
    """
    try:
        return query, run_query(query)[0][0][0]
    except:
        FAILED.append(query)
        return query, None


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
    queries = get_queries()
    if os.path.exists(csv_name):
        print(f"{csv_name} already exists")
        return
    with open(csv_name, "w", newline='') as csv_file:
        csv_writer = writer(csv_file)
        csv_writer.writerow(FIELDS)
        dict_writer = DictWriter(csv_file, fieldnames=FIELDS)
        pool = Pool(processes=4)
        result = pool.imap_unordered(get_explain_output, queries[len(queries)//2:])
        i = 1
        # for query in queries:
        #     q, plan = get_explain_output(query)
        #     dict_writer.writerow(make_row_dict(q, plan))
        #     print(f"{i}/{len(queries)} done so far")
        #     i += 1
        for query, output in result:
            if not output:
                continue
            print(f"{i}/{len(queries)//2} done so far")
            i += 1
            row_dict = make_row_dict(query, output)
            dict_writer.writerow(row_dict)
    print("DONE!")
    for q in FAILED:
        print(f"query failed: {q}")
    return f"{int(time.time()-start)} seconds elapsed"


if  __name__ == "__main__":
    pass

    #all queries ran: the point was to test all possible selects that could be made following the logic flow of each function
    # test_select_1 = ['EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select n_name from nation ;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select MAX(n_name) from nation ;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select n_name from nation order by n_name ;']
    # test_select_2 = ['EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select n_name, n_nationkey, n_comment from nation ;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select n_name,n_comment, MAX(n_nationkey) from nation GROUP BY n_name,n_comment;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select n_name, n_nationkey, n_comment from nation order by n_nationkey ;']  
    # test_filter_1 = ["EXPLAIN (ANALYZE true, COSTS true, FORMAT json) select n_name from nation            where n_name > 'ALGERIA                  ' ;", "EXPLAIN (ANALYZE true, COSTS true, FORMAT json) select count(n_name) from nation            where n_name > 'ALGERIA                  ' ;", "EXPLAIN (ANALYZE true, COSTS true, FORMAT json) select n_name from nation            where n_name > 'ALGERIA                  ' order by n_name ;", "EXPLAIN (ANALYZE true, COSTS true, FORMAT json) select count(n_name) from nation            where n_name > 'ALGERIA                  ' group by n_name ;"]
    # test_filter_2 = ['EXPLAIN (ANALYZE true, COSTS true, FORMAT json) select n_nationkey from nation            where n_nationkey > 0.0 ;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json) select count(n_nationkey) from nation            where n_nationkey > 0.0 ;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json) select n_nationkey from nation            where n_nationkey > 0.0 order by n_nationkey ;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json) select count(n_nationkey) from nation            where n_nationkey > 0.0 group by n_nationkey ;']
    # test_join_1 = ['EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select * from part        right join partsupp on part.p_partkey=partsupp.ps_partkey        where part.p_partkey > 1.0 and partsupp.ps_partkey > 12.0 ;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select Count(*) from part    right join partsupp on part.p_partkey=partsupp.ps_partkey    where part.p_partkey > 1.0 and partsupp.ps_partkey > 12.0 ;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select * from part    right join partsupp on part.p_partkey=partsupp.ps_partkey    where part.p_partkey > 1.0 and partsupp.ps_partkey > 12.0 order by p_partkey ;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select * from part    right join partsupp on part.p_partkey=partsupp.ps_partkey    where part.p_partkey > 1.0 and partsupp.ps_partkey > 12.0 order by ps_partkey ;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select Max(ps_partkey) from part    right join partsupp on part.p_partkey=partsupp.ps_partkey    where part.p_partkey > 1.0 and partsupp.ps_partkey > 12.0 group by p_partkey ;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select Max(p_partkey) from part    right join partsupp on part.p_partkey=partsupp.ps_partkey    where part.p_partkey > 1.0 and partsupp.ps_partkey > 12.0 group by ps_partkey ;']
    # test_join_2 = ['EXPLAIN (ANALYZE true, COSTS true, FORMAT json, BUFFERS true) Select * from part        right join partsupp on part.p_partkey=partsupp.ps_partkey ;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select Count(*) from part        right join partsupp on part.p_partkey=partsupp.ps_partkey ;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select * from part    right join partsupp on part.p_partkey=partsupp.ps_partkey    order by ps_partkey ;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select * from part    right join partsupp on part.p_partkey=partsupp.ps_partkey    order by p_partkey ;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select Max(ps_partkey) from part    right join partsupp on part.p_partkey=partsupp.ps_partkey    group by p_partkey ;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select Max(p_partkey) from part    right join partsupp on part.p_partkey=partsupp.ps_partkey    group by ps_partkey ;']

    # results = [test_gen_sql(i) for i in [test_select_1, test_select_2, test_filter_1, test_filter_2, test_join_1, test_join_2]]
    # print(f"Result of my tests: {all(results)}")
    test_join_1 = ['EXPLAIN (ANALYZE true, COSTS true, FORMAT json, BUFFERS true) Select p_partkey, ps_partkey from part        right join partsupp on part.p_partkey=partsupp.ps_partkey        where part.p_partkey > 36.0 and partsupp.ps_partkey > 376.0 ;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json, BUFFERS true) Select Count(*) from part    right join partsupp on part.p_partkey=partsupp.ps_partkey    where part.p_partkey > 36.0 and partsupp.ps_partkey > 376.0 ;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json, BUFFERS true) Select p_partkey, ps_partkey from part    right join partsupp on part.p_partkey=partsupp.ps_partkey    where part.p_partkey > 36.0 and partsupp.ps_partkey > 376.0 order by p_partkey ;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json, BUFFERS true) Select Max(ps_partkey) from part    right join partsupp on part.p_partkey=partsupp.ps_partkey    where part.p_partkey > 36.0 and partsupp.ps_partkey > 376.0 group by p_partkey ;']
    test_join_2 = ['EXPLAIN (ANALYZE true, COSTS true, FORMAT json, BUFFERS true) Select p_partkey, ps_partkey from part        right join partsupp on part.p_partkey=partsupp.ps_partkey ;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json, BUFFERS true) Select Count(*) from part        right join partsupp on part.p_partkey=partsupp.ps_partkey ;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json, BUFFERS true) Select p_partkey, ps_partkey from part    right join partsupp on part.p_partkey=partsupp.ps_partkey    order by ps_partkey ;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json, BUFFERS true) Select Max(ps_partkey) from part    right join partsupp on part.p_partkey=partsupp.ps_partkey    group by p_partkey ;']
    # test_join_2 = []
    # results = [test_gen_sql(i) for i in [test_join_1, test_join_2]]
    # print(f"Result of my tests: {all(results)}")
    # print(len(generate_two_table_joins()))
    # print(generate_two_table_joins())
    print(create_data_set("data_v31.csv"))


    
    

    
