from generate_sql import generate_selects, generate_filters, generate_two_table_joins, table_columns
from RDS_query import run_query
from dataset_generator import create_data_set, dataset_iter, FIELDS


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

if  __name__ == "__main__":

    # all queries ran: the point was to test all possible selects that could be made following the logic flow of each function
    test_select_1 = ['EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select n_name from nation ;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select MAX(n_name) from nation ;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select n_name from nation order by n_name ;']
    test_select_2 = ['EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select n_name, n_nationkey, n_comment from nation ;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select n_name,n_comment, MAX(n_nationkey) from nation GROUP BY n_name,n_comment;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select n_name, n_nationkey, n_comment from nation order by n_nationkey ;']  
    test_filter_1 = ["EXPLAIN (ANALYZE true, COSTS true, FORMAT json) select n_name from nation            where n_name > 'ALGERIA                  ' ;", "EXPLAIN (ANALYZE true, COSTS true, FORMAT json) select count(n_name) from nation            where n_name > 'ALGERIA                  ' ;", "EXPLAIN (ANALYZE true, COSTS true, FORMAT json) select n_name from nation            where n_name > 'ALGERIA                  ' order by n_name ;", "EXPLAIN (ANALYZE true, COSTS true, FORMAT json) select count(n_name) from nation            where n_name > 'ALGERIA                  ' group by n_name ;"]
    test_filter_2 = ['EXPLAIN (ANALYZE true, COSTS true, FORMAT json) select n_nationkey from nation            where n_nationkey > 0.0 ;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json) select count(n_nationkey) from nation            where n_nationkey > 0.0 ;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json) select n_nationkey from nation            where n_nationkey > 0.0 order by n_nationkey ;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json) select count(n_nationkey) from nation            where n_nationkey > 0.0 group by n_nationkey ;']
    test_join_1 = ['EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select * from part        right join partsupp on part.p_partkey=partsupp.ps_partkey        where part.p_partkey > 1.0 and partsupp.ps_partkey > 12.0 ;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select Count(*) from part    right join partsupp on part.p_partkey=partsupp.ps_partkey    where part.p_partkey > 1.0 and partsupp.ps_partkey > 12.0 ;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select * from part    right join partsupp on part.p_partkey=partsupp.ps_partkey    where part.p_partkey > 1.0 and partsupp.ps_partkey > 12.0 order by p_partkey ;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select * from part    right join partsupp on part.p_partkey=partsupp.ps_partkey    where part.p_partkey > 1.0 and partsupp.ps_partkey > 12.0 order by ps_partkey ;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select Max(ps_partkey) from part    right join partsupp on part.p_partkey=partsupp.ps_partkey    where part.p_partkey > 1.0 and partsupp.ps_partkey > 12.0 group by p_partkey ;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select Max(p_partkey) from part    right join partsupp on part.p_partkey=partsupp.ps_partkey    where part.p_partkey > 1.0 and partsupp.ps_partkey > 12.0 group by ps_partkey ;']
    test_join_2 = ['EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select * from part        right join partsupp on part.p_partkey=partsupp.ps_partkey ;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select Count(*) from part        right join partsupp on part.p_partkey=partsupp.ps_partkey ;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select * from part    right join partsupp on part.p_partkey=partsupp.ps_partkey    order by ps_partkey ;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select * from part    right join partsupp on part.p_partkey=partsupp.ps_partkey    order by p_partkey ;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select Max(ps_partkey) from part    right join partsupp on part.p_partkey=partsupp.ps_partkey    group by p_partkey ;', 'EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select Max(p_partkey) from part    right join partsupp on part.p_partkey=partsupp.ps_partkey    group by ps_partkey ;']

    # results = [test_gen_sql(i) for i in [test_select_1, test_select_2, test_filter_1, test_filter_2, test_join_1, test_join_2]]
    # print(f"Result of my tests: {all(results)}")
    
    

    
