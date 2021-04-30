from RDS_query import run_query
import json
import sys
import itertools
import random
import math
import os

join_types = ["right", "left", "inner", "outer"]

prefixs = {'part': "p_", 'customer': "c_", "lineitem": "l_", "nation": "n_", "orders": "o_", "partsupp": "ps_", "region": "r_", "supplier": "s_"}

joins = {('part', 'partsupp'): (["part.p_partkey=partsupp.ps_partkey"], ['partkey']), ('supplier', 'partsupp'): (["supplier.s_suppkey=partsupp.ps_suppkey"], ['suppkey']),
            ('customer', 'nation'): (["customer.c_nationkey=nation.n_nationkey"], ['nationkey']), ('supplier', 'nation'): (["supplier.s_nationkey=nation.n_nationkey"], ['nationkey']),
            ('partsupp', 'lineitem'): (["partsupp.ps_partkey=lineitem.l_partkey", "partsupp.ps_suppkey=lineitem.l_suppkey"], ['partkey', 'suppkey']), 
            ('customer', 'orders'): (["customer.c_custkey=orders.o_custkey"], ['custkey']), ('orders', 'lineitem'): (["orders.o_orderkey=lineitem.l_orderkey"], ['orderkey']),
            ('region', 'nation'): (["region.r_regionkey=nation.n_regionkey"], ['regionkey'])}


tables_query = "SELECT tablename FROM pg_catalog.pg_tables \
        WHERE schemaname != 'pg_catalog' AND \
        schemaname != 'information_schema';"

columns_query = "SELECT column_name FROM information_schema.columns\
    WHERE table_name = '{}';"

stats_get = "select histogram_bounds, most_common_vals, tablename, attname FROM pg_stats\
    where tablename in {}"

def get_table_sizes(tables):
    size_get = "select count(*) FROM {}"
    size_dict = {}
    for table in tables:
        size = run_query(size_get.format(table))[0][0]
        size_dict[table] = size
    return size_dict

def get_all_stats(tables):
    stats_get = "select tablename, attname, histogram_bounds, most_common_vals, avg_width, null_frac, n_distinct, correlation  FROM pg_stats\
        where tablename in {}"
    stats = run_query(stats_get.format(tuple(tables)))
    size_dict = get_table_sizes(tables)
    total_stats = {}
    for stat in stats:
        table = stat[0]
        column = stat[1]
        hist_bounds = stat[2]
        common_vals = stat[3]
        avg_width = stat[4]
        null_frac = stat[5]
        n_distinct = stat[6]
        correlation = stat[7]
        size = size_dict[table]
        total_stats.setdefault(table, {})
        total_stats[table].setdefault(column, {})
        total_stats[table][column] = {'hist_bounds': hist_bounds, 'common_vals': common_vals, 'avg_width': avg_width, 'null_frac': null_frac,
                                        'n_distinct': n_distinct, 'correlation': correlation, 'rows': size}
    return total_stats
    


tables = [i[0] for i in run_query(tables_query)]
table_columns = {}
stats_dict = {}


print("\n\n\n\n\n\n")
for table in tables:
    table_columns[table] = [i[0] for i in run_query(columns_query.format(table))]
    stats_dict[table] = {}


stats = run_query(stats_get.format(tuple(tables)))
for stat in stats:
    table = stat[2]
    column = stat[3]
    try:
        stats_dict[table][column] = (stat[0], stat[1])
    except:
        print(stat[0])
        sys.exit()

def get_set_from_string(set_string):
    #account for date?
    ret_list = []
    set_string = set_string[1:-1]
    vals_list = set_string.split(",")
    try:
        for val in vals_list:
            ret_list.append(float(val))
    except ValueError:
        for val in vals_list:
            ret_list.append(val[1:-1].strip())
    return ret_list
        

def get_percentiles(table, column):
    percentiles = []
    
    if stats_dict[table][column][0] is not None:
        #print(table, column, stats_dict[table][column])
        buckets = get_set_from_string(stats_dict[table][column][0])
        min_val = buckets[0]
        max_val = buckets[-1]
        bucket_increment = math.ceil((len(buckets) - 1) / 20)
        rough_percent = (len(buckets) - 1) / bucket_increment
        for i in range(0,len(buckets), bucket_increment):
            percentiles.append(buckets[i])
        return rough_percent, percentiles, min_val, max_val   
    else:
        #print(table, column, "none")
        return None, None, None, None

def findsubsets(s, n):
    #print(n, list(map(set, itertools.combinations(s, n))))
    return list(map(set, itertools.combinations(s, n)))

def get_column_subsets(columns):
    subs = {}

    for n in range(1, len(columns) + 1):
        subsets = findsubsets(columns, n)
        k = min(10, len(subsets))
        picked_subsets = random.sample(subsets, k=k)
        temp = []
        for sub in picked_subsets:
            temp.append(", ".join(sub))
        subs[n] = temp
    return subs
        

def generate_selects(table_columns):
    base  = "Select {} from {};"
    sqls = {}
    for table in table_columns:
        sqls[table] = []
        columns = table_columns[table] 
        subsets = get_column_subsets(columns)
        for n in subsets:
            for subset in subsets[n]:
                filled_base = base.format(subset, table)
                sqls[table].append(filled_base)
    return sqls

def generate_filters(table_columns):
    sqls = []
    base = "select {} from {}\
            where {} > {};"
    for table in table_columns:
        for column in table_columns[table]:
            percent_inc, percentiles_table, min_val, max_val = get_percentiles(table, column)
            if percentiles_table is None:
                continue
            for val in percentiles_table:
                sqls.append(base.format(column, table, column, val))
    return sqls
    

def generate_joins():
    sqls = []
    base_filter = "Select * from {}\
        {} join {} on {}\
        where {}.{} > {} and {}.{} > {};"
    base_no_filter = "Select * from {}\
        {} join {} on {};"
    for join in joins:
        actual_join, columns_involved = joins[join]
        for i in range(len(actual_join)):
            join_statement = actual_join[i]
            column = columns_involved[i]
            for join_type in join_types:
                sqls.append(base_no_filter.format(join[0], join_type, join[1], join_statement))
            percent_inc1, percentiles_table_1, min_val_1, max_val_1 = get_percentiles(join[0], prefixs[join[0]] + column)
            percent_inc_2, percentiles_table_2, min_val_2, max_val_2 = get_percentiles(join[1], prefixs[join[1]] + column)
            if percentiles_table_1 is None or percentiles_table_2 is None:
                continue
            n1 = len(percentiles_table_1)//4
            tab1_vals = [min_val_1, percentiles_table_1[n1], percentiles_table_1[n1*2], percentiles_table_1[n1*3], max_val_1]
            n2 = len(percentiles_table_2)//4
            tab2_vals = [min_val_2, percentiles_table_2[n2], percentiles_table_2[n2*2], percentiles_table_2[n2*3], max_val_2]
            for val1 in tab1_vals:
                for val2 in tab2_vals:
                    for join_type in join_types:
                        sqls.append(base_filter.format(join[0], join_type, join[1], join_statement, join[0], prefixs[join[0]] + column, val1, join[1], prefixs[join[1]] + column, val2)) 
    return sqls
            
        





if __name__ =="__main__":

    total_sqls = []
    table_stats = get_all_stats(tables)
    #json_stats = json.loads(table_stats)
    print("done")

    folder = "{}/{}".format(os.getcwd(), "json")
    with open("table_stats.json", "w") as f:
            json.dump(table_stats, f, indent=4)
    


    # print("start")
    # test = generate_selects(table_columns)
    # print([(i, len(test[i])) for i in test])
    # test2 = generate_joins()
    # print("step2", len(test2))
    # test3 = generate_filters(table_columns)
    # print("finished", len(test3))
    # for i in test:
    #     total_sqls += test[i]
    #     print(random.choice(test[i]))

    # total_sqls += test2
    # total_sqls += test3
    # print(random.choice(test2))
    # print(random.choice(test3))

    
    # test = get_percentiles("nation", "n_nationkey")
    # print("test")
    # print(type(test))
    # print(test)
    #print(get_column_subsets(["col1", "col2", "col3", "col4", "col5", "col6"]))
    #print("\n".join(generate_selects(table_columns)))