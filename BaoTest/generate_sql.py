from RDS_query import run_query
import json
import sys
import itertools
import random
import math
import os
import datetime

BUCKET_SIZE = 20

join_types = ["right", "left", "inner"]

prefixs = {'part': "p_", 'customer': "c_", "lineitem": "l_", "nation": "n_", "orders": "o_", "partsupp": "ps_", "region": "r_", "supplier": "s_"}

joins = {('part', 'partsupp'): (["part.p_partkey=partsupp.ps_partkey"], ['partkey']), ('supplier', 'partsupp'): (["supplier.s_suppkey=partsupp.ps_suppkey"], ['suppkey']),
            ('customer', 'nation'): (["customer.c_nationkey=nation.n_nationkey"], ['nationkey']), ('supplier', 'nation'): (["supplier.s_nationkey=nation.n_nationkey"], ['nationkey']),
            ('customer', 'orders'): (["customer.c_custkey=orders.o_custkey"], ['custkey']),
            ('region', 'nation'): (["region.r_regionkey=nation.n_regionkey"], ['regionkey']),
            ('partsupp', 'lineitem'): (["partsupp.ps_partkey=lineitem.l_partkey", "partsupp.ps_suppkey=lineitem.l_suppkey"], ['partkey', 'suppkey']), ('orders', 'lineitem'): (["orders.o_orderkey=lineitem.l_orderkey"], ['orderkey'])}

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
        if hist_bounds: hist_bounds = get_set_from_string(hist_bounds)
        if common_vals: common_vals = get_set_from_string(common_vals)
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
# print(f'tables: {tables}')
table_columns = {}
stats_dict = {}


# print("\n\n\n\n\n\n")
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

print(stats_dict.keys())
print(stats_dict['nation'].keys())
print("stats dict made!")
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
            ret_list.append(val.strip().replace("\"", ""))
    return ret_list
        

def get_percentiles(table, column):
    percentiles = []
    
    if stats_dict[table][column][0] is not None:
        #print(table, column, stats_dict[table][column])
        buckets = get_set_from_string(stats_dict[table][column][0])
        min_val = buckets[0]
        max_val = buckets[-1]
        bucket_increment = math.ceil((len(buckets) - 1) / BUCKET_SIZE)
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
    base  = "EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select {} from {} ;"
    base_max_1 = "EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select MAX({}) from {} ;"
    base_max_mult = "EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select {}, MAX({}) from {} GROUP BY {};"
    base_order = "EXPLAIN (ANALYZE true, COSTS true, FORMAT json) Select {} from {} order by {} ;"
    sqls = {}
    for table in table_columns:
        sqls[table] = []
        columns = table_columns[table] 
        subsets = get_column_subsets(columns)
        for n in subsets:
            for subset in subsets[n]:
                if n == 0:
                    continue
                elif n == 1:
                    filled_base_max = base_max_1.format(subset, table)
                    filled_base_order = base_order.format(subset, table, subset)
                else:
                    cols = [i.strip() for i in subset.split(",")]
                    col = random.choice(cols)
                    rest = ','.join([i for i in cols if i != col])
                    filled_base_max = base_max_mult.format(rest, col, table, rest)
                    filled_base_order = base_order.format(subset, table, col)

                filled_base = base.format(subset, table)
                sqls[table].extend([filled_base, filled_base_max, filled_base_order])
               
  
    return sqls

def generate_filters(table_columns):
    sqls = []
    base = "EXPLAIN (ANALYZE true, COSTS true, FORMAT json) select {} from {}\
            where {} > {} ;"
    base_agg = "EXPLAIN (ANALYZE true, COSTS true, FORMAT json) select count({}) from {}\
            where {} > {} ;"

    base_agg_order = "EXPLAIN (ANALYZE true, COSTS true, FORMAT json) select {} from {}\
            where {} > {} order by {} ;"

    base_agg_group = "EXPLAIN (ANALYZE true, COSTS true, FORMAT json) select count({}) from {}\
            where {} > {} group by {} ;"
    
    base_max = "EXPLAIN (ANALYZE true, COSTS true, FORMAT json) select Max({}) from {}\
            where {} > {} ;"

    for table in table_columns:
        for column in table_columns[table]:
            percent_inc, percentiles_table, min_val, max_val = get_percentiles(table, column)
            if percentiles_table is None:
                continue
            for val in percentiles_table:
                if type(val) != float:
                    query = base.format(column, table, column, "'" + val + "'")
                    query_agg = base_agg.format(column, table, column, "'" + val + "'")
                    query_agg_order = base_agg_order.format(column, table, column, "'" + val + "'", column)
                    query_agg_group = base_agg_group.format(column, table, column, "'" + val + "'", column)
                    sqls.extend([query, query_agg, query_agg_order, query_agg_group])

                else:
                    query = base.format(column, table, column, val)
                    query_agg = base_agg.format(column, table, column, val)
                    query_agg_order = base_agg_order.format(column, table, column, val, column)
                    query_agg_group = base_agg_group.format(column, table, column, val, column)
                    sqls.extend([query, query_agg, query_agg_order, query_agg_group])
                    


    return sqls
    

def generate_two_table_joins():
    sqls = []
    base_filter = "EXPLAIN (ANALYZE true, COSTS true, FORMAT json, BUFFERS true) Select {} from {}\
        {} join {} on {}\
        where {}.{} > {} and {}.{} > {} ;"

    base_filter_agg = "EXPLAIN (ANALYZE true, COSTS true, FORMAT json, BUFFERS true) Select Count(*) from {}\
    {} join {} on {}\
    where {}.{} > {} and {}.{} > {} ;"

    base_filter_order = "EXPLAIN (ANALYZE true, COSTS true, FORMAT json, BUFFERS true) Select {} from {}\
    {} join {} on {}\
    where {}.{} > {} and {}.{} > {} order by {} ;"

    base_filter_max = "EXPLAIN (ANALYZE true, COSTS true, FORMAT json, BUFFERS true) Select Max({}) from {}\
    {} join {} on {}\
    where {}.{} > {} and {}.{} > {} group by {} ;"

    base_no_filter = "EXPLAIN (ANALYZE true, COSTS true, FORMAT json, BUFFERS true) Select {} from {}\
        {} join {} on {} ;"
    
    base_no_filter_agg = "EXPLAIN (ANALYZE true, COSTS true, FORMAT json, BUFFERS true) Select Count(*) from {}\
        {} join {} on {} ;"
    
    base_no_filter_order = "EXPLAIN (ANALYZE true, COSTS true, FORMAT json, BUFFERS true) Select {} from {}\
    {} join {} on {}\
    order by {} ;"

    base_no_filter_max = "EXPLAIN (ANALYZE true, COSTS true, FORMAT json, BUFFERS true) Select Max({}) from {}\
    {} join {} on {}\
    group by {} ;"



    for join in joins:
        actual_join, columns_involved = joins[join]
        for i in range(len(actual_join)):
            join_statement = actual_join[i]
            column = columns_involved[i]
            project = f"{prefixs[join[0]] + column}, {prefixs[join[1]] + column}"
            for join_type in join_types:
                query = base_no_filter.format(project, join[0], join_type, join[1], join_statement)
                query_agg = base_no_filter_agg.format(join[0], join_type, join[1], join_statement)
                query_order_0 = base_no_filter_order.format(project, join[0], join_type, join[1], join_statement, prefixs[join[1]]+column)
                query_max_0 = base_no_filter_max.format(prefixs[join[1]]+column, join[0], join_type, join[1], join_statement, prefixs[join[0]]+column)
                sqls.extend([query, query_agg, query_order_0, query_max_0])
                # print([query, query_agg, query_order_0, query_max_0])
                # return
             
            percent_inc1, percentiles_table_1, min_val_1, max_val_1 = get_percentiles(join[0], prefixs[join[0]] + column)
            percent_inc_2, percentiles_table_2, min_val_2, max_val_2 = get_percentiles(join[1], prefixs[join[1]] + column)
            if percentiles_table_1 is None or percentiles_table_2 is None:
                continue
            n1 = len(percentiles_table_1)//4
            tab1_vals = [min_val_1, percentiles_table_1[n1], percentiles_table_1[n1*2], percentiles_table_1[n1*3], max_val_1]
            n2 = len(percentiles_table_2)//4
            tab2_vals = [min_val_2, percentiles_table_2[n2], percentiles_table_2[n2*2], percentiles_table_2[n2*3], max_val_2]
            tab1_vals = [elm for i, elm in enumerate(percentiles_table_1) if i % 3 == 0]
            tab2_vals = [elm for i, elm in enumerate(percentiles_table_2) if i % 3 == 0]

            #print("len of tab 1: {}, len of tab2: {}".format(len(tab1_vals), len(tab2_vals)))
            for val1 in percentiles_table_1:#tab1_vals:
                for val2 in percentiles_table_2:#
                    for join_type in join_types:
                        if type(val1) != float:
                            val1 = "'" + val1 + "'"
                        if type(val2) != float:
                            val2 = "'" + val2 + "'"
                        query = base_filter.format(project, join[0], join_type, join[1], join_statement, join[0], prefixs[join[0]] + column, val1, join[1], prefixs[join[1]] + column, val2)#columns = 
                        query_agg = base_filter_agg.format(join[0], join_type, join[1], join_statement, join[0], prefixs[join[0]]+column, val1, join[1], prefixs[join[1]]+column, val2) 
                        query_order_0 = base_filter_order.format(project, join[0], join_type, join[1], join_statement, join[0], prefixs[join[0]]+column, val1, join[1], prefixs[join[1]]+column, val2, prefixs[join[0]]+column)
                        query_max_0 = base_filter_max.format(prefixs[join[1]]+column, join[0], join_type, join[1], join_statement, join[0], prefixs[join[0]]+column, val1, join[1], prefixs[join[1]]+column, val2, prefixs[join[0]]+column)
                        sqls.extend([query, query_agg, query_order_0, query_max_0])
                        # if (join[0] != "lineitem" and join[1] != "lineitem"):
                        #     print([query, query_agg, query_order_0, query_max_0])
                        #     return
                        

    return sqls
            
        
def get_done_queries():
    queries = set()
    with open("trial_one_data.csv", 'r') as f:
        first = f.readline()
        while True:
            line = f.readline()
            if line == "":
                break
            queries.add(line.split(";")[0][1:] + ";")
    return queries
            

if __name__ == "__main__":
    pass