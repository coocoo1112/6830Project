# from generate_sql import tables
import numpy as np
from featurize import JOIN_TYPES, LEAF_TYPES
import re
import json
from RDS_query import run_query
# from generate_sql import columns_query 
from dataset_generator import dataset_iter

ALL_TABLES = ["customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier"]
COLUMNS = ['c_custkey', 'c_name', 'c_address', 'c_nationkey', 'c_phone', 'c_acctbal', 'c_mktsegment', 'c_comment', 'l_orderkey', 'l_partkey', 'l_suppkey', 'l_linenumber', 'l_quantity', 'l_extendedprice', 'l_discount', 'l_tax', 'l_returnflag', 'l_linestatus', 'l_shipdate', 'l_commitdate', 'l_receiptdate', 'l_shipinstruct', 'l_shipmode', 'l_comment', 'n_nationkey', 'n_name', 'n_regionkey', 'n_comment', 'o_orderkey', 'o_custkey', 'o_orderstatus', 'o_totalprice', 'o_orderdate', 'o_orderpriority', 'o_clerk', 'o_shippriority', 'o_comment', 'p_partkey', 'p_name', 'p_mfgr', 'p_brand', 'p_type', 'p_size', 'p_container', 'p_retailprice', 'p_comment', 'ps_partkey', 'ps_suppkey', 'ps_availqty', 'ps_supplycost', 'ps_comment', 'r_regionkey', 'r_name', 'r_comment', 's_suppkey', 's_name', 's_address', 's_nationkey', 's_phone', 's_acctbal', 's_comment']

def join_matrix(plan):
    """
    :plan dict the explain output from postgres
    :returns a 1d np array 1 hot encoding indicating if tables are joined in a query
    """
    index_pos = {}
    ix = 0
    for i in range(len(ALL_TABLES)):
        for j in range(i+1, len(ALL_TABLES)):
            index_pos[f'{ALL_TABLES[i]}_{ALL_TABLES[j]}'] = ix
            ix += 1
    ans = [0 for i in range(ix)]

    def encode_joins(join_type, node):
        """
        :join_type is the join type
        :child is the node in the postgreSQL tree explain output
        """
        if join_type == "Hash Join" or join_type == "Merge Join":
            cond = "Hash Cond" if "Hash Cond" in node else "Merge Cond"
            joined_tables = [ i for i in re.split('[.= ]', node[cond][1:-1]) if i in ALL_TABLES]
            key = f'{joined_tables[0]}_{joined_tables[1]}' if f'{joined_tables[0]}_{joined_tables[1]}' in index_pos else f'{joined_tables[1]}_{joined_tables[0]}'
            ans[index_pos[key]] = 1
        elif join_type == "Nested Loop":
            # TODO: if it's a hash/merge, the subtables will be joined, do I join all tables involved?
            # can join anything joins, scans, etc
            # haven't seen nested loops joins yet in the csv
            pass
        else:
            # bao tree encoding code only considers three join types, this would be a significant refactor
            raise NameError(f'{join_type} join needs to be accounted for')

    def recurse(plan):        
        node_type = plan["Node Type"]
        if node_type in JOIN_TYPES:
            encode_joins(node_type, plan)
        
        if "Plans" not in plan:
            return

        for child in plan["Plans"]:
            recurse(child)

    recurse(plan["Plan"])
    return np.array(ans)


def get_selectivity(node):
    """
    estimate selectivity of filter of a node using uniformity assumptions using postgres estimations
    :node is a node in the postgreSQL node plan tree
    """
    filter_ = node['Filter']
    table_ = node["Relation Name"]
    est_rows_ = node["Plan Rows"]
    with open(f"../table_info/{table_}_table_stats.json", 'r') as f:
        stats = json.load(f)
        return est_rows_/stats[f"{table_}_rows"]

def get_col_name(filt):
    """
    parse through filter and get the column name
    """
    filt = re.split("[()::><=]", filt)
    for f in filt:
        if f.strip() in COLUMNS:
            return f.strip()

def histogram_encoding(plan):
    """
    :plan explain from postgres
    :return histogram encoding of predicates in the query plan
    """
    index_mapping = {c: i for i, c in enumerate(COLUMNS)}
    ans = [0 for _ in range(len(COLUMNS))]

    def recurse(plan):
        if plan["Node Type"] in LEAF_TYPES:
            if "Filter" in plan:
                ans[index_mapping[get_col_name(plan["Filter"])]] = get_selectivity(plan)
                
        if "Plans" not in plan:
            return

        for child in plan["Plans"]:
            recurse(child)

    recurse(plan["Plan"])
    return np.array(ans)

    


if __name__ == "__main__":
    # test of join matrix
    # with open("../json/aggregate_join.json", "r") as w:
    #     f = json.load(w)[0][0][0]["Plan"]
    #     print(join_matrix(f))

    # get all columns
    # columns = []
    # for t in ALL_TABLES:
    #     columns.extend([i[0] for i in run_query(columns_query.format(t))])
    # print(columns)
    good = True
    for i, data in enumerate(dataset_iter("data_v3.csv")):
        print("-----------")
        print(i)
        try:
            print(join_matrix(data["plan"]))
            print()
            print(histogram_encoding(data["plan"]))
        except:
            print(f"failed on this plan {data['plan']}")
            good = False
        print("-----------")
    print(f"Result of this test was {good}")
    