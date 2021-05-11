# from generate_sql import tables
import numpy as np
from featurize import JOIN_TYPES
import re
import json

ALL_TABLES = ["customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier"]


def join_matrix(plan):
    """
    :plan dict the explain output from postgres
    :returns a 1d np array 1 hot encoding indicating if tables are joined in a query
    """
    encoding = {}
    for i in range(len(ALL_TABLES)):
        for j in range(i+1, len(ALL_TABLES)):
            encoding [f'{ALL_TABLES[i]}_{ALL_TABLES[j]}'] = 0
    
    def recurse(plan):
        if "Plans" not in plan:
            return
        else:
            for child in plan["Plans"]:
                node_type = child["Node Type"]
                if node_type in JOIN_TYPES:
                    if node_type == "Hash Join" or node_type == "Merge Join":
                        cond = "Hash Cond" if "Hash Cond" in child else "Merge Cond"
                        joined_tables = [ i for i in re.split('[.= ]', child[cond][1:-1]) if i in ALL_TABLES]
                        key = f'{joined_tables[0]}_{joined_tables[1]}' if f'{joined_tables[0]}_{joined_tables[1]}' in encoding else f'{joined_tables[1]}_{joined_tables[0]}'
                        encoding[key] = 1
                    elif node_type == "Nested Loop":
                        # TODO: if it's a hash/merge, the subtables will be joined, do I join all tables involved?
                        # can join anything joins, scans, etc
                        # haven't seen nested loops joins yet in the csv
                        pass
                    else:
                        # bao tree encoding code only considers three join types, this would be a significant refactor
                        raise NameError(f'{node_type} join needs to be accounted for')
                recurse(child)

    recurse(plan)
    return np.array(list(encoding.values()))

# run query to get all schema columns

def histogram_encoding(plan):
    """
    :plan explain from postgres
    :return histogram encoding of predicates in the query plan
    """
    pass

    


if __name__ == "__main__":
    # print(join_matrix(dict()))
    # k = "(customer.c_nationkey = nation.n_nationkey)"                        
    # joined_tables = re.split('[.=]', k[1:-1])
    # print(joined_tables)
    with open("../json/aggregate_join.json", "r") as w:
        f = json.load(w)[0][0][0]["Plan"]
        print(join_matrix(f))
