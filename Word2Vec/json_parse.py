from RDS_query import connect, run_query 
import json
import os 


"""
only run when trying to get a sample test suite of json files to parse
"""
def dump_to_json(name, query, folder):

    filename = '{}.json'.format(name)
    dest = os.path.join(folder, filename)

    if not os.path.exists(dest): 
        print("executing {}".format(name))
        conn.execute(query)
        json_output = conn.fetchall()
        with open(dest, "w") as f:
            json.dump(json_output, f, indent=4)
        print('dumped {} to json'.format(name))
    else:
        print('{} already exists'.format(filename))


"""
do a dfs on the plan portion and perform a function on each node in the plan
"""
def explore_plan(plan, function, keys=False):
    if "Plans" not in plan:
        return function(plan)
    else:
        result = function(plan)
        if not keys:
            result["Plans"] = []
        for subplan in plan["Plans"]:
            if not keys:
                result["Plans"].append(explore_plan(subplan, function, keys))
            else:
                result |= explore_plan(subplan, function, keys)
        return result

"""
get keys from explain output, right now just getting keys from 8 explain test queries run, should run fine
"""
def get_keys(foldername):
    folder = "{}/{}".format(os.getcwd(), foldername)
    json_files = [pos_json for pos_json in os.listdir(folder) if pos_json.endswith('.json')]
    keys = set()
    # filename for keys hardcoded, this is ok for our purposes
    filename = "keys.json"
    if not os.path.exists(filename):

        for explain in json_files:
            with open(os.path.join(folder, explain)) as json_file:
                json_text = json.load(json_file)
                plan = json_text[0][0][0]["Plan"]
                print('exploring {}'.format(explain))
                keys |= explore_plan(plan, lambda x : set(x.keys()), True)
        print("got all keys")
        with open(filename, "w") as f:
            json.dump(list(keys), f)
            print("keys dumped to json")
        return keys
    else:
        print("file exists")
        with open(filename) as json_file:
            return set(json.load(json_file))


"""
parse plan node in postgres explain analyze json output
"""
def parse_node(node, keys):
   return {i:node[i] for i in keys if i in node}

"""
return a dict with the plan, each node_type, operation, and postgres
"""
def parse_plan(plan):

    res = {}
    res["Execution Time"] = plan["Execution Time"]
    # foldername param for get_keys is hardcoded, fine for our purposes
    keys = get_keys("explain_json")
    keys.remove("Plans")

    res["Plan"] = explore_plan(plan["Plan"], lambda x: parse_node(x, keys), False)

    return res

       
def get_explains(tests, foldername):
    folder = "{}/{}".format(os.getcwd(), foldername)
    if not os.path.exists(folder):
        os.mkdir(folder)

    for name, query in tests.items():
        dump_to_json(name, query, folder)

if __name__ == '__main__':
    # query partitions
    # select query
    # join
    # aggregate on 1 table
    # aggregate on join
    # join filter
    # projection
    # (ANALYZE true, COSTS true, FORMAT json) 
    project_select = "EXPLAIN (COSTS true, FORMAT json) select l_partkey, l_orderkey from lineitem where l_partkey=12821"
    select = "EXPLAIN (COSTS true, FORMAT json) select * from lineitem where l_partkey=12821"

    project_join =  "EXPLAIN (COSTS true, FORMAT json) select p_partkey, p_name, ps_supplycost from partsupp, part where part.p_partkey=partsupp.ps_partkey"
    join = "EXPLAIN (COSTS true, FORMAT json) select * from partsupp, part where part.p_partkey=partsupp.ps_partkey"

    project_join_filter = "EXPLAIN (COSTS true, FORMAT json) select customer.c_name, orders.o_custkey from customer, orders where customer.c_custkey=orders.o_custkey and c_custkey <= 15470 and c_custkey >= 10000;"
    join_filter = "EXPLAIN (COSTS true, FORMAT json) select customer.c_name, orders.o_custkey from customer, orders where customer.c_custkey=orders.o_custkey and c_custkey <= 15470 and c_custkey >= 10000;"


    aggregate = "EXPLAIN (COSTS true, FORMAT json) select p_name, count(*) from part group by p_name;"
    aggregate_simple = "EXPLAIN (COSTS true, FORMAT json) select count(*) from part;"

    aggregate_join = "EXPLAIN (COSTS true, FORMAT json) Select max(c_acctbal) from customer, nation where customer.c_nationkey=nation.n_nationkey group by n_nationkey"

    tests = {
        'project_select':project_select,
        'select': select,
        'project_join': project_join,
        'join': join,
        'project_join_filter': project_join_filter,
        'join_filter': join_filter,
        'aggregate_simple': aggregate_simple,
        'aggregate' : aggregate,
        'aggregate_join': aggregate_join
    }


    conn = connect()
    print("connected")

    # get_explains(tests, "explain_json")

    folder = "{}/{}".format(os.getcwd(), "json")
    with open(os.path.join(folder, "project_join_filter.json")) as json_file:
        text = json.load(json_file)
        plan = text[0][0][0]
        json_output = parse_plan(plan)
        with open("test_parse1.json", "w") as f:
            json.dump(json_output, f, indent=4)
        print('dumped test_parse.json to json')