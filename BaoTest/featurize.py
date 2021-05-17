import numpy as np
import json
import re
import sys
from data_utils import dataset_iter
from query_encoding import histogram_encoding, join_matrix, JOIN_TYPES, LEAF_TYPES


ALL_TABLES = ["customer", "lineitem", "nation", "orders", "part", "partsupp", "region", "supplier"]
ALL_TYPES = JOIN_TYPES + LEAF_TYPES
ROW_LENGTH = len(JOIN_TYPES) + len(LEAF_TYPES) * len(ALL_TABLES)
print("Row length:", ROW_LENGTH)



class TreeBuilderError(Exception):
    def __init__(self, msg):
        self.__msg = msg

def is_join(node):
    return node["Node Type"] in JOIN_TYPES

def is_scan(node):
    return node["Node Type"] in LEAF_TYPES


class NeoTreeBuilder:
    def __init__(self, stats_extractor, relations):
        self.__stats = stats_extractor
        self.__relations = sorted(relations, key=lambda x: len(x), reverse=True)
        print(self.__relations)

    def __relation_name(self, node):
        if "Relation Name" in node:
            return node["Relation Name"]

        #dont think we need this as we dont have bitmap scans?
        if node["Node Type"] == "Bitmap Index Scan":
            # find the first (longest) relation name that appears in the index name
            name_key = "Index Name" if "Index Name" in node else "Relation Name"
            if name_key not in node:
                print(node)
                raise TreeBuilderError("Bitmap operator did not have an index name or a relation name")
            for rel in self.__relations:
                if rel in node[name_key]:
                    return rel

            raise TreeBuilderError("Could not find relation name for bitmap index scan")

        raise TreeBuilderError("Cannot extract relation type from node")
                
    def __featurize_join(self, node):
        assert is_join(node)
        return self.encode_joins(node["Node Type"], node)
        print(enc)
        arr = np.zeros(len(ALL_TYPES))
        arr[ALL_TYPES.index(node["Node Type"])] = 1
        return arr
        return np.concatenate((arr, self.__stats(node)))

    def __featurize_scan(self, node, query_encoding):
        assert is_scan(node)
        # print(query_encoding)
        # print(self.encode_scans(node["Node Type"], node))
        first = np.concatenate((np.array(query_encoding), self.encode_scans(node["Node Type"], node)))
        return (first, self.__relation_name(node))
        arr = np.zeros(len(ALL_TYPES))
        arr[ALL_TYPES.index(node["Node Type"])] = 1
        return arr

        return (np.concatenate((arr, self.__stats(node))),
                self.__relation_name(node))
    
    def encode_scans(self, scan_type, node):
        #will have to be add in index scan logic if adding index scans
        #if scan_type == "Seq Scan" or scan_type == "Index Scan":
        table = node["Relation Name"]
        arr = np.zeros(ROW_LENGTH)
        idx = len(JOIN_TYPES) + (ALL_TABLES.index(table) * len(LEAF_TYPES)) + LEAF_TYPES.index(scan_type)
        arr[idx] = 1
        return np.concatenate((arr, self.__stats(node)))
        # else:
        #     raise TreeBuilderError("Scan type has not been accounted for", scan_type)


    def encode_joins(self, join_type, node):
        """
        :join_type is the join type
        :child is the node in the postgreSQL tree explain output
        """
        if join_type == "Hash Join" or join_type == "Merge Join":
            arr = np.zeros(ROW_LENGTH)
            arr[JOIN_TYPES.index(join_type)] = 1
            return np.concatenate((arr, self.__stats(node)))
        else:
            # bao tree encoding code only considers three join types, this would be a significant refactor
            raise NameError(f'{join_type} join needs to be accounted for')


    def plan_to_feature_tree(self, plan, query_encoding):
        # print("QUERY ENCODING SIZE: ", len(query_encoding))
        children = plan["Plans"] if "Plans" in plan else []

        if len(children) == 1:
            return self.plan_to_feature_tree(children[0], query_encoding)

        if is_join(plan):
            assert len(children) == 2
            my_vec = np.concatenate((query_encoding, self.__featurize_join(plan)))
            left = self.plan_to_feature_tree(children[0], query_encoding)
            right = self.plan_to_feature_tree(children[1], query_encoding)
            if len(left) == 2:
                left_scan = left[0][len(query_encoding) + len(JOIN_TYPES):]
            else:
                left_scan = left[len(query_encoding) + len(JOIN_TYPES):]
            if len(right) == 2:
                right_scan = right[0][len(query_encoding) + len(JOIN_TYPES):]
            else: 
                right_scan = right[len(query_encoding) + len(JOIN_TYPES):]
            new_arr = np.zeros(len(left_scan))
            for i in range(len(left_scan)):
                new_arr[i] = 1 if left_scan[i] == 1 or right_scan[i] == 1 else 0
            # print(my_vec)
            # print(new_arr)
            my_vec = np.concatenate((my_vec[:len(query_encoding)+len(JOIN_TYPES)], new_arr))
            # print("MY VEC LEN: ", len(my_vec))
            # print("Spliced vec len: ", len(new_arr))
            # print("left size: ", len(left))
            # print("right size: ", len(right))
            #my_vec = np.concatenate((query_encoding, np.concatenate((my_vec[len(query_encoding):len(query_encoding)+len(JOIN_TYPES)], new_arr))))
            return (my_vec, left, right)

        if is_scan(plan):
            assert not children
            # print(query_encoding)
            # print(self.__featurize_scan(plan))
            return self.__featurize_scan(plan, query_encoding)

        raise TreeBuilderError("Node wasn't transparent, a join, or a scan: " + str(plan))


class TreeBuilder:
    def __init__(self, stats_extractor, relations):
        self.__stats = stats_extractor
        self.__relations = sorted(relations, key=lambda x: len(x), reverse=True)
        print(self.__relations)

    def __relation_name(self, node):
        if "Relation Name" in node:
            return node["Relation Name"]

        if node["Node Type"] == "Bitmap Index Scan":
            # find the first (longest) relation name that appears in the index name
            name_key = "Index Name" if "Index Name" in node else "Relation Name"
            if name_key not in node:
                print(node)
                raise TreeBuilderError("Bitmap operator did not have an index name or a relation name")
            for rel in self.__relations:
                if rel in node[name_key]:
                    return rel

            raise TreeBuilderError("Could not find relation name for bitmap index scan")

        raise TreeBuilderError("Cannot extract relation type from node")
                
    def __featurize_join(self, node):
        assert is_join(node)
        arr = np.zeros(len(ALL_TYPES))
        arr[ALL_TYPES.index(node["Node Type"])] = 1
        return np.concatenate((arr, self.__stats(node)))

    def __featurize_scan(self, node):
        assert is_scan(node)
        arr = np.zeros(len(ALL_TYPES))
        arr[ALL_TYPES.index(node["Node Type"])] = 1
        return (np.concatenate((arr, self.__stats(node))),
                self.__relation_name(node))

    def plan_to_feature_tree(self, plan):
        children = plan["Plans"] if "Plans" in plan else []

        if len(children) == 1:
            return self.plan_to_feature_tree(children[0])

        if is_join(plan):
            assert len(children) == 2
            my_vec = self.__featurize_join(plan)
            left = self.plan_to_feature_tree(children[0])
            right = self.plan_to_feature_tree(children[1])
            return (my_vec, left, right)

        if is_scan(plan):
            assert not children
            return self.__featurize_scan(plan)

        raise TreeBuilderError("Node wasn't transparent, a join, or a scan: " + str(plan))





def norm(x, lo, hi):
    return (np.log(x + 1) - lo) / (hi - lo)

def get_buffer_count_for_leaf(leaf, buffers):
    total = 0
    if "Relation Name" in leaf:
        total += buffers.get(leaf["Relation Name"], 0)

    if "Index Name" in leaf:
        total += buffers.get(leaf["Index Name"], 0)

    return total

class StatExtractor:
    def __init__(self, fields, mins, maxs):
        self.__fields = fields
        self.__mins = mins
        self.__maxs = maxs

    def __call__(self, inp):
        res = []
        for f, lo, hi in zip(self.__fields, self.__mins, self.__maxs):
            if f not in inp:
                res.append(0)
            else:
                res.append(norm(inp[f], lo, hi))
        return res

def get_plan_stats(data):
    costs = []
    rows = []
    bufs = []
    
    def recurse(n, buffers=None):
        costs.append(n["Total Cost"])
        rows.append(n["Plan Rows"])
        if "Buffers" in n:
            bufs.append(n["Buffers"])

        if "Plans" in n:
            for child in n["Plans"]:
                recurse(child)

    for plan in data:
        recurse(plan["Plan"], buffers=plan.get("Buffers", None))

    costs = np.array(costs)
    rows = np.array(rows)
    bufs = np.array(bufs)
    
    costs = np.log(costs + 1)
    rows = np.log(rows + 1)
    bufs = np.log(bufs + 1)

    costs_min = np.min(costs)
    costs_max = np.max(costs)
    rows_min = np.min(rows)
    rows_max = np.max(rows)
    bufs_min = np.min(bufs) if len(bufs) != 0 else 0
    bufs_max = np.max(bufs) if len(bufs) != 0 else 0

    if len(bufs) != 0:
        return StatExtractor(
            ["Buffers", "Total Cost", "Plan Rows"],
            [bufs_min, costs_min, rows_min],
            [bufs_max, costs_max, rows_max]
        )
    else:
        return StatExtractor(
            ["Total Cost", "Plan Rows"],
            [costs_min, rows_min],
            [costs_max, rows_max]
        )
        

def get_all_relations(data):
    all_rels = []
    
    def recurse(plan):
        if "Relation Name" in plan:
            yield plan["Relation Name"]

        if "Plans" in plan:
            for child in plan["Plans"]:
                yield from recurse(child)

    for plan in data:
        all_rels.extend(list(recurse(plan["Plan"])))
        
    return set(all_rels)

def get_featurized_trees(data):
    all_rels = get_all_relations(data)
    stats_extractor = get_plan_stats(data)

    t = TreeBuilder(stats_extractor, all_rels)
    trees = []

    for plan in data:
        tree = t.plan_to_feature_tree(plan)
        trees.append(tree)
            
    return trees

def _attach_buf_data(tree):
    if "Buffers" not in tree:
        return

    buffers = tree["Buffers"]

    def recurse(n):
        if "Plans" in n:
            for child in n["Plans"]:
                recurse(child)
            return
        
        # it is a leaf
        n["Buffers"] = get_buffer_count_for_leaf(n, buffers)

    recurse(tree["Plan"])

def get_query_enc(plan):
    return np.concatenate((join_matrix(plan), histogram_encoding(plan)))



class TreeFeaturizer:
    def __init__(self):
        self.__tree_builder = None

    def fit(self, trees):
        for t in trees:
            _attach_buf_data(t)
        # print(trees[0])
        # sys.exit()
        all_rels = get_all_relations(trees)
        stats_extractor = get_plan_stats(trees)
        self.__tree_builder = TreeBuilder(stats_extractor, all_rels)

    def transform(self, trees):
        for t in trees:
            _attach_buf_data(t)
        return [self.__tree_builder.plan_to_feature_tree(x["Plan"]) for x in trees]

    def num_operators(self):
        return len(ALL_TYPES)


class NeoTreeFeaturizer:
    def __init__(self):
        self.__tree_builder = None

    def fit(self, trees):
        for t in trees:
            _attach_buf_data(t)
        # print(trees[0])
        # sys.exit()
        all_rels = get_all_relations([data[0] for data in trees])
        stats_extractor = get_plan_stats([data[0] for data in trees])
        self.__tree_builder = NeoTreeBuilder(stats_extractor, all_rels)

    def transform(self, trees):
        # for t in trees:
        #     _attach_buf_data(t)
        return [self.__tree_builder.plan_to_feature_tree(x[0]["Plan"], get_query_enc(x[1])) for x in trees]


    def num_operators(self):
        return len(ALL_TYPES)
        

if __name__ == "__main__":
    print("test")
    neo = NeoTreeBuilder()
    csv_file = "data_v3.csv"
    pairs = []
    fail = 0
    failed_plans = []
    i = 1
    for row in dataset_iter(csv_file):
        print(i)
        i += 1
        pairs = [row["plan"], row["execution_time (ms)"]]
        plan = pairs[0]
        node_type = plan["Plan"]["Node Type"]
        try:
            print(plan)
            # print(histogram_encoding(plan))
            # print(join_matrix(plan))
            query_enc = np.concatenate((join_matrix(plan), histogram_encoding(plan)))
            print(len(join_matrix(plan)))
            print(len(histogram_encoding(plan)))
            print(len(query_enc))
            neo.plan_to_feature_tree(plan["Plan"], query_enc)
                
        except Exception as e:
            print(e)
            print(plan)
            sys.exit()
            failed_plans.append(plan)
            fail += 1
    
    print(f'there were {fail} failures')
    # for i in range(len(res)):
    #     print(len(res[i]))
    # print(ROW_LENGTH)
