import os.path as osp

root_path = osp.abspath(osp.dirname(__file__))
raw_data_path = osp.join(root_path, 'data')
processed_data_path = osp.join(root_path, 'data')
table_data_path = osp.join(root_path, 'data', 'table_info')
