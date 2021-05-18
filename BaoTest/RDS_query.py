import psycopg2 as ps
import os
# postgres_index1 is index instance
# index1 indices:
"""
    part on p_partkey
    supplier on s_suppkey
    customer on c_custkey
    orders on o_orderkey

    lineitem on l_orderkey
    partsupp on ps_suppkey

    potentially cluster

    clustered: lineitem, partsupp, customer
    all indices are tablename_idx named


"""

db_name = 'tpch-big'
#db_name = 'postgres'
credentials = {
    'POSTGRES_ADDRESS': 'tpch-big.cj2xgw6moc7i.us-east-1.rds.amazonaws.com',
    'POSTGRES_PORT': '5432', 'POSTGRES_USERNAME': 'postgres',
    'POSTGRES_PASSWORD': '6830Project', 'POSTGRES_DBNAME': db_name
}





def connect():
    os.environ["PGOPTIONS"] = '-c statement_timeout=45000'
    conn = ps.connect(host=credentials['POSTGRES_ADDRESS'],
                      database=credentials['POSTGRES_DBNAME'],
                      user=credentials['POSTGRES_USERNAME'],
                      password=credentials['POSTGRES_PASSWORD'],
                      port=credentials['POSTGRES_PORT'])
    return conn.cursor()

cursor = connect()
print("connected!")


def run_query(query):
    global cursor
    try:
        cursor.execute(query)
        return cursor.fetchall()
    except:
        cursor.close()
        cursor = connect()
        raise RuntimeError("Query took too long")

