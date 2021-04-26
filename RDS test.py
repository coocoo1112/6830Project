print("test")

import psycopg2 as ps
# define credentials
credentials = {
    'POSTGRES_ADDRESS': 'db-test.ch9w9rkl1agb.us-east-2.rds.amazonaws.com',  # change to your endpoint
    'POSTGRES_PORT': '5432',  # change to your port
    'POSTGRES_USERNAME': 'postgres',  # change to your username
    'POSTGRES_PASSWORD': '6830Project',  # change to your password
    'POSTGRES_DBNAME': 'postgres'
}  # change to your db name
# create connection and cursor
print("test2")
conn = ps.connect(host=credentials['POSTGRES_ADDRESS'],
                  database=credentials['POSTGRES_DBNAME'],
                  user=credentials['POSTGRES_USERNAME'],
                  password=credentials['POSTGRES_PASSWORD'],
                  port=credentials['POSTGRES_PORT'])
cur = conn.cursor()
cur.execute("""SELECT table_name FROM information_schema.tables
       WHERE table_schema = 'public'""")

print(cur.fetchall())
print("test3")

tables = [
    "supplier", 'part', 'partsupp', 'customer', 'nation', 'lineitem', 'region',
    'orders'
]
for table in tables:
    query = "SELECT count(*) FROM {};".format(table)
    print(query)
    cur.execute(query)
    print(cur.fetchall())
