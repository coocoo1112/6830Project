from RDS_query import run_query

joins = [
    "part.partkey=partsupp.partkey", "supplier.suppkey=partsupp.suppkey",
    "customer.nationkey=nation.nationkey",
    "customer.nationkey=nation.nationkey", "partsupp.partkey=lineitem.partkey",
    "partsupp.suppkey=lineitem.suppkey", "customer.custkey=orders.custkey",
    "orders.orderkey=lineitem.orderkey", "region.regionkey=nation.nationkey"
]
tables_query = "SELECT tablename FROM pg_catalog.pg_tables \
    WHERE schemaname != 'pg_catalog' AND \
    schemaname != 'information_schema';"

columns_query = "SELECT column_name FROM information_schema.columns\
    WHERE table_name = '{}';"

stats_get = "select histogram_bounds, most_common_vals, tablename, attname FROM pg_stats\
    where tablename in {}"

tables = [i[0] for i in run_query(tables_query)]
table_columns = {}
stats_dict = {}

print(tables)
for table in tables:
    table_columns[table] = [i[0] for i in run_query(columns_query.format(table))]
    stats_dict[table] = {}


stats = run_query(stats_get.format(tuple(tables)))
for stat in stats:
    table = stat[2]
    column = stat[3]
    stats_dict[table][column] = (stat[0], stat[1])



