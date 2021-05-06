import psycopg2 as ps

credentials = {
    'POSTGRES_ADDRESS': 'db-test.ch9w9rkl1agb.us-east-2.rds.amazonaws.com',
    'POSTGRES_PORT': '5432', 'POSTGRES_USERNAME': 'postgres',
    'POSTGRES_PASSWORD': '6830Project', 'POSTGRES_DBNAME': 'postgres'
}


def connect():
    conn = ps.connect(host=credentials['POSTGRES_ADDRESS'],
                      database=credentials['POSTGRES_DBNAME'],
                      user=credentials['POSTGRES_USERNAME'],
                      password=credentials['POSTGRES_PASSWORD'],
                      port=credentials['POSTGRES_PORT'])
    return conn.cursor()


def run_query(query):
    cursor = connect()
    cursor.execute(query)
    return cursor.fetchall()
