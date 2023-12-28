import duckdb

database_uri = "../../../database.db"
conn = duckdb.connect(database=database_uri, read_only=False)
