import duckdb

database_name = "database.db"
database_uri = "../../../database.db"
conn = duckdb.connect(database=database_uri, read_only=False)
