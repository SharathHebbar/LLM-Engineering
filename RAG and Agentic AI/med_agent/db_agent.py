import duckdb
import regex as re

import pandas as pd


def create_and_insert_db(table_name, df, columns_sql):
    con = duckdb.connect("agentic_db.duckdb")
    # table_name = table_name.split(".")[0]
    if table_name in list(con.execute("Show tables").df()['name']):
        if list(con.execute(f"Select count(*) from '{table_name}'").df()['count_star()'])[0] > 0:
            return con
        else:
            insert_sql = f"INSERT INTO {table_name} SELECT * from df"
            con.execute(insert_sql)
    else:
        create_sql = f"CREATE TABLE IF NOT EXISTS '{table_name}' ({columns_sql});"
        print(create_sql)
        con.execute(create_sql)
        insert_sql = f"INSERT INTO '{table_name}' SELECT * from df"
        con.execute(insert_sql)
    return con
