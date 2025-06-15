from openai import OpenAI
import pandas as pd
import os
import re
import duckdb
from db_agent import create_and_insert_db
from dotenv import load_dotenv

load_dotenv()
ENDPOINT = os.getenv("ENDPOINT")

ASSET_DIR = "./assets"
def remove_special_characters(text):
    pattern = r"[^a-zA-Z0-9\s]"
    text = text.replace(" ", "_")
    cleansed_text = re.sub(pattern, "", text)
    return cleansed_text

model_name = "medgemma-4b-it"

client = OpenAI(
    base_url=ENDPOINT,
    api_key="lm-studio"
)

def cleanse_code(sql_code):
    sql_code_execute = None
    start_idx = 0
    end_idx = 0
    if "```" in sql_code:
        if "sql" in sql_code:
            start_idx = sql_code.find("sql") + len("sql")
        else:
            start_idx = sql_code.find("```") + len("```")
        end_idx = sql_code.rfind("```")
    sql_code_execute = sql_code[start_idx: end_idx]
    return sql_code_execute


def brain_functions(df, table_name, user_input):
    # df = pd.read_csv(f"{dataframe}")
    if "Unnamed: 0" in list(df.columns):
        df.drop("Unnamed: 0", inplace=True, axis=1)

    type_mapping = {
    "object": "TEXT",
    "int16": "SMALLINT",
    "int32": "INT",
    "int64": "BIGINT",
    "int128": "HUGEINT",
    "float64": "FLOAT",
    "bool": "BOOLEAN"
    }

    columns_sql = ", ".join(
        f"{remove_special_characters(col)} {type_mapping[str(dtype)]}" for col, dtype in zip(df.columns, df.dtypes)
    )

    con = create_and_insert_db(table_name, df, columns_sql)

    system_context = f"""
    Given the dataset '{table_name}' with below schema:
    {columns_sql}

    Top 5 rows of the dataset is as follows:
    {df.head()}

    Bottom 5 rows of the dataset is as follows:
    {df.tail()}

    Using the above details generate a sql code for the user query.
    The SQL code will be run on Python interpreter.
    Note: 
    - **If the user input is of greetings then reply back with greetings.**
    - **If the user input is of generic questions0 then reply back with generic answer.**
    - **If the user input is of out of context then reply back with Out of Context.**

    Example:
    user_input: Hello
    output: Hello, Nice to chat with you.
    user_input: List out all the records in the dataset.
    output: ```sql\nselect * from table```
    """

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_context},
            {"role": "user", "content": user_input}
        ], 
    )

    sql_code = response.choices[0].message.content
    # print(sql_code, type(sql_code))
    if "Out of Context" in sql_code:
        return None
    sql_code_execute = cleanse_code(sql_code)

    if (sql_code_execute is None or sql_code_execute == "") and "Out of Context" not in sql_code:
        return sql_code
    elif (sql_code_execute is None or sql_code_execute == ""):
        return None

    try:
        table_op = con.execute(sql_code_execute).df()
    except Exception as e:
        table_op=None
        print(f"Exception occured: {e}")
    finally:
        con.close()
    return table_op