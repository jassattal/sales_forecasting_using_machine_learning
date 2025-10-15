
# Running SQL statements from Python
import pyodbc
import pandas.io.sql as psql
import stacks.config as config
import warnings
# import pandas as pd
warnings.filterwarnings("ignore")


def load(stream):
    print("Data Load started for {stream_name}".format(stream_name=stream))
    #Changes for liquor
    #if stream=="liquor":
    #sqlDF=pd.read_csv("inputs/liquor_groupedby_markup.csv")
    cnxn = pyodbc.connect(config.curl_dictionary[stream])
    sqlDF = psql.read_sql(config.query_dictionary[stream], cnxn)
    cnxn.close()
    print("Data Loaded")

    return sqlDF
