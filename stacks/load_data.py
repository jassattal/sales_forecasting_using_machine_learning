
# Running SQL statements from Python
import stacks.config as config
import warnings
import pandas as pd
warnings.filterwarnings("ignore")


def load(stream):
    file_path = "inputs/Warehouse_and_Retail_Sales.csv"
    df=pd.read_csv(file_path)
    # Load the latest version
    df["stream"]=stream
    print("First 5 records:", df.head())

    return df

    