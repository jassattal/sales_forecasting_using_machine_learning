import pandas as pd
import requests
import stats_can
import warnings
warnings.filterwarnings("ignore")
# Oil_Prices
response = requests.get(
    "https://api.economicdata.alberta.ca/api/data?code=1da37895-ed56-405e-81de-26231ffc6472")
Oil_Prices = pd.DataFrame(response.json())
# CPI
cpi = stats_can.table_to_df("18-10-0004-01")
cpi = cpi[(cpi["GEO"] == "Alberta") & (cpi["Products and product groups"]
                                       == "All-items") & (cpi["UOM"] == "2002=100")][["REF_DATE", "VALUE"]]
cpi["Value-12"] = cpi["VALUE"].shift(12)
cpi["Percentage"] = (cpi["VALUE"]-cpi["Value-12"])/cpi["Value-12"]
cpi.drop("Value-12", axis=1, inplace=True)
cpi.dropna(axis=0, inplace=True)

# Unemployment Rate
unemployment = stats_can.table_to_df("14-10-0017-01")
unemployment = unemployment[(unemployment["GEO"] == "Alberta") & (unemployment["Gender"] == "Total - Gender") & (
    unemployment["Labour force characteristics"] == "Unemployment rate") & (unemployment["Age group"] == "15 years and over")][["REF_DATE", "VALUE"]]
# Migration_interprov
Migration_interprov = stats_can.table_to_df("17-10-0020-01")
Migration_interprov = Migration_interprov[(Migration_interprov["GEO"] == "Alberta")].pivot_table(
    values="VALUE", index="REF_DATE", columns="Interprovincial migration")
Migration_interprov["Net_Interprovincial"] = Migration_interprov["In-migrants"] - \
    Migration_interprov["Out-migrants"]
Migration_interprov = Migration_interprov[["Net_Interprovincial"]].reset_index()
# Migration_international
Migration_inter = stats_can.table_to_df("17-10-0040-01")
Migration_inter = Migration_inter[(Migration_inter["GEO"] == "Alberta")].pivot_table(
    values="VALUE", index="REF_DATE", columns="Components of population growth")
Migration_inter = Migration_inter[["Immigrants"]].reset_index()

Migration=Migration_inter.merge(Migration_interprov)
Migration["Net_Migration"]=Migration["Immigrants"]+Migration["Net_Interprovincial"]
Migration=Migration[["REF_DATE","Net_Migration"]]


# GDP
gdp = stats_can.table_to_df("36-10-0222-01")
gdp = gdp[(gdp["GEO"] == "Alberta") & (gdp["Prices"] == "Chained (2017) dollars") & (
    gdp["Estimates"] == "Gross domestic product at market prices")][["REF_DATE", "VALUE"]]
# Population
population = stats_can.table_to_df("17-10-0009-01")
population=population[population["GEO"]=="Alberta"][["REF_DATE","VALUE"]]
population['POP LAG-12']=population["VALUE"].shift(4)
population.dropna(inplace=True)
population["%Change_population"]=((-population['POP LAG-12']+population['VALUE'])/population['VALUE'])*100
population=population[["REF_DATE","VALUE","%Change_population"]]
population["REF_DATE"]=pd.to_datetime(population["REF_DATE"])
population.columns=["Month_Adjusted","Population_number","%Change_population"]





# Eco Dashboard Data
unemployment["Date"] = pd.to_datetime(unemployment["REF_DATE"])
unemployment = unemployment[["Date", "VALUE"]]
unemployment.columns = ["Month_Adjusted", "Unemployment_Rate"]
# Migration_Eco Dash
Migration["Date"] = pd.to_datetime(Migration["REF_DATE"])
Migration = Migration[["Date", "Net_Migration"]]
Migration.columns = ["Month_Adjusted", "Net_Migration"]
# Nat_Gas_Eco Dash
# Nat_Gas_Price["Date"]=pd.to_datetime(Nat_Gas_Price["D"])
# Nat_Gas_Price=Nat_Gas_Price[["Date","Value"]]
# Nat_Gas_Price.columns=["Month_Adjusted","Nat_Gas_Prices"]
# Oil_Prices Eco dash
Oil_pivot = Oil_Prices.pivot_table(
    values="Value", index="Date", columns='Type ')
Oil_pivot_2 = Oil_pivot.reset_index()
Oil_pivot_2["Date"] = pd.to_datetime(Oil_pivot_2["Date"])
Oil_pivot_2.columns = ["Month_Adjusted", "WCS","WTI"]
# Retail_sales_Eco Dash
# Retail_Sales["Date"]=pd.to_datetime(Retail_Sales["Date"])
# Retail_Sales=Retail_Sales[["Date","Value"]]
# Retail_Sales.columns=["Month_Adjusted","Retail_Sale"]
# CPI Eco Dashboard Data
cpi["Date"] = pd.to_datetime(cpi["REF_DATE"])
cpi = cpi[["Date", "Percentage","VALUE"]]
cpi.columns = ["Month_Adjusted", "CPI_Percentage", "CPI_2002_Base"]
# gdp Eco Dashboard
gdp["Date"] = pd.to_datetime(gdp["REF_DATE"])
gdp = gdp[["Date", "VALUE"]]
gdp.columns = ["Month_Adjusted", "GDP_2002_Base"]
gdp = pd.concat([gdp, pd.DataFrame([[pd.to_datetime("2023-01-01"),
                347000]], columns=["Month_Adjusted", "GDP_2002_Base"])])
