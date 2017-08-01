# cd /Users/mooncalf/Dropbox/skb/coursera/PythonFundamentals/Week3Assignment
# python3 week3.py
# sudo pip3 install pandas
# sudo pip3 install xlrd
# use python3 in terminal

##### set up df
import pandas as pd
import numpy as np

# ######### ENERGY INDICATORS
# xl = pd.ExcelFile("Energy Indicators.xls", skiprows=18)
# xl.sheet_names
# #Energy
# energy = xl.parse("Energy")
# energy.drop(energy.columns[[0,1]], axis=1,inplace=True)
# #drop header
# energy.drop(energy.index[0:16], axis=0,inplace=True)
# #drop footer
# energy.drop(energy.index[227:], axis=0,inplace=True)
# #rename columns
# energy.rename(columns = {'Environmental Indicators: Energy':'Country', 'Unnamed: 3':'Energy Supply', 'Unnamed: 4':'Energy Supply per Capita', 'Unnamed: 5':'% Renewable'}, inplace = True)
# #convert NaNs
# energy.replace({'...': np.nan}, inplace=True)
# #convert to gigajoules
# energy['Energy Supply'] = energy['Energy Supply'].apply(lambda x: x*1000000)
# 
# #kill crazy footnotes
# energy['Country'] = energy['Country'].str.replace(r'[0-9]', '')
# 
# #change names
# energy['Country'].replace({'Republic of Korea': 'South Korea', 'United States of America': 'United States', 'United Kingdom of Great Britain and Northern Ireland': 'United Kingdom', 'China, Hong Kong Special Administrative Region': 'Hong Kong'}, inplace=True)
# 
# #kill parens
# energy['Country'] = energy['Country'].str.replace(r'\(.*?\)', '')
# 
# 
# #debug
# #hongkong = energy.loc[energy['Country'] == 'United States of America']
# #dude = energy[energy['Country'].str.contains("Bolivia")]
# #print(dude)
# 
# #print(energy.head(n=10))
# ######### ENERGY INDICATORS
# 
# 
# ######### GDP
# gdp = pd.read_csv('world_bank.csv', skiprows=4)
# #drop empty column at end
# gdp.drop(gdp.columns[[61]], axis=1,inplace=True)
# #rename country name to country
# gdp.rename(columns = {'Country Name':'Country'}, inplace = True)
# #fix some country names
# gdp['Country'].replace({'Korea, Rep.': 'South Korea', 'Iran, Islamic Rep.': 'Iran', 'Hong Kong SAR, China': 'Hong Kong'}, inplace=True)
# 
# #debug
# #dude = gdp[gdp['Country'].str.contains("Hong Kong")]
# #print(dude)
# 
# #print(gdp.head(n=10))
# ######### GDP

######### ScimEn
x2 = pd.ExcelFile("scimagojr-3.xlsx")

print(x2.sheet_names)
#Energy
#energy = xl.parse("Energy")


