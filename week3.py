# cd /Users/mooncalf/Dropbox/skb/coursera/PythonFundamentals/Week3Assignment
# python3 week3.py
# sudo pip3 install pandas
# sudo pip3 install xlrd
# use python3 in terminal

##### set up df
import pandas as pd

xl = pd.ExcelFile("Energy Indicators.xls", skiprows=18)
xl.sheet_names
#Energy
energy = xl.parse("Energy")
energy.drop(energy.columns[[0,1]], axis=1,inplace=True)
#drop header
energy.drop(energy.index[0:16], axis=0,inplace=True)
#drop footer
energy.drop(energy.index[227:], axis=0,inplace=True)
energy.rename(columns = {'Environmental Indicators: Energy':'Country', 'Unnamed: 3':'Energy Supply', 'Unnamed: 4':'Energy Supply per Capita', 'Unnamed: 5':'% Renewable'}, inplace = True)
print(energy)
