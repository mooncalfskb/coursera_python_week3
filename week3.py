# cd /Users/mooncalf/Dropbox/skb/coursera/PythonFundamentals/Week3Assignment
# python3 week3.py
# sudo pip3 install pandas
# sudo pip3 install xlrd
# use python3 in terminal

##### set up df
import pandas as pd
import numpy as np

######### ENERGY INDICATORS
xl = pd.ExcelFile("Energy Indicators.xls", skiprows=18)
xl.sheet_names
#Energy
energy = xl.parse("Energy")
energy.drop(energy.columns[[0,1]], axis=1,inplace=True)
#drop header
energy.drop(energy.index[0:16], axis=0,inplace=True)
#drop footer
energy.drop(energy.index[227:], axis=0,inplace=True)
#rename columns
energy.rename(columns = {'Environmental Indicators: Energy':'Country', 'Unnamed: 3':'Energy Supply', 'Unnamed: 4':'Energy Supply per Capita', 'Unnamed: 5':'% Renewable'}, inplace = True)
#convert NaNs
energy.replace({'...': np.nan}, inplace=True)
#convert to gigajoules
energy['Energy Supply'] = energy['Energy Supply'].apply(lambda x: x*1000000)
#kill crazy footnotes
energy['Country'] = energy['Country'].str.replace(r'[0-9]', '')

#change names
energy['Country'].replace({'Republic of Korea': 'South Korea', 'United States of America': 'United States', 'United Kingdom of Great Britain and Northern Ireland': 'United Kingdom', 'China, Hong Kong Special Administrative Region': 'Hong Kong'}, inplace=True)

#kill parens
energy['Country'] = energy['Country'].str.replace(r'\(.*?\)', '')

#trim Country names in case there's some bullshit, like in iran
energy['Country'] = energy['Country'].apply(lambda x: x.lstrip().rstrip())


#debug
#iran = energy.loc[energy['Country'] == 'Iran']
#print(iran)

#print(energy.head(n=10))
#print(energy.shape)
######### ENERGY INDICATORS


######### GDP
gdp = pd.read_csv('world_bank.csv', skiprows=4)
#drop empty column at end
#gdp.drop(gdp.columns[[61]], axis=1,inplace=True)
#rename country name to country
gdp.rename(columns = {'Country Name':'Country'}, inplace = True)
#fix some country names
gdp['Country'].replace({'Korea, Rep.': 'South Korea', 'Iran, Islamic Rep.': 'Iran', 'Hong Kong SAR, China': 'Hong Kong'}, inplace=True)
#trim Country names in case there's some bullshit, like in iran
gdp['Country'] = gdp['Country'].apply(lambda x: x.lstrip().rstrip())
#debug
#dude = gdp[gdp['Country'].str.contains("Iran")]
#iran = gdp.loc[gdp['Country'] == 'Iran']
#print(iran)

#print(gdp.head(n=10))
#print(gdp.shape)

######### GDP

######### ScimEn
x2 = pd.ExcelFile("scimagojr-3.xlsx")
ScimEn = x2.parse("Sheet1")
#trim Country names in case there's some bullshit, like in iran
ScimEn['Country'] = ScimEn['Country'].apply(lambda x: x.lstrip().rstrip())
#iran = ScimEn.loc[ScimEn['Country'] == 'Iran']
#print(iran)

#print(ScimEn.head(n=10))
#print(ScimEn.shape)
######### ScimEn


######### Big Dataframe

#merge dataframes
df_temp = pd.merge(ScimEn, energy, on='Country', how='inner')
df = pd.merge(df_temp, gdp, on='Country', how='inner')

df.set_index('Country', inplace=True)
#print(df.head(n=15))

#drop extra years
df.drop(df.columns[13:59], axis=1,inplace=True)
#drop country code from gdp
df.drop(df.columns[10:13], axis=1,inplace=True)
#drop 2016
#df.drop(df.columns[[20]], axis=1,inplace=True)

#drop footer
df.drop(df.index[15:], axis=0,inplace=True)

#debug
#print(df.head(n=15))
#print(df.shape)
#list column names:
#print(list(df))

######### Big Dataframe


def answer_one():
	return df
	
#answer_one()	

######### Question 2

#how many countries in ScimEn?
idx1 = pd.Index(ScimEn['Country'])
#how many countries in energy?
idx2 = pd.Index(energy['Country'])
#difference
diff1 = idx1.difference(idx2)
#print(diff1.size)
#how many countries in the merge of ScimEn and energy?
indx3 = pd.Index(df_temp['Country'])
#how many countries in gdp?
indx4 = pd.Index(gdp['Country'])
#difference
diff2 = indx3.difference(indx4)
#print(diff2)
#print(diff2.size)

def answer_two():
	return diff1.size + diff2.size
	
#print(answer_two())

######### Question 3
#What is the average GDP over the last 10 years for each country? (exclude missing values from this calculation.)

#*This function should return a Series named `avgGDP` with 15 countries and their average GDP sorted in descending order.*	

def answer_three():
    Top15 = answer_one()
    Top15['avg'] = Top15[['2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']].mean(axis=1)
    #ditch columns
    avg_df = Top15.drop(['Rank', 'Documents', 'Citable documents', 'Citations', 'Self-citations', 'Citations per document', 'H index', 'Energy Supply', 'Energy Supply per Capita', '% Renewable', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015'], axis=1)
    #sort (note the 13 on the us number, not 12)
    avg_df.sort_values(by=['avg'], ascending=[False], inplace=True)
    #pick first column as series
    avgGDP = avg_df.ix[:,0]
	#print(avgGDP)
	#print(type(avgGDP))
    return avgGDP
    
######### Question 4    
#Question 4 (6.6%)
#By how much had the GDP changed over the 10 year span for the country with the 6th largest average GDP?
#This function should return a single number. 
#United Kingdom        2.487907e+12
   
def answer_four():
    Top15 = answer_one()
    avgGDP = answer_three()
	#took an hour to figure this out although seems very easy
    sixth = avgGDP.index[5]
    #print(sixth)
    country_six = Top15.loc[sixth]
    #print(country_six)
    return country_six['2015'] - country_six['2006']
    

#answer_four()

### Question 5 (6.6%)
#What is the mean `Energy Supply per Capita`?
def answer_five():
    Top15 = answer_one()
    energy_supply = Top15['Energy Supply per Capita']
    mean_energy = energy_supply.mean()
    return mean_energy
    
#print(answer_five())    


### Question 6 (6.6%)
#What country has the maximum % Renewable and what is the percentage?
#This function should return a tuple with the name of the country and the percentage.    

def answer_six():
    Top15 = answer_one()
    energy_supply = Top15['Energy Supply per Capita']
    mean_energy = energy_supply.mean()
    return mean_energy
    
    
    

