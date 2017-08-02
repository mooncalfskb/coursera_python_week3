# cd /Users/mooncalf/Dropbox/skb/coursera/PythonFundamentals/Week3Assignment
# python3 week3.py
# sudo pip3 install pandas
# sudo pip3 install xlrd
# use python3 in terminal

##### set up df
import pandas as pd
import numpy as np
import matplotlib as plt


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


def answer_two():
	#how many countries in ScimEn?
	idx1 = pd.Index(ScimEn['Country'])
	#how many countries in energy?
	idx2 = pd.Index(energy['Country'])
	#difference
	diff1 = idx1.difference(idx2)
	list1 = list(diff1)
	#print(diff1.size)
	#how many countries in the merge of ScimEn and energy?
	indx3 = pd.Index(df_temp['Country'])
	#how many countries in gdp?
	indx4 = pd.Index(gdp['Country'])
	#difference
	diff2 = indx3.difference(indx4)
	list2 = list(diff2)
	return list1 + list2
	#print(diff2)
	#print(diff2.size)
	#return diff1 + diff2
	
print(answer_two())

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
    max_renew = Top15['% Renewable'].max()
    max_row = Top15.loc[Top15['% Renewable'] == max_renew]
    country = max_row.index[0]
    #print(country)
    #not a string!
    renewable_rate = max_row['% Renewable'][0]
    #print(renewable_rate)
    my_list = [country, renewable_rate]
    return tuple(my_list)
    
#print(answer_six())    
    
### Question 7 (6.6%)
#Create a new column that is the ratio of Self-Citations to Total Citations. What is the maximum value for this new column, and what country has the highest ratio?
#This function should return a tuple with the name of the country and the ratio.    

def answer_seven():
    Top15 = answer_one()
    Citations = Top15
    Citations['citation_ratio'] = Citations['Self-citations'] / Citations['Citations']
    max_renew = Top15['citation_ratio'].max()
    #print(Top15)
    #print(max_renew)
    max_row = Citations.loc[Top15['citation_ratio'] == max_renew]
    country = max_row.index[0]
    #print(country)
    renewable_rate = max_row['citation_ratio'][0]
    #print(renewable_rate)
    my_list = [country, renewable_rate]
    return tuple(my_list)

#print(answer_seven())    
  
### Question 8 (6.6%)
#Create a column that estimates the population using Energy Supply and Energy Supply per capita. What is the third most populous country according to this estimate?
#This function should return a single string value.

def answer_eight():
    Top15 = answer_one()
    Pop = Top15
    Pop['est_pop'] = Pop['Energy Supply'] / Pop['Energy Supply per Capita']
    Pop.sort_values(by=['est_pop'], ascending=[False], inplace=True)
    return Pop.index[2]

#print(answer_eight())   


#### Question 9 (6.6%)
#Create a column that estimates the number of citable documents per person. What is the correlation between the number of citable documents per capita and the energy supply per capita? Use the .corr() method, (Pearson's correlation).
#This function should return a single number.
#(Optional: Use the built-in function plot9() to visualize the relationship between Energy Supply per Capita vs. Citable docs per Capita)

def answer_nine():
    Top15 = answer_one()
    Top15['PopEst'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']
    Top15['Citable docs per Capita'] = Top15['Citable documents'] / Top15['PopEst']
    correlation = Top15['Citable docs per Capita'].corr(Top15['Energy Supply per Capita'])
    return correlation

#print(answer_nine())   
	
	
def plot9():
    import matplotlib as plt
    #%matplotlib inline
    
    Top15 = answer_one()
    Top15['PopEst'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']
    Top15['Citable docs per Capita'] = Top15['Citable documents'] / Top15['PopEst']
    Top15.plot(x='Citable docs per Capita', y='Energy Supply per Capita', kind='scatter', xlim=[0, 0.0006])

#print(plot9())


### Question 10 (6.6%)
#Create a new column with a 1 if the country's % Renewable value is at or above the median for all countries in the top 15, and a 0 if the country's % Renewable value is below the median.
#This function should return a series named HighRenew whose index is the country name sorted in ascending order of rank.

#my correct answer in python 
# def answer_ten():
#     Top15 = answer_one()
#     renew = Top15['% Renewable']
#     mean_renew = renew.mean()
#     Renewable = Top15
#     Renewable.sort_index(inplace=True)
#     #rate = lambda T: 200*exp(-T) if T>200 else 400*exp(-T)
#     Renewable['HighRenew'] = Renewable['% Renewable'].apply(lambda x: 1 if x >= mean_renew else 0)
#     Renewable.sort_values(by=['HighRenew'], ascending=[True], inplace=True)
#     Renewable = Renewable.drop(['Rank', 'Documents', 'Citable documents', 'Citations', 'Self-citations', 'Citations per document', 'H index', 'Energy Supply', 'Energy Supply per Capita', '% Renewable', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015'], axis=1)
#     HighRenew = Renewable.ix[:,0]
#     return HighRenew

#print(answer_ten())   

#submitted for answer because weirdness on site
def answer_ten():
    Top15 = answer_one()
    renew = Top15['% Renewable']
    mean_renew = renew.median()
    Renewable = Top15
    Renewable.sort_index(inplace=True)
    #rate = lambda T: 200*exp(-T) if T>200 else 400*exp(-T)
    Renewable['HighRenew'] = Renewable['% Renewable'].apply(lambda x: 1 if x >= mean_renew else 0)
    Renewable.sort_values(by=['HighRenew'], ascending=[True], inplace=True)
    Renewable = Renewable.drop(['Rank', 'Documents', 'Citable documents', 'Citations', 'Self-citations', 'Citations per document', 'H index', 'Energy Supply', 'Energy Supply per Capita', '% Renewable', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015'], axis=1)
    HighRenew = Renewable.ix[:,5]
    return HighRenew


### Question 11 (6.6%)
#Use the following dictionary to group the Countries by Continent, then create a dateframe that displays the sample size (the number of countries in each continent bin), and the sum, mean, and std deviation for the estimated population of each country.

ContinentDict  = {'China':'Asia', 'United States':'North America', 'Japan':'Asia', 'United Kingdom':'Europe', 'Russian Federation':'Europe', 'Canada':'North America', 'Germany':'Europe', 'India':'Asia', 'France':'Europe', 'South Korea':'Asia', 'Italy':'Europe', 'Spain':'Europe', 'Iran':'Asia','Australia':'Australia', 'Brazil':'South America'}

#This function should return a DataFrame with index named Continent ['Asia', 'Australia', 'Europe', 'North America', 'South America'] and columns ['size', 'sum', 'mean', 'std']

def answer_eleven():
    Top15 = answer_one()
    Conts = Top15
    #set up population estimate
    Conts['PopEst'] = Conts['Energy Supply'] / Conts['Energy Supply per Capita']
    length = len(Conts)
    #print(length)
    #set continent = 0 
    Conts['Continent'] = 0
    #print(Conts['Continent'])

    #tried to do this with lambda but couldn't figure it out.
    pd.options.mode.chained_assignment = None
    for x in range(0, length):
        Conts['Continent'].iloc[x] = ContinentDict[Conts.index[x]]
    
    cont_size = Conts['Continent'].value_counts()
    
    #start your new dataframe
    df = pd.DataFrame({'Continent':cont_size.index, 'size':cont_size.values, 'sum':0, 'mean':0, 'std':0})
    
    length_df = len(df)
    for x in range(0, length_df):
    	this_cont = Conts[Conts['Continent'] == df['Continent'].iloc[x]]
    	df['sum'].iloc[x] = this_cont['PopEst'].sum()
    	df['mean'].iloc[x] = this_cont['PopEst'].mean()
    	df['std'].iloc[x] = this_cont['PopEst'].std()
    df.set_index('Continent', inplace=True)

    return df

#print(answer_eleven())   


### Question 12 (6.6%)
#Cut % Renewable into 5 bins. Group Top15 by the Continent, as well as these new % Renewable bins. How many countries are in each of these groups?
#This function should return a Series with a MultiIndex of Continent, then the bins for % Renewable. Do not include groups with no countries.

def answer_twelve():
    Top15 = answer_one()
    Conts = Top15
    #set up population estimate
    Conts['PopEst'] = Conts['Energy Supply'] / Conts['Energy Supply per Capita']
    length = len(Conts)
    #print(length)
    #set continent = 0 
    Conts['Continent'] = 0
    #print(Conts['Continent'])

    #tried to do this with lambda but couldn't figure it out.
    pd.options.mode.chained_assignment = None
    for x in range(0, length):
        Conts['Continent'].iloc[x] = ContinentDict[Conts.index[x]]
    
    renew = Top15['% Renewable']
    renew_bins = pd.cut(renew, 5)
    group_names = ['F', 'D', 'C', 'B', 'A']
    Conts['bins'] = pd.cut(Conts['% Renewable'], 5)
    Conts['categories'] = pd.cut(Conts['% Renewable'], 5, labels=group_names)
    
    #df1.groupby( [ "Name", "City"] ).count()
    grouped = Conts.groupby(['Continent', 'bins']).count()
    grouped = grouped.drop(['Rank', 'Documents', 'Citable documents', 'Citations', 'Self-citations', 'Citations per document', 'H index', 'Energy Supply', 'Energy Supply per Capita', '% Renewable', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', 'PopEst'], axis=1)
    grouped.dropna(axis=0,inplace=True)
    grouped.rename(columns = {'categories':'countries'}, inplace = True)
    group_series = grouped.ix[:,0]
    return group_series

#print(answer_twelve())

### Question 13
#Convert the Population Estimate series to a string with thousands separator (using commas). Do not round the results.
#e.g. 317615384.61538464 -> 317,615,384.61538464
#This function should return a Series PopEst whose index is the country name and whose values are the population estimate string.

# def answer_thirteen():
#     Top15 = answer_one()
#     Pops = Top15
#     #set up population estimate
#     Pops['PopEst'] = Pops['Energy Supply'] / Pops['Energy Supply per Capita']
#     
#     #print ("{:,.10f}".format(Pops['PopEst'].iloc[0]))
#     Pops['PopEstThou'] = Pops['PopEst'].apply(lambda x: "{:,.8f}".format(x))
#     #Pops = Pops.drop(['Rank', 'Documents', 'Citable documents', 'Citations', 'Self-citations', 'Citations per document', 'H index', 'Energy Supply', 'Energy Supply per Capita', '% Renewable', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015'])
#     #print(Pops)
#     #note this was 25 on the thing because of crazy code interacting.
#     PopEst = Pops.ix[:,21]
#     return PopEst
# print(answer_thirteen())


#submitted for grade because weirdness on site
def answer_thirteen():
    Top15 = answer_one()
    Pops = Top15
    #set up population estimate
    Pops['PopEst'] = Pops['Energy Supply'] / Pops['Energy Supply per Capita']
    
    #print ("{:,.10f}".format(Pops['PopEst'].iloc[0]))
    #without forced float
    Pops['PopEstThou'] = Pops['PopEst'].apply(lambda x: "{:,}".format(x))
    #Pops = Pops.drop(['Rank', 'Documents', 'Citable documents', 'Citations', 'Self-citations', 'Citations per document', 'H index', 'Energy Supply', 'Energy Supply per Capita', '% Renewable', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015'])
    #print(Pops)
    PopEst = Pops.ix[:,29]
    return PopEst
