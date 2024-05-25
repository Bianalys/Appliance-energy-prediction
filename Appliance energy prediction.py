#!/usr/bin/env python
# coding: utf-8

# In[2]:


#importing all necessary library for this project
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib_inline
import datetime


# In[3]:


#Importing the data from database
df=pd.read_csv('data_application_energy.csv')


# In[4]:


df.head(1)


# In[5]:


df.tail(1)


# In[6]:


df.shape


# In[7]:


df.isnull().sum()


# In[8]:


df.info()


# In[9]:


df.describe()


# In[10]:


df1=df.rename(columns={'date':'Date', 'Appliances':'Appliances_energy', 'lights':'Lights_watt', 'T1':'Kitchen_T', 'RH_1':'Kitchen_H', 'T2':'Livingroom_T', 'RH_2':'Livingroom_H', 'T3':'Laundaryroom_T', 'RH_3':'Laundaryroom_H', 'T4':'Officeroom_T', 'RH_4':'Officeroom_H', 'T5':'Bathroom_T', 'RH_5':'Bathroom_H', 'T6':'Outside_buil(Northside)_T', 'RH_6':'Outside_buil(Northside)_H', 'T7':'Ironingroom_T', 'RH_7':'Ironingroom_H', 'T8':'Teenagerroom_T', 'RH_8':'Teenagerroom_H', 'T9':'Parentsroom_T', 'RH_9':'Parentsroom_H', 'T_out':'Temp_outside', 'Press_mm_hg':'Pressure_load', 'RH_out':'Outside_H', 'Windspeed':'Windspeed_m/s', 'Visibility':'visiblity_dis', 'Tdewpoint':'Dew_temp', 'rv1':'Reserve_energy1', 'rv2':'Reserve_energy2'})


# In[11]:


plt.figure(figsize=(10,10))
sns.displot(data=df1.isna().melt(value_name='missing'),y='variable',hue='missing',multiple='fill',aspect=1.50)
plt.title('Null values in Energy data')
plt.show()


# In[12]:


def get_uniquevalues(df1):
    unique_values=df1.apply
    return unique_values

unq_values=get_uniquevalues(df1)

for i in df1.columns.tolist():
    print('No of unique values in', i, 'is', df1[i].nunique())


# In[13]:


df_Energy=df1


# In[14]:


df_Energy['Date']=pd.to_datetime(df_Energy['Date'])


# In[15]:


df_Energy.info()


# In[16]:


df_Energy['month'] = df_Energy['Date'].dt.month
df_Energy['weekday'] = df_Energy['Date'].dt.weekday
df_Energy['hour'] = df_Energy['Date'].dt.hour
df_Energy['week'] = df_Energy['Date'].dt.isocalendar().week
df_Energy['day_of_week'] = df_Energy['Date'].dt.day_name()
df_Energy['week_of_month'] = (df_Energy['Date'].dt.day-1) // 7 + 1


df_Energy.drop('Date',axis=1,inplace=True)

df_Energy.head(1)


# In[17]:


#Define a dictionary mapping day names to numerical representations
day_mapping = {
    'Monday': 1,
    'Tuesday': 2,
    'Wednesday': 3,
    'Thursday': 4,
    'Friday': 5,
    'Saturday': 6,
    'Sunday': 7
}

# Assuming 'day_of_week' is the name of the column containing day of week information
df_Energy['day_of_week'] = df_Energy['day_of_week'].map(day_mapping)

# Convert the column to float type
df_Energy['day_of_week'] = df_Energy['day_of_week'].astype(float)


# In[18]:


corr_matrix = df_Energy.corr()

# create the heatmap with a larger size
plt.figure(figsize=(25, 20))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True)

# show the plot
plt.show()


# In[ ]:





# In[19]:


for col in df_Energy.columns:  
  mean = df_Energy[col].mean()
  std = df_Energy[col].std()
  threshold = 3
  outliers = df_Energy[(df_Energy[col] - mean).abs() > threshold * std]
  df_Energy = df_Energy.drop(outliers.index)


# In[20]:


df_Energy.info()


# In[21]:


df_Energy.shape


# In[22]:


df_Energy['week'] = df_Energy['week'].astype(np.int32)


# In[ ]:





# In[37]:


#  Plotting the distribution of energy useage by appliances
plt.figure(figsize=(10,6))
sns.histplot(data=df_Energy, x='Appliances_energy',bins =20 ,color = 'blue', alpha =0.7)
plt.title('Distribution of Energy Usage by Appliances')
plt.xlabel('Energy usage (Wh)')
plt.ylabel('Frequrency')
plt.grid()
plt.show()


# In[38]:


#  Plotting the distribution of Light
plt.figure(figsize=(10,6),facecolor='white')
sns.histplot(data=df_Energy,x= 'Lights_watt',bins=20, color='green')
plt.title('Distribution of lights')
plt.xlabel('light usage (Wh)')
plt.ylabel('Frequency')
plt.grid()
plt.show()


# In[23]:


mean_by_hour = df_Energy.groupby('hour')['Appliances_energy'].mean()
mean_by_hour.plot(kind='line', figsize=(10, 6))
plt.title('Mean Energy Consumption by Hour')
plt.xlabel('Hour')
plt.ylabel('Mean Energy Consumption')
plt.grid(True)
plt.show()
df.head()


# In[24]:


df_Energy['type_of_day'] = np.where(df_Energy['day_of_week'] < 5, 'Weekday', 'Weekend')


# In[25]:


#  Plotting the distribution Temperatures of indoor
selected_columns = ['Kitchen_T','Livingroom_T','Laundaryroom_T','Officeroom_T','Bathroom_T','Outside_buil(Northside)_T','Ironingroom_T','Teenagerroom_T','Parentsroom_T']
plt.figure(figsize=(8, 3))
df_Energy[selected_columns].hist(bins=15, color='Blue',edgecolor='black', grid=True, layout=(3, 3), figsize=(12, 8))
plt.suptitle('Histograms of Indoor Temperatures', x=0.5, y=1.02, fontsize=16)
plt.tight_layout()
plt.show()


# In[26]:


#  Plotting the distribution Temperatures of indoor
selected_columns = ['Kitchen_H','Livingroom_H','Laundaryroom_H','Outside_buil(Northside)_H','Ironingroom_H','Teenagerroom_H','Parentsroom_H']
plt.figure(figsize=(8, 3))
df_Energy[selected_columns].hist(bins=15, color='magenta',edgecolor='black', grid=True, layout=(3, 3), figsize=(12, 8))
plt.suptitle('Histograms of Indoor Humidity', x=0.5, y=1.02, fontsize=16)
plt.tight_layout()
plt.show()


# In[27]:


#  Plotting the distribution Temperatures of indoor
selected_columns = ['Temp_outside', 'Pressure_load','Outside_H', 'Windspeed_m/s', 'visiblity_dis', 'Dew_temp']
plt.figure(figsize=(8, 3))
df_Energy[selected_columns].hist(bins=15, color='Green',edgecolor='black', grid=True, layout=(3, 3), figsize=(12, 8))
plt.suptitle('Histograms of Weathercon_columns', x=0.5, y=1.02, fontsize=16)
plt.tight_layout()
plt.show()


# In[28]:


#  Plotting the distribution Temperatures of indoor
selected_columns = ['Reserve_energy1', 'Reserve_energy2']
plt.figure(figsize=(8, 3))
df_Energy[selected_columns].hist(bins=15, color='lightcoral',edgecolor='black', grid=True, layout=(3, 3), figsize=(12, 8))
plt.suptitle('Histograms of Random_vars', x=0.5, y=1.02, fontsize=16)
plt.tight_layout()
plt.show()


# In[29]:


Energy_contr = df_Energy.groupby('type_of_day')['Appliances_energy'].sum()

# Plotting
plt.figure(figsize=(8, 5), facecolor='white')
plt.bar(Energy_contr.index, Energy_contr.values, color=['blue', 'green'])
plt.title('Energy Consumption of Appliances: Weekdays vs Weekends')
plt.xlabel('Type of Day')
plt.ylabel('Total Energy Consumption (Wh)')
plt.show()
df.head()


# In[46]:


mean_by_weekday = df_Energy.groupby('day_of_week')['Appliances_energy'].mean()

# Your data and plotting code
plt.figure(figsize=(10,6))
colors = ['Red', 'green', 'magenta', 'purple', 'orange', 'lavender', 'cyan']
plt.bar(mean_by_weekday.index, mean_by_weekday.values,color=colors)
plt.xlabel('Day of the Week')
plt.ylabel('Mean Energy Consumption')
plt.title('Mean Energy Consumption by Day of the Week')
plt.show()


# In[47]:


df_Energy.head(1)


# In[48]:


# Grouping by hour to find average energy usage per hour
df_hourly = df_Energy.groupby('hour')[['Appliances_energy', 'Lights_watt']].mean()


# Plotting the average energy usage by hour
df_hourly.plot(kind='bar', figsize=(15, 7), title='Average Energy Usage by Hour of the Day')
plt.xlabel('Hour of the Day')
plt.ylabel('Average Energy Usage (Wh)')
plt.grid(True)
plt.show()


# In[ ]:





# In[30]:


plt.figure(figsize=(8,4))
sns.lineplot(df_Energy,x='month',y='Appliances_energy',hue='weekday')
plt.grid()
plt.show()


# In[31]:


plt.figure(figsize=(8,4))
sns.lineplot(df_Energy,x='month',y='Appliances_energy')
plt.grid()
plt.show()


# In[32]:


plt.figure(figsize=(8,4))
sns.lineplot(df_Energy,x='week',y='Appliances_energy')
plt.grid()
plt.show()


# In[51]:


# Selecting the temperature features and 'Appliances' column
temperature_columns = ['Kitchen_T','Livingroom_T','Laundaryroom_T','Officeroom_T','Bathroom_T','Outside_buil(Northside)_T','Ironingroom_T','Teenagerroom_T','Parentsroom_T','Temp_outside']
df_temp = df_Energy[temperature_columns + ['Appliances_energy']]

# Visualizing the relationship using scatterplot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Temp_outside', y='Appliances_energy', data=df_temp)
plt.title('Temperature vs. Appliance Energy Consumption')
plt.xlabel('Outside Temperature (Temp_outside)')
plt.ylabel('Appliance Energy Consumption')
plt.grid(True)
plt.show()

# Calculating the correlation coefficient
correlation = df_temp['Temp_outside'].corr(df_temp['Appliances_energy'])
print("Correlation coefficient between T_out and Appliances:", correlation)


# In[53]:


df_Energy.columns


# In[57]:


# Calculate temperature and humidity differences between indoors and outdoors
temp_columns = ['Kitchen_T', 'Livingroom_T', 'Laundaryroom_T', 'Officeroom_T', 'Bathroom_T', 'Outside_buil(Northside)_T', 'Ironingroom_T', 'Teenagerroom_T', 'Parentsroom_T']
for col in temp_columns:
    df_Energy[f'T_diff_{col}'] = df_Energy[col] - df_Energy['Temp_outside']

humidity_columns = ['Kitchen_H', 'Livingroom_H', 'Laundaryroom_H', 'Officeroom_H', 'Bathroom_H', 'Outside_buil(Northside)_H', 'Ironingroom_H', 'Teenagerroom_H', 'Parentsroom_H']
for col in humidity_columns:
    df_Energy[f'RH_diff_{col}'] = df_Energy[col] - df_Energy['Outside_H']


# Plotting temperature differences
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_Energy.filter(like='T_diff'), orient='h')
plt.title('Temperature Difference between Indoors and Outdoors')
plt.xlabel('Temperature Difference (°C)')
plt.ylabel('Room')
plt.grid(True)
plt.show()

# Plotting humidity differences
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_Energy.filter(like='RH_diff'), orient='h')
plt.title('Humidity Difference between Indoors and Outdoors')
plt.xlabel('Humidity Difference (%)')
plt.ylabel('Room')
plt.grid(True)
plt.show()


# In[50]:


# Assuming you have a DataFrame named df containing 'hour', 'weekday', 'month', and 'Appliances' columns

# Box plot for appliance energy consumption variation throughout the day
plt.figure(figsize=(10, 6))
sns.boxplot(x='hour', y='Appliances_energy', data=df_Energy)
plt.title('Appliance Energy Consumption Variation Throughout the Day')
plt.xlabel('Hour of the Day')
plt.ylabel('Appliance Energy Consumption')
plt.grid(True)
plt.show()
# Box plot for appliance energy consumption variation throughout the week
plt.figure(figsize=(10, 6))
sns.boxplot(x='week', y='Appliances_energy', data=df_Energy)
plt.title('Appliance Energy Consumption Variation Throughout the Week')
plt.xlabel('Weekday')
plt.ylabel('Appliance Energy Consumption')
plt.grid(True)
plt.show()

# Box plot for appliance energy consumption variation throughout the month
plt.figure(figsize=(10, 6))
sns.boxplot(x='month', y='Appliances_energy', data=df_Energy)
plt.title('Appliance Energy Consumption Variation Throughout the Month')
plt.xlabel('Month')
plt.ylabel('Appliance Energy Consumption')
plt.grid(True)


# In[63]:


# focussed displots for RH_6 , RH_out , 'Windspeed_m/s', 'visiblity_dis' due to irregular distribution
f, ax = plt.subplots(2,2,figsize=(12,8))
vis1 = sns.distplot(df_Energy["Officeroom_H"],bins=10, ax= ax[0][0])
vis2 = sns.distplot(df_Energy["Outside_H"],bins=10, ax=ax[0][1])
vis3 = sns.distplot(df_Energy["visiblity_dis"],bins=10, ax=ax[1][0])
vis4 = sns.distplot(df_Energy["Windspeed_m/s"],bins=10, ax=ax[1][1])
plt.show()


# In[59]:


df_Energy.columns


# In[33]:


cols = ['Appliances_energy', 'Lights_watt','Temp_outside', 'Pressure_load','Outside_H', 'Windspeed_m/s', 'visiblity_dis', 'Dew_temp']

# Pairplot to visualize the relationships
sns.pairplot(df_Energy[cols], kind='scatter', diag_kind='kde', plot_kws={'alpha':0.6, 's':30, 'edgecolor':'k'}, diag_kws={'shade':True})
plt.suptitle('Relationships between External Weather Conditions and Energy Usage', y=1.02)
plt.show()


# In[ ]:




