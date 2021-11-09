#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import os 
import collections
from datetime import datetime as dt
from os import walk

import streamlit as st

dir = "Datasets"
random_seed = 42

print('Ready to Use! ')
st.set_option('deprecation.showPyplotGlobalUse', False)


# In[ ]:


QuestionSB = st.sidebar.selectbox("Questions", ["Question 1", "Question 2", "Question 3", "Question 4", "Question 5", "Question 6", "Question 7"], key="QuestionSB")


# # General Functions

# ### Directory Looping 
# 

# In[2]:


def loop_dir(directory):
    file_list = []

    for subdir, dirs, files in os.walk(directory):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".csv"):
                file_list.append(filepath)

    return file_list


# ### Read csv
# 

# In[3]:


def get_info(directory):
    df = pd.read_csv(directory)
    cols = list(df.columns)
        
    return df


# ### Recording the MCO time frames
# This data is recorded from 'https://en.wikipedia.org/wiki/Malaysian_movement_control_order'. However, the date that I have set does not include periods of RMCO which we personally thinks that it is not qualified as a lockdown. We have chosen the date based on our own experience of locking down at home

# In[4]:


mco_period = []

mco1 = ['2020-03-18', '2020-05-03']
mco_period.append(mco1)
mco2 = ['2020-05-04', '2020-06-09']
mco_period.append(mco2)
mco3 = ['2021-06-01', '2021-10-01']
mco_period.append(mco3)


# 

# ### Comparison Plot 

# In[5]:


# This chart uses 2 y-axis so that we could visualize the both the data in the same scale for easier understanding for human

def compare_plots(df1, df1_cols, df2, df2_cols, plot_title, show_mco=True):

    start_date = df1['date'].min()
    end_date = df1['date'].max()
    
    df1['date'] = pd.to_datetime(df1['date'])
    df2['date'] = pd.to_datetime(df2['date'])
    
    mask = (df2['date'] > start_date) & (df2['date'] <= end_date)
    df2 = df2.loc[mask]
    
    fig, axs = plt.subplots(figsize=(10, 7))
    
    # Creating a shading area for MCO periods
    if show_mco == True:
        for timemco in mco_period:
            mco_start_time = dt.strptime(timemco[0], '%Y-%m-%d')
            mco_end_time = dt.strptime(timemco[1], '%Y-%m-%d')
            plt.axvspan(mco_start_time, mco_end_time,
                        color='orange', alpha=0.3, lw=0.3)


    for data in df1_cols:
        line1 = axs.plot(df1['date'], df1[data], label=data)

    ax2 = axs.twinx()
    for data2 in df2_cols:
        line2 = ax2.plot(df2['date'], df2[data2],
                         label=data2, linestyle='-', color='tab:red')
        for label in ax2.get_yticklabels():
            label.set_color("tab:red")

    axs.set_xlabel('Date')
    axs.set_ylabel(df1_cols[0], color='black')
    axs.set_title(plot_title)

    lines = line1+line2
    labs = [l.get_label() for l in lines]
    axs.legend(lines, labs, loc='upper left')
    axs.grid(linestyle='--')
    fig.tight_layout()
    plt.show()
    print('Starting at: ', start_date)
    print('Ending at: ', end_date)
    
    st.pyplot()
    st.write('Starting at: ', start_date)
    st.write('Ending at: ', end_date)


# ### Regular Ploting

# In[6]:


def simple_chart(dataframe, data_cols, x_label='date', y_label='RANGE', show_mco=True):
    linestyle = ['-']
    i = 0
    
    dataframe['date'] = pd.to_datetime(dataframe['date'])
    x = dataframe['date']

    
    fig, axs = plt.subplots(figsize=(10, 7))

    # Create a shading area for MCO Periods
    if show_mco == True:
        for timemco in mco_period:
            mco_start_time = dt.strptime(timemco[0], '%Y-%m-%d')
            mco_end_time = dt.strptime(timemco[1], '%Y-%m-%d')
            plt.axvspan(mco_start_time, mco_end_time,
                        color='orange', alpha=0.3, lw=0.3)


    for features in data_cols:
        if i > 9:
            i = 0
        else:
            plt.plot(x, dataframe[features],
                     linestyle=linestyle[0], label=features)
            i += 1

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='upper left')
    plt.grid(linestyle='--')
    plt.show()
    
    st.pyplot()


# ### Prediction vs True value scatter plot

# In[7]:


def true_vs_pred(true_value, predicted_value):
    
    plt.figure(figsize=(7,7))
    plt.scatter(true_value, predicted_value, c='crimson')
    plt.yscale('log')
    plt.xscale('log')
    
    # Creates the perfect regression line
    p1 = max(max(predicted_value), max(true_value))
    p2 = min(min(predicted_value), min(true_value))
    plt.plot([p1, p2], [p1, p2], 'b-')
    
    # Set Labels
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.show()
    
    st.pyplot()


# # Loading in the Dataframes into a array for easy access

# In[8]:


# file_list = loop_dir(dir)

# df_title = []
# df = {}

# # Get specific names for different dataframes
# for directory in file_list:
#     title  = directory[9:-4]
#     df_title.append(title)
# i = 0
# for names in df_title:
#     df[names] = get_info(file_list[i])
#     i += 1

# # Get the array list for States in Malaysia
# state_list = list(df['deaths_state']['state'][:16])
# print('There is a total of',len(state_list),'category of states')
# print('There is a total of',len(df_title),'dataframes')

file_list = next(walk(os.getcwd()+"/"+dir), (None, None, []))[2]
df_title = []
df = {}
    
for file in file_list:
    df_title.append(file[:-4])
    df[file.replace(".csv",'')] = pd.read_csv(os.path.dirname(os.getcwd()+"/"+dir)+"/"+dir+"/"+file)
    
# Get the array list for States in Malaysia
state_list = list(df['deaths_state']['state'][:16])
print('There is a total of',len(state_list),'category of states')
print('There is a total of',len(df_title),'dataframes')


# ### Population (Static Data)

# In[9]:


population = df['population'][['state', 'pop']]


# # Exploratory Data Analysis
# 
# ## (Q1) Correlation between daily vaccinations and daily cases in the country  and state level
# #### This section uses data preprocessing techniques, data structuring techniques, data analysis, data correlation and data visulization

# Fist, we decided to use a top down approach, which is to visualize the COVID-19 situation starting from the country as a whole, then only start analysing the situation state by state.

# In[10]:


if QuestionSB == "Question 1":
    st.title("Correlation between daily vaccinations and daily cases in country level")
    compare_plots(df['vax_malaysia'], ['daily_full'], df['cases_malaysia'], ['cases_new'], 'Number of Daily Vaccinations to Daily Cases', show_mco=False)


# In[11]:


from scipy.stats import pearsonr
from scipy.stats import spearmanr

q1_cases_msia_df = df['cases_malaysia']
q1_start_date = df['vax_malaysia']['date'].min()
q1_end_date = df['vax_malaysia']['date'].max()
mask = (q1_cases_msia_df['date'] > q1_start_date) & (q1_cases_msia_df['date'] <= q1_end_date)
q1_cases_msia_df = q1_cases_msia_df.loc[mask]
q1_vacc_msia_df = df['vax_malaysia']

q1_data1 = q1_cases_msia_df['cases_new']
q1_data2 = q1_vacc_msia_df['daily_full'][1:]

q1_pearson = pearsonr(q1_data1, q1_data2)
q1_spearman = spearmanr(q1_data1, q1_data2)

print('The Pearson Correlation between Daily cases and Daily Vaccination is :', q1_pearson[0])
print('\nThe Spearman Correlation between Daily cases and Daily Vaccination is:', q1_spearman[0])

if QuestionSB == "Question 1":
    st.write('The Pearson Correlation between Daily cases and Daily Vaccination is :', round(q1_pearson[0],2))
    st.write('\nThe Spearman Correlation between Daily cases and Daily Vaccination is:', round(q1_spearman[0],2))


# ### Here we tried to redefine the fully vaccination by 2nd dose + 14 days after
# #### extra data point and feature engineering

# In[12]:


# Define fully vaccinated as 2nd dose + 14 days
q1_tempnorm_df = q1_cases_msia_df.merge(q1_vacc_msia_df, how='left', on='date')

q1_fullvac_msia_df = q1_tempnorm_df
q1_fullvac_msia_df['fully_vac'] = q1_fullvac_msia_df['daily_full'].shift(14)
q1_fullvac_msia_df = q1_fullvac_msia_df.fillna(0)

# compare_plots(q1_fullvac_msia_df, ['fully_vac'], q1_fullvac_msia_df, ['cases_new'], 'Number of Full Daily Vaccinations to Daily Cases', show_mco=False)

q1_fullvac_pearson = pearsonr(q1_fullvac_msia_df['cases_new'], q1_fullvac_msia_df['fully_vac'])
q1_fullvac_spearman = spearmanr(q1_fullvac_msia_df['cases_new'], q1_fullvac_msia_df['fully_vac'])

if QuestionSB == "Question 1":
    st.header("Redefine the fully vaccination by 2nd dose + 14 days after")
    compare_plots(q1_fullvac_msia_df, ['fully_vac'], q1_fullvac_msia_df, ['cases_new'], 'Number of Full Daily Vaccinations to Daily Cases', show_mco=False)
    st.write('The Pearson Correlation between Daily cases and Daily Vaccination is :', round(q1_fullvac_pearson[0],2))
    st.write('\nThe Spearman Correlation between Daily cases and Daily Vaccination is:', round(q1_fullvac_spearman[0],2))


print('The Pearson Correlation between Daily cases and Daily Vaccination is :', q1_fullvac_pearson[0])
print('\nThe Spearman Correlation between Daily cases and Daily Vaccination is:', q1_fullvac_spearman[0])


# From both the chart and the correlation formula used on the daily cases and daily vaccinations dataset (we consider people that took 2 doses as vaccianted), it seems to have a very high positive correlation with each other(Both showing readings higher than 80). In other words, The increase in the number of vaccinations per day does not reduce the daily cases in Malaysia. We then tried to define the full vaccination as a person that has taken 2 doses + 14 days, but the results was very similar, almost identical  to the first one. However irony the results has shown, it actually is very logical and the reason behind it is very persuasive too, which we will discuss after we disaggregate the data into more minor level, observing the situation state by state.
# 
# ## (Q2) More disaggregate data analysis state by state

# In[13]:


q2_cases_state_df = {}
q2_vac_state_df ={}
q2_high_corr_state = []
q2_low_corr_state = []

if QuestionSB == "Question 2":
    st.title("Correlation between daily vaccinations and daily cases in state level")
    Question2SB = st.selectbox("State", state_list, key="Question2SB")
    for state in state_list:
        print('Currently Processing state: '+state)

        # Daily cases dataset processing 
        q2_cases_state_df[state] = df['cases_state'][df['cases_state']['state'] == state]
        q2_cases_state_df[state] = q2_cases_state_df[state][-219:]
        # Daily vaccinations dataset processing 
        q2_vac_state_df[state] = df['vax_state'][df['vax_state']['state'] == state]

        if(Question2SB == state):
            compare_plots(q2_vac_state_df[state], ['daily_full'], q2_cases_state_df[state], ['cases_new'], 'Number of Daily Vaccinations to Daily Cases: '+state, show_mco=False)
        q2_pearson = pearsonr(q2_cases_state_df[state]['cases_new'], q2_vac_state_df[state]['daily_full'])
        q2_spearman = spearmanr(q2_cases_state_df[state]['cases_new'], q2_vac_state_df[state]['daily_full'])
        print('The Pearson Correlation between Daily cases and Daily Vaccination is :', q2_pearson[0])
        print('The Spearman Correlation between Daily cases and Daily Vaccination is:', q2_spearman[0])
        print('\n')
        print('-'*100)
        
        if(Question2SB == state):
            st.write('The Pearson Correlation between Daily cases and Daily Vaccination is :', round(q2_pearson[0],2))
            st.write('The Spearman Correlation between Daily cases and Daily Vaccination is:', round(q2_spearman[0],2))


        if ((q2_pearson[0] >= 0.7) and (q2_spearman[0] >= 0.7)):
            q2_high_corr_state.append(state)

        if ((q2_pearson[0] <= 0.4) and (q2_spearman[0] <= 0.4)):
            q2_low_corr_state.append(state)


# In[14]:


print('The high correlation between daily cases and daily vaccination states consist of:', q2_high_corr_state)
print('\nThe low correlation states between daily cases and daily vaccination states consist of:', q2_low_corr_state)

if(QuestionSB == "Question 2"):
    st.write('The high correlation between daily cases and daily vaccination states consist of:', ', '.join(map(str, q2_high_corr_state)))
    st.write('\nThe low correlation states between daily cases and daily vaccination states consist of:', ', '.join(map(str, q2_low_corr_state)))


# ## To answer the example question in the project template given (v), which is to investigate if there is any correlation between vaccination and daily cases for Selangor, Sabah, Sarawak and many more. 
# 
# The answer based on the results given by a combination of visualization, Spearman's correlation and Pearson's Correlation, it has shown that the correlation between the two variables differs state by state. However, the majority of the states (9 out of 16) has shown a high and positive correlation with each other, meaning that higher vaccination number does not decrease daily cases. 4 states has shown moderate correlation and 3 states has shown very weak correlation between the two variables.
# 
# The results could be explained by the cluster formed in the vaccination centers in our opinion. As we have seen in the data, there are tens of thousand citizens being vaccinated everyday, meaning that there will be a huge gathering which might increase the posibility for COVID-19 transmission. Another theory is that after vaccinations, people feel safe to be in the public and meeting with each other, but according to an article here (https://www.cdc.gov/coronavirus/2019-ncov/vaccines/fully-vaccinated.html), it suggest that vaccination does not prevent the transmission of COVID-19 but significantly reduces the likelihood of the virus being critical to a person health. These are the two theories that we have in mind for the explanation of the results. 

# ## (Q3) What effects does vaccination has
# #### Data preprocessing and missing value detection
# 
# We decided not to apply outlier detection because we believe that it is not necessary in this case specifically for this dataset, because the cases increased at and alarming state in 2021 which might make the formula think that it is an outlier for daily cases etc.

# In[15]:


q3_icu_df = {}
q3_cases_df = {}
q3_hospital_df = {}
q3_vac_df = {}
q3_combined_df = {}

if(QuestionSB == "Question 3"):
    st.title("What effects does vaccination has")
    Question3SB = st.selectbox("State", state_list, key="Question3SB")
for state in state_list: 
    # Load in the dataframes that is needed to answer the question 
    q3_icu_df[state] = df['icu'][df['icu']['state'] == state]
    q3_cases_df[state] = df['cases_state'][df['cases_state']['state'] == state]
    q3_vac_df[state] = df['vax_state'][df['vax_state']['state'] == state]
    q3_hospital_df[state] = df['hospital'][df['hospital']['state'] == state]

    # Cut the timeframe of the dataframes into the same with vaccination data
    q3_icu_df[state] = q3_icu_df[state][-219:]
    q3_cases_df[state] = q3_cases_df[state][-219:]
    q3_hospital_df[state] = q3_hospital_df[state][-219:]

    # Merge the required features into a single dataframe for easier manipulation
    q3_combined_df[state] = q3_cases_df[state][['date', 'cases_new', 'cases_recovered']]
    q3_combined_df[state] = q3_combined_df[state].merge(q3_vac_df[state][['date', 'daily_full', 'cumul_full']], how='left', on='date')
    q3_combined_df[state] = q3_combined_df[state].merge(q3_icu_df[state][['date', 'icu_covid']], how='left', on='date')
    q3_combined_df[state] = q3_combined_df[state].merge(q3_hospital_df[state][['date', 'hosp_covid']], how='left', on='date')

    # Checking for null values 
    print('Currently at state: '+state)
    print(q3_combined_df[state].isnull().sum())
    print('-'*50)

    if(QuestionSB == "Question 3" and Question3SB == state):
        st.subheader("Missing value detection")
        st.table(q3_combined_df[state].isnull().sum())

# Replace with clustering method in the future
# Only Missing values in the state Putrajaya for 'icu_covid' and 'hosp_covid', decide to use backward fill
q3_combined_df['W.P. Putrajaya'] = q3_combined_df['W.P. Putrajaya'].fillna(method='bfill')
    
    


# ## We will now try to define a couple more features such as the cases to hospital ratio etc.
# #### Extra data point and feature engineering

# In[16]:


# Get some extra data features for EDA
for state in state_list: 
    # Get the population number and convert it into float
    q3_population = population[population['state'] == state]
    q3_population = float(q3_population['pop'])
    
    # Get the vaccination rate of states
    q3_combined_df[state]['vac_rate'] = (q3_combined_df[state]['cumul_full']/q3_population)*100
    q3_combined_df[state]['vac_rate'] = q3_combined_df[state]['vac_rate'].replace([np.inf, -np.inf], np.nan)
    
    # Get the hotpital_covid to daily cases ratio
    # Due to the ratio formula, there will be infinite value if the daily cases is 0, however the inf values are relatively small, therefore filling with value 1
    q3_combined_df[state]['hosp2cases_ratio'] = (q3_combined_df[state]['hosp_covid']/q3_combined_df[state]['cases_new'])*100
    q3_combined_df[state]['hosp2cases_ratio'] = q3_combined_df[state]['hosp2cases_ratio'].replace([np.inf, -np.inf], np.nan)
    q3_combined_df[state]['hosp2cases_ratio'] = q3_combined_df[state]['hosp2cases_ratio'].fillna(1)

    # Get the icu_covid to daily cases ratio
    q3_combined_df[state]['icu2cases_ratio'] = (q3_combined_df[state]['icu_covid']/q3_combined_df[state]['cases_new'])*100
    q3_combined_df[state]['icu2cases_ratio'] = q3_combined_df[state]['icu2cases_ratio'].replace([np.inf, -np.inf], np.nan)
    q3_combined_df[state]['icu2cases_ratio'] = q3_combined_df[state]['icu2cases_ratio'].fillna(1)

    # Get the icu_covid to hosp_covid ratio
    q3_combined_df[state]['icu2hosp_ratio'] = (q3_combined_df[state]['icu_covid']/q3_combined_df[state]['hosp_covid'])*100
    q3_combined_df[state]['icu2hosp_ratio'] = q3_combined_df[state]['icu2hosp_ratio'].replace([np.inf, -np.inf], np.nan)
    q3_combined_df[state]['icu2hosp_ratio'] = q3_combined_df[state]['icu2hosp_ratio'].fillna(1)

    # Check if there is any null value in the dataframe
#     print(q3_combined_df[state].isnull().sum())


# In[17]:


q3_neg_corr_state = []
q3_pos_corr_state = []
q3_high_corr_state = []

# Visualize the comparison and calculate the correlation
for state in state_list:
    print('-'*100)
    print('Current State: ' +state)
    if(QuestionSB == "Question 3" and Question3SB == state):
        st.subheader("Correlation between Vacinnation rate, ICU case ratio and ICU to hospital ratio")
        simple_chart(q3_combined_df[state],['vac_rate', 'icu2cases_ratio', 'icu2hosp_ratio'], show_mco=False)
    q3_pearson = pearsonr(q3_combined_df[state]['vac_rate'], q3_combined_df[state]['icu2cases_ratio'])
    q3_spearman = spearmanr(q3_combined_df[state]['vac_rate'], q3_combined_df[state]['icu2cases_ratio'])
    print('The Pearson Correlation between Daily cases and Daily Vaccination is :', q3_pearson[0])
    print('The Spearman Correlation between Daily cases and Daily Vaccination is:', q3_spearman[0])
    
    if(QuestionSB == "Question 3" and Question3SB == state):
        st.write('The Pearson Correlation between Daily cases and Daily Vaccination is :', round(q3_pearson[0],2))
        st.write('The Spearman Correlation between Daily cases and Daily Vaccination is:', round(q3_spearman[0],2))
    
    if ((q3_pearson[0] <= 0) and (q3_spearman[0] <= 0)):
        q3_neg_corr_state.append(state)
        
    if ((q3_pearson[0] >= 0) and (q3_spearman[0] >= 0)):
        q3_pos_corr_state.append(state)
        
    if ((q3_pearson[0] <= -0.5) and (q3_spearman[0] <= -0.5)):
        q3_high_corr_state.append(state)
    


# In[18]:


print('The state that consist of negative correlation contains:', q3_neg_corr_state)
print('\nThe state that consist of positive correlation contains:', q3_pos_corr_state)
print('\nThe states that has shown significant effect:', q3_high_corr_state)

if(QuestionSB == "Question 3"):
    st.write('The state that consist of negative correlation contains:', ', '.join(map(str, q3_neg_corr_state)))
    st.write('\nThe state that consist of positive correlation contains:', ', '.join(map(str, q3_pos_corr_state)))
    st.write('\nThe states that has shown significant effect:', ', '.join(map(str, q3_high_corr_state)))
    st.write("Observing from the results, the majority state(14 states out of 16) has shown that the population vaccination rate has a negative correlation with the ICU to Cases ratio. In other words, the higher vaccination rate, the lower the ICU to Cases ratio, unlike the first part of the EDA where it results that the daily vaccinations does not lower the daily cases. Kedah, Sabah and Sarawak has shown the most effect of vacciantions on icu to daily cases ratio.")
    st.write("In a nut shell, although vaccination rates might not have much effect on daily cases, the vaccines does help in reducing ratio of ICU to daily cases, meaning that the probability of the virus causing fatal health issues has been reduced by higher rates of vaccinations of the population.")


# ### Answering the question of the effects of vaccines and the states that has shown the most effective results
# 
# Observing from the results, the majority state(14 states out of 16) has shown that the population vaccination rate has a negative correlation with the ICU to Cases ratio. In other words, the higher vaccination rate, the lower the ICU to Cases ratio, unlike the first part of the EDA where it results that the daily vaccinations does not lower the daily cases. Kedah, Sabah and Sarawak has shown the most effect of vacciantions on icu to daily cases ratio.
# 
# In a nut shell, although vaccination rates might not have much effect on daily cases, the vaccines does help in reducing ratio of ICU to daily cases, meaning that the probability of the virus causing fatal health issues has been reduced by higher rates of vaccinations of the population.

# # (Q4) Assuming herd immunity by 80% of the population vaccinated, could we achieve that by middle of November 2021. 
# 
# ### To answer this question, we have decided to use a regression model to predict the vaccination rate at middle of  November 2021. The features that we will be using is focused on the 'vax_reg' dataset because we believe that the logical effector of vacciantion rate is the number of registration for vaccination and the volume of vaccines purchased, which is not available, so we will be working on the registration rate.

# In[19]:


q4_combined_msia_df = {}
q4_vaxreg_msia_df = df['vaxreg_malaysia']
q4_vaxreg_msia_df['date'] = pd.to_datetime(q4_vaxreg_msia_df['date'])
q4_vax_msia_df = df['vax_malaysia']
q4_vax_msia_df['date'] = pd.to_datetime(q4_vax_msia_df['date'])
q4_pop_msia = population[population['state'] == 'Malaysia']
q4_pop_msia = float(q4_pop_msia['pop'])

# Combine the required features into one dataframe for easier manipulation
q4_combined_msia_df = q4_vax_msia_df[['date','daily_partial', 'daily_full','cumul_partial', 'cumul_full']]
q4_combined_msia_df = q4_combined_msia_df.merge(q4_vaxreg_msia_df[['date', 'total']], how='left', on='date')
q4_combined_msia_df.rename(columns={'total': 'vaxreg_total'}, inplace=True)

# Some data feature engineering and data manipulation on dataset
q4_combined_msia_df['vaxreg_diff'] = q4_combined_msia_df['vaxreg_total'].diff()
q4_combined_msia_df['vaxreg2pop_ratio'] = q4_combined_msia_df['vaxreg_total']/q4_pop_msia
q4_combined_msia_df['vaxfull2pop_ratio'] = q4_combined_msia_df['cumul_full']/q4_pop_msia
q4_combined_msia_df['vaxpartial2pop_ratio'] = q4_combined_msia_df['cumul_partial']/q4_pop_msia
# Shift the 'cumul_full' and 'cumul_partial' down 6 weeks (more explanation in report)
q4_combined_msia_df['vaxfull2pop_ratio_shifted'] = q4_combined_msia_df['vaxfull2pop_ratio'].shift(-42)

# Clear the nan values in this case instead of filling it
q4_temp_df = q4_combined_msia_df.copy()
q4_combined_msia_df = q4_combined_msia_df.dropna()


# In[1]:


# Feature Selection

if(QuestionSB == "Question 4"):
    st.title("Assuming herd immunity by 80% of the population vaccinated, could we achieve that by the middle of November 2021?")
    simple_chart(q4_combined_msia_df, ['vaxreg2pop_ratio', 'vaxfull2pop_ratio', 'vaxpartial2pop_ratio'], show_mco=False)
q4_corr_df = q4_combined_msia_df.corr()
q4_vac_corr_df = q4_corr_df.loc[q4_corr_df.index == 'vaxfull2pop_ratio_shifted']
# q4_vac_corr_df = q4_corr_df.loc[q4_corr_df.index == 'cumul_full']

# Ploting the heatmap and shows the features that has a correlation higher than 0.95

if(QuestionSB == "Question 4"):
    plt.figure(figsize=(18,2))
    ax = sns.heatmap(q4_vac_corr_df, vmin=-1, vmax=1, cbar=False, cmap='coolwarm', annot=True)
    # Set the condition so that only features with 0.7 or higher correlation will show 
    for text in ax.texts:
        t = float(text.get_text())
        if -0.95 < t < 0.95:
            text.set_text('')
        else:
            text.set_text(round(t, 2))
        text.set_fontsize('x-large')
    plt.xticks( size='large')
    plt.yticks(rotation=0, size='large')
    plt.show()
    st.header("Correlation Analysis")
    st.pyplot()
q4_strong_features = q4_corr_df['vaxfull2pop_ratio_shifted']
# q4_strong_features = q4_corr_df['cumul_full']

q4_strong_features = q4_strong_features.loc[(q4_strong_features > 0.95)|(q4_strong_features < -0.95)]
q4_training_features = list(q4_strong_features.index)

print('The', len(q4_training_features), 'out of 10 features that is highly correlated are:\n',q4_training_features)
if(QuestionSB == "Question 4"):
    st.write('The', len(q4_training_features), 'out of 10 features that is highly correlated are:\n', ', '.join(map(str, q4_training_features)))


# In[21]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder

# Data Preprocessing for machine learning
q4_training_df = q4_combined_msia_df[q4_training_features]

q4_training_df = q4_training_df.dropna()
q4_y = q4_training_df['vaxfull2pop_ratio_shifted']
# q4_y = q4_training_df['cumul_full']

# Remove this because it is a label
q4_X = q4_training_df.drop(['vaxfull2pop_ratio_shifted'], axis=1)
# q4_X = q4_training_df.drop('cumul_full', axis=1)

q4_X_train, q4_X_test, q4_y_train, q4_y_test = train_test_split(q4_X, q4_y, test_size=0.2, random_state = random_seed)

# Defining the Regression Model 
q4_lr = LinearRegression(normalize=False, n_jobs=-1)

# Defining Decision Tree Model
q4_dt = DecisionTreeClassifier(criterion='gini', max_depth=3)

# Model Training & Prediction for Linear Regression
q4_lr.fit(q4_X_train, q4_y_train)
q4_predictions = q4_lr.predict(q4_X_test)

# Model Training & Prediction for Decision Tree Clasifier
q4_label_enc = LabelEncoder()
q4_encoded = q4_label_enc.fit_transform(q4_y_train)
q4_dt.fit(q4_X_train, q4_encoded)
q4_dt_predictions = q4_dt.predict(q4_X_test)
q4_prob_DT = q4_dt.predict_proba(q4_X_test)[:, 1]

# Model Evalutation for Linear Regression
q4_r2 = q4_lr.score(q4_X_test, q4_y_test)
q4_mae = mean_absolute_error(q4_y_test, q4_predictions)
q4_mape = mean_absolute_percentage_error(q4_y_test, q4_predictions)
print('From the test results, these are the accuracy of the model for linear regression.')
print('\nThe R Squared, coefficient of determination is:', round(q4_r2,2))
print('The Mean Absolute Error is:', round(q4_mae,2))
print('The Mean Absolute Percentage Error is:', round(q4_mape,2))
# true_vs_pred(q4_y_test, q4_predictions)

if(QuestionSB == "Question 4"):
    st.header("Deviation charts")
    true_vs_pred(q4_y_test, q4_predictions)
    st.write('From the test results, these are the accuracy of the model for linear regression.')
    st.write('\nThe R Squared, coefficient of determination is:', round(q4_r2,2))
    st.write('The Mean Absolute Error is:', round(q4_mae,2))
    st.write('The Mean Absolute Percentage Error is:', round(q4_mape,2))

print('-'*100)

# Model Evaluation for Decision Tree Classifier
q4_y_test_encode = q4_label_enc.fit_transform(q4_y_test)
q4_dt_r2 = q4_dt.score(q4_X_test, q4_y_test_encode)
q4_dt_mae = mean_absolute_error(q4_y_test_encode, q4_dt_predictions)
q4_dt_mape = mean_absolute_percentage_error(q4_y_test_encode, q4_dt_predictions)
print('From the test results, these are the accuracy of the model for Decision Tree.')
print('\nThe R Squared, coefficient of determination is:', round(q4_dt_r2,2))
print('The Mean Absolute Error is:', round(q4_dt_mae,2))
print('The Mean Absolute Percentage Error is:', round(q4_dt_mape,2))

if(QuestionSB == "Question 4"):
    true_vs_pred(q4_y_test_encode, q4_dt_predictions)
    st.write('From the test results, these are the accuracy of the model for Decision Tree.')
    st.write('\nThe R Squared, coefficient of determination is:', round(q4_dt_r2,2))
    st.write('The Mean Absolute Error is:', round(q4_dt_mae,2))
    st.write('The Mean Absolute Percentage Error is:', round(q4_dt_mape,2))

# true_vs_pred(q4_y_test_encode, q4_dt_predictions)
# q4_dt_fpr, q4_dt_tpr, q4_dt_ = roc_curve(q4_y_test_encode,  q4_prob_DT)
# plt.plot(q4_dt_fpr, q4_dt_tpr, label='Decision Tree')
# plt.title('Decision Tree ROC Curve')
plt.show()


# The regression model is chosen because of it's better performance

# In[22]:


# Predict the vaccination to population rate at early November
q4_X_prediction = q4_temp_df[list(q4_X.columns)].tail(1)
q4_midnov_predict = q4_lr.predict(q4_X_prediction)
print('\nThe % of population vaccinated by mid November forecasted is:', q4_midnov_predict*100,"%")
if(QuestionSB == "Question 4"):
    # Todo: bold it
    st.write('\nThe % of population vaccinated by mid November forecasted is:', str(q4_midnov_predict*100),"%")


# ### According to the model, the percentage of population that will be fully vaccinated by middle of November will be 69.2%. We have chosen to predict the middle of November because our theory is that there is a 6 week gap between registration and fully vaccination. Building on this theory, we have manipulated our label, 'vaxfull2pop_ratio' 42 days backwards to train the model with the registration rate, daily vaccinations number etc, making it aggregate the current data and train it to forecast the label 42 days down the line. In our data, the ending date is 30-September, making it the middle of November after 6 weeks.

# ## (Q5) Building a model to forecast daily cases at a state level using "Johor", "Sarawak", "Selangor", "Pahang", "Kedah"

# In[23]:


q5_state_list = ["Johor", "Sarawak", "Selangor", "Pahang", "Kedah"]

# Remove unwanted dataframes due to variable reasons, explained more in report
# Comment the codes or try restarting kernel
q5_df_title = df_title
q5_df_title.remove('cases_malaysia')
q5_df_title.remove('checkin_malaysia')
q5_df_title.remove('deaths_malaysia')
q5_df_title.remove('population')
q5_df_title.remove('tests_malaysia')
q5_df_title.remove('tests_state')
q5_df_title.remove('trace_malaysia')
q5_df_title.remove('vax_malaysia')
q5_df_title.remove('vax_state')
q5_df_title.remove('vaxreg_malaysia')
q5_df_title.remove('vaxreg_state')

print(q5_df_title)


# In[24]:


q5_combined_df = {}

if(QuestionSB == "Question 5"):
    for q5_states in q5_state_list:
        i = 0
        for q5_title in q5_df_title:
            q5_df = df[q5_title]
            q5_df['date'] = pd.to_datetime(q5_df['date'])
            print('Currently Extracting ' + q5_title +' from '+ q5_states)
            q5_temp_df = q5_df[q5_df['state'] == q5_states]
            if i == 0: 
                q5_combined_df[q5_states] = q5_temp_df

            else:
                q5_combined_df[q5_states] = q5_combined_df[q5_states].merge(q5_temp_df, how='left', on='date')
            i+=1

    for states in q5_state_list:
        q5_combined_df[states] = q5_combined_df[states].drop(columns=['state_x', 'state_y'])
        q5_combined_df[states]['state'] = states
    


# ### Feature Selection

# In[25]:


# Data checking on dataframe
# When we merge and isolate the dataframe by state in this very particular way, there will be null values for the 'pkrc'
# part of the dataframe because some states does not record that data in that particular time. 
# We've decide to fill the null values with 0 after examining the csv file, recognizing during that period, Malaysia is under
# a state where there is minimal infections and the other states that has recorded the data shows data very close to zero.
q5_top_corr_features = []
Question5SB = ""

if(QuestionSB == "Question 5"):
    st.title("Building a model to forecast daily cases using datasets from Johor, Sarawak, Selangor, Pahang and Kedah")
    # Pearson's correlation
    if(QuestionSB == "Question 5"):
        Question5SB = st.selectbox("State", q5_state_list, key="Question3SB")
    
    for q5_corr_state in q5_state_list:
        print('-'*60)
        print('Currently processing:' +q5_corr_state)
        # Check for null values
    #     print(q5_combined_df[q5_state].isnull().sum())
        q5_corr_df = q5_combined_df[q5_corr_state]
        q5_corr_df = q5_corr_df.fillna(0)
        q5_corr_df = q5_corr_df.drop(columns=['state'])
        q5_corr_df = q5_corr_df.corr()
        q5_cases_corr_df = q5_corr_df.loc[q5_corr_df.index == 'cases_new']
        # Ploting the heatmap and shows the features that has a correlation higher than 0.7
        plt.figure(figsize=(22,2))
        ax = sns.heatmap(q5_cases_corr_df, vmin=-1, vmax=1, cbar=False,
                         cmap='coolwarm', annot=True)
        # Set the condition so that only features with 0.7 or higher correlation will show 
        for text in ax.texts:
            t = float(text.get_text())
            if -0.7 < t < 0.7:
                text.set_text('')
            else:
                text.set_text(round(t, 2))
            text.set_fontsize('large')
        plt.xticks( size='large')
        plt.yticks(rotation=0, size='large')
        plt.show()
        if(QuestionSB == "Question 5" and Question5SB == q5_corr_state):
            st.header("Correlation Analysis")
            st.pyplot()

        strong_features = q5_corr_df['cases_new']
        strong_features = strong_features.loc[strong_features > 0.75]
        strong_features_list = list(strong_features.index)
        for corr_feat in strong_features_list:
            q5_top_corr_features.append(corr_feat)
    


# In[26]:


# print(len(q5_top_corr_features))


# In[27]:


from sklearn.feature_selection import mutual_info_classif

threshold = 10
q5_top_mutual_features = []

if(QuestionSB == "Question 5"):
    # Mutual info classifiaction filtering 
    for q5_filter_states in q5_state_list:
        print('-'*60)
        print('Currently processing state: '+ q5_filter_states)
        q5_filter_df = q5_combined_df[q5_filter_states]
        filter_y = q5_filter_df['cases_new']
        filter_X = q5_filter_df.drop(columns=['state', 'cases_new', 'date'])
        filter_X = filter_X.fillna(0)

        # Plot a barchart for easier visualization 
        filter_feature_score = mutual_info_classif(filter_X, filter_y)
        feat_importances = pd.Series(filter_feature_score, filter_X.columns)
        plt.figure(figsize=(15, 7))
        feat_importances.plot(kind='bar')
        plt.show()
        if(QuestionSB == "Question 5" and Question5SB == q5_filter_states):
            st.header("Top Mutual Info Classification")
            st.pyplot()

        # Display the top 10 most relevant features based on mutual info classification 
        print('Top 10 most relevant')
        temp = pd.DataFrame([], columns=['Columns', 'Score'])  
        for score, f_name in sorted(zip(filter_feature_score, filter_X.columns), reverse=True)[:threshold]:
            print(f_name, score) 
            temp = temp.append({'Columns':f_name, 'Score':score}, ignore_index=True)
            q5_top_mutual_features.append(f_name)
        if(QuestionSB == "Question 5" and Question5SB == q5_filter_states):
            st.table(temp)
 


# In[28]:


if(QuestionSB == "Question 5"):
    # Get the repeated features in mutual_info_class
    mutual_word_counts = collections.Counter(q5_top_mutual_features)
    repeated_mutual_features = []

    for mutual_word, mutual_count in sorted(mutual_word_counts.items()):
        print('"%s" is repeated %d time%s.' % (mutual_word, mutual_count, "s" if mutual_count > 1 else ""))
        if mutual_count >= 3:
            repeated_mutual_features.append(mutual_word)

    print('\nMutual_info_classification features:' ,repeated_mutual_features)
    print('-'*100)

    # Get the repeated features in correlation table 
    repeated_corr_features = []
    corr_word_counts = collections.Counter(q5_top_corr_features)

    for corr_word, corr_count in sorted(corr_word_counts.items()):
        print('"%s" is repeated %d time%s.' % (corr_word, corr_count, "s" if corr_count > 1 else ""))
        if corr_count >= 3:
            repeated_corr_features.append(corr_word)

    print("\nPearson's correlation features:" ,repeated_corr_features)
    print('-'*100)

    # Extract the features for the chosen columns later in the model training phase 
    joined_features = repeated_corr_features + repeated_mutual_features
    repeated_joined_features = []
    joined_word_counts = collections.Counter(joined_features)

    for joined_word, joined_count in sorted(joined_word_counts.items()):
        print('"%s" is repeated %d time%s.' % (joined_word, joined_count, "s" if joined_count > 1 else ""))
        if joined_count == 2:
            repeated_joined_features.append(joined_word)

    training_features = joined_features

    for repeated in repeated_joined_features:
        training_features.remove(repeated)

    print('\nFeatures that will be used in model training: ',training_features)
    print('\nTotal number of training features: ', len(training_features))


# In[29]:


# Model defining
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

if(QuestionSB == "Question 5"):
    random_seed = 9

    q5_classifier_names = ['rf_class', 'rf_reg']
    q5_classifiers = {}

    rf_class_clf = RandomForestClassifier(n_estimators=200, criterion='gini', max_depth = 5, random_state=random_seed, n_jobs=-1)
    q5_classifiers['rf_class'] = rf_class_clf

    rf_reg_clf = RandomForestRegressor(n_estimators=1000, criterion='mae', random_state=1, n_jobs=-1)
    q5_classifiers['rf_reg'] = rf_reg_clf


# In[30]:


mae_score_list = []

if(QuestionSB == "Question 5"):
    for q5_states in q5_state_list:
        q5_model_df = q5_combined_df[q5_states]
        q5_model_df = q5_model_df.fillna(0)
        q5_y = q5_model_df['cases_new']
        q5_X = q5_model_df[training_features]
        q5_X_train, q5_X_test, q5_y_train, q5_y_test = train_test_split(q5_X, q5_y, test_size = 0.2, random_state = random_seed)

        # Model Training
        for classifiers in q5_classifier_names:
            print('Currently Training '+classifiers+' model for ' + q5_states)
            q5_classifiers[classifiers].fit(q5_X_train, q5_y_train)

            # Evalutation 
            q5_predictions = q5_classifiers[classifiers].predict(q5_X_test)
            q5_mae = mean_absolute_error(q5_y_test, q5_predictions)
            q5_mape = mean_absolute_percentage_error(q5_y_test, q5_predictions)
            q5_mae_score_round = round(q5_mae, 3)
            mae_score_list.append(q5_mae_score_round)
#             q5_mape_score_round  = round(q5_mape, 3)
            print('The MAE for'+classifiers+' is:', q5_mae_score_round)
#             print('The Mean Absolute Percentage Error is:', q5_mape_score_round)
            if(QuestionSB == "Question 5" and Question5SB == q5_states):
                st.header(classifiers+' model for ' + q5_states)
                st.write('The MAE for'+classifiers+' is:', q5_mae_score_round)
#                 st.write('The Mean Absolute Percentage Error is:', q5_mape_score_round)
                true_vs_pred(q5_y_test, q5_predictions)
            print('-'*100)

        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.bar(q5_classifier_names, mae_score_list)
        ax.set_ylabel('MAE Score')
        ax.set_title('Performance of each model in ' + q5_states)
        plt.show()
        if(QuestionSB == "Question 5" and Question5SB == q5_states):
            st.header("Performance Comparison Chart")
            st.pyplot()
        print('*'*100)
        mae_score_list.clear()


# # Sudish Part 

# In[32]:


if(QuestionSB == "Question 6"):
    st.title("Use Clustering method to cluster daily cases at a Country level")
    #Define a function to extract the appropriate Covid data dataset
    def extract_cases_df(df_name):
        return pd.DataFrame(dictionary_all_cases_df[df_name])
    #Define a function to extract the appropriate Vaccination data dataset
    def extract_vacc_df(df_name):
        return pd.DataFrame(dictionary_all_vacc_df[df_name])

    # name list of Cases datasets
    # Make sure that you're connected to internet
    dataframe_names = ['cases_malaysia' , 'cases_state' , 'deaths_malaysia' ,
                       'deaths_state' , 'clusters' , 'pkrc' , 'hospital' ,
                       'icu' , 'tests_malaysia' , 'tests_state']
    # Make an empty dictionary to store all datasets in it
    dictionary_all_cases_df = {}
    for i in dataframe_names:
        dictionary_all_cases_df[i] = pd.read_csv(f'https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/epidemic/{i}.csv', header= 0 ,date_parser=['date'])

    # name list of datasets
    # Make sure that you're connected to internet
    df_names = {'vaccination' : ['vax_malaysia' , 'vax_state'] ,
                'registration' : ['vaxreg_malaysia' , 'vaxreg_state'],
                'static' : ['population']}
    # Make an empty dictionary to store all datasets in it
    dictionary_all_vacc_df = {}
    for key in df_names.keys():
        for val in df_names[key]:
            dictionary_all_vacc_df[val] = pd.read_csv(f'https://raw.githubusercontent.com/CITF-Malaysia/citf-public/main/{key}/{val}.csv', header= 0 ,date_parser=['date'])


# In[34]:


if(QuestionSB == "Question 6"):
    # Important Cases Datasets
    df_cases_malaysia = extract_cases_df('cases_malaysia')
    df_deaths_malaysia = extract_cases_df('deaths_malaysia')
    df_tests_malaysia = extract_cases_df('tests_malaysia')
    df_pkrc = extract_cases_df('pkrc')
    df_hospital = extract_cases_df('hospital')
    df_icu = extract_cases_df('icu')
    # Important Vaccinations Datasets
    df_vax_malaysia = extract_vacc_df('vax_malaysia')
    df_vaxreg_malaysia = extract_vacc_df('vaxreg_malaysia')
    df_population = extract_vacc_df('population')


# In[39]:


if(QuestionSB == "Question 6"):
    # We should make a appropraite dataframe from our datasets for more analysis 
    df_Malayisa = df_cases_malaysia.loc[:,['date' ,'cases_new' , 'cases_active' , 'cases_recovered',
                                          'cases_child' , 'cases_adolescent' ,'cases_adult',
                                          'cases_elderly']]
    # concatinate the tests dataset to the main dataset
    df_Malayisa = df_Malayisa.merge(df_tests_malaysia , how = 'left' , on = 'date')
    # concatinate the deaths dataset to the main dataset
    df_Malayisa = df_Malayisa.merge(df_deaths_malaysia[['date' , 'deaths_new' , 'deaths_bid']] ,
                                    how = 'left', on= 'date').fillna(0)
    # concatinate the pkrc dataset to the main dataset, so first we should grouped this dataset
                                      # based on date 
    df_Malayisa = df_Malayisa.merge(df_pkrc.groupby('date').sum().reset_index()[['date','beds' , 'pkrc_covid']],
                                    how = 'left', on = 'date')
    df_Malayisa.rename(columns = {'beds' : 'beds_pkrc'} , inplace = True)
    # concatinate the hospital dataset to the main dataset, so first we should grouped this dataset
                                      # based on date
    df_Malayisa = df_Malayisa.merge(df_hospital.groupby('date').sum().reset_index()[['date','beds','hosp_covid']],
                                    how = 'left', on = 'date')
    df_Malayisa.rename(columns = {'beds' : 'beds_hosp'} , inplace = True)

    # concatinate the icu dataset to the main dataset, so first we should grouped this dataset
                                      # based on date
    df_Malayisa = df_Malayisa.merge(df_icu.groupby('date').sum().reset_index()[['date','beds_icu', 'beds_icu_total',
                                                                                'beds_icu_covid' ,'vent','icu_covid']],
                                    how = 'left', on = 'date')

    # concatinate the vaccination dataset to the main dataset
    df_Malayisa = df_Malayisa.merge(df_vax_malaysia[['date' , 'daily' , 'daily_partial' , 'daily_full', 'cumul',
                                                     'pfizer1', 'pfizer2', 'sinovac1',	'sinovac2',	'astra1',
                                                     'astra2', 'cansino' ]] , how = 'left' , on = 'date').fillna(0)


    #To evaluate the effect of new strains of this virus on the age of new patients,
                                                        # we considered it necessary to
                                                        # obtain the share of each age group
                                                        # in cases of new infections.
    df_Malayisa['perc_child_case'] = round(df_Malayisa['cases_child'] / df_Malayisa['cases_new'],3 ) * 100
    df_Malayisa['perc_adolescent_case'] = round(df_Malayisa['cases_adolescent'] / df_Malayisa['cases_new'],3 ) * 100
    df_Malayisa['perc_adult_case'] = round(df_Malayisa['cases_adult'] / df_Malayisa['cases_new'],3 ) * 100
    df_Malayisa['perc_elderly_case'] = round(df_Malayisa['cases_elderly'] / df_Malayisa['cases_new'],3 ) * 100

    df_Malayisa['cases_new_lag14'] = df_Malayisa['cases_new'].shift(14)

    #To evaluate the condition of the country's medical system, we had to evaluate the 
                                                  #facilities of this system in different
                                                  # conditions per number of active patients.
    df_Malayisa['pkrc_beds_acive_cases'] = round(df_Malayisa['pkrc_covid'] / df_Malayisa['cases_active'] , 3 ) * 100
    df_Malayisa['hosp_beds_active_cases'] = round(df_Malayisa['hosp_covid'] / df_Malayisa['cases_active'] , 3 ) * 100
    df_Malayisa['icu_beds_acive_cases'] = round(df_Malayisa['icu_covid'] / df_Malayisa['cases_active'] , 3 ) * 100
    df_Malayisa['vent_icu_beds'] = round(df_Malayisa['vent'] / df_Malayisa['icu_covid'] , 3 ) * 100


    df_Malayisa.fillna(0 , inplace= True)
    df_Malayisa.date = pd.to_datetime(df_Malayisa.date)


# # PCA

# In[42]:


if(QuestionSB == "Question 6"):
    features = df_Malayisa.drop(columns = [ 'date' ,'vent_icu_beds'], axis = 1).values


# In[43]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

if(QuestionSB == "Question 6"):
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    features = pd.DataFrame(features)
    pca = PCA().fit(features)


# In[44]:


if(QuestionSB == "Question 6"):
    plt.bar(range(10) , pca.explained_variance_ratio_[:10] )
    st.header('PCA Explained Ratio for first 10 Component')
    st.pyplot()


# # Clustering the days during covid

# Finally, we clustered the corona days in Malaysia. In this regard, we first randomly set the number of clusters to 5 and clustered the data using the K-Means algorithm. Of course, before doing this step, I normalized the data using the standardscaler module in the data sklearn library.

# ## K-Means Clustering

# In[45]:


if(QuestionSB == "Question 6"):
    features.columns = ['cases_new', 'cases_active', 'cases_recovered',
                                                 'cases_child', 'cases_adolescent', 'cases_adult',
                                                 'cases_elderly', 'rtk-ag', 'pcr', 'deaths_new',
                                                 'deaths_bid', 'beds_pkrc', 'pkrc_covid', 
                                                 'beds_hosp', 'hosp_covid', 'beds_icu', 'beds_icu_total',
                                                 'beds_icu_covid', 'vent', 'icu_covid', 'daily', 
                                                 'daily_partial', 'daily_full', 'cumul', 'pfizer1', 'pfizer2',
                                                 'sinovac1', 'sinovac2', 'astra1', 'astra2', 'cansino',
                                                 'perc_child_case', 'perc_adolescent_case', 'perc_adult_case',
                                                 'perc_elderly_case', 'cases_new_lag14', 'pkrc_beds_acive_cases',
                                                 'hosp_beds_active_cases', 'icu_beds_acive_cases']

    features.head()
    features.set_index(df_Malayisa.date , inplace = True)


# ## First Clustering application

# In[53]:


from sklearn.cluster import KMeans

if(QuestionSB == "Question 6"):
    seg_km1 = KMeans(n_clusters=5 , init= 'random' ,
                     random_state = 123 , n_init = 1).fit(features)

    df_Malayisa['seg_km1'] = seg_km1.predict(features)

    plt.scatter(x = df_Malayisa['date'] , y = df_Malayisa['cases_new'],
                c = df_Malayisa['seg_km1'] , alpha = .3 )

    plt.xlabel('date')
    plt.ylabel('New daily Cases')
    st.header("First Clustering application")
    st.pyplot()


# ## Second Clustering Application

# In[52]:


if(QuestionSB == "Question 6"):
    seg_km2 = KMeans(n_clusters=5 , init= 'k-means++' ,
                     random_state = 1234 , n_init = 15).fit(features)

    df_Malayisa['seg_km2'] = seg_km2.predict(features)

    plt.scatter(x = df_Malayisa['date'] , y = df_Malayisa['cases_new'],
                c = df_Malayisa['seg_km2'] , alpha = .3 )

    plt.xlabel('date')
    plt.ylabel('New daily Cases')
    st.header("Second Clustering application")
    st.pyplot()


# ## Finding the Optimal numbers of clusters
# 
# #### Elbow Method

# In[55]:


sse = []

if(QuestionSB == "Question 6"):
    for k in range(1,11):
      kmeans = KMeans(n_clusters= k , init= 'k-means++' , n_init = 10)
      kmeans.fit(features)
      sse.append(kmeans.inertia_)

    plt.plot(range(1,11) , sse)
    plt.xticks(range(1,11))
    plt.xlabel('Num of Clusters')
    plt.ylabel('SSE')
    plt.title('Elbow Method')

    plt.show()
    st.header("Optimal numbers of clusters")
    st.pyplot()


# #### Silhouette Coefficient

# In[57]:


from sklearn.metrics import silhouette_score
silhouette_coefficients = []

if(QuestionSB == "Question 6"):
    for k in range(2, 11):
        kmeans = KMeans(n_clusters = k, init = 'k-means++', random_state = 1234, n_init = 10)
        kmeans.fit(features)
        score = silhouette_score(features, kmeans.labels_)
        silhouette_coefficients.append(score)

    plt.plot(range(2, 11), silhouette_coefficients)
    plt.xticks(range(2, 11))
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Coefficient")
    plt.title('Silhouette Method')
    plt.show()
    st.header("Silhouette Coefficient")
    st.pyplot()


# ## Final clustering

# In[58]:


if(QuestionSB == "Question 6"):
    seg_km_final = KMeans(n_clusters=2 , init= 'k-means++' ,
                     random_state = 1234 , n_init = 15).fit(features)

    df_Malayisa['seg_km_final'] = seg_km_final.predict(features)

    plt.scatter(x = df_Malayisa['date'] , y = df_Malayisa['cases_new'],
                c = df_Malayisa['seg_km_final'] , alpha = .3 )

    plt.xlabel('date')
    plt.ylabel('New daily Cases')
    st.header("Final clustering")
    st.pyplot()
    st.write("The final number of clusters used is 2, and the number of initial setup is set to 15. ")


# In[ ]:


from copy import deepcopy
from tqdm import tqdm

if(QuestionSB == "Question 7"):
    st.title("Association Rule mining for COVID-19 symptoms")
    
    def discretize_df(data, cols):
        df = deepcopy(data)
        count2 = 0
        count1 = 0
        count3 = 0
        for col in cols:
            try:
                df[col] = pd.qcut(df[col], 3, labels=["l","m","h"])
                count3 += 1
            except:
                try:
                    df[col] = pd.qcut(df[col], 2, labels=["l","h"])
                    count2 += 1
                except:
                    df[col] = pd.qcut(df[col], 1, labels=["m"])
                    count1 += 1
        print("The numbers of 1-3 are: {}, {}, {}".format( count1, count2, count3))
        return df

    def reshape_into_mlextend_format(df): # returns True/False matris that shows the presence/absence of an item (e.g., daily_new_cases:low=True)
        new_df = pd.DataFrame()
        index=0
        for c in tqdm(df.columns):
            for cut in df[c].unique(): # for recoded it is  {l,m,h} for categorical: it is unique category value 
                new_df.insert(index,'{}:{}'.format(c,cut),df[c]==cut, allow_duplicates=True)
                index+=1
        return new_df
    
    # Data Preprocessing with discretization
    data=pd.read_csv('df_Malayisa.csv',index_col='date')
    cols = data.columns[:-3]
    data_recoded = discretize_df(data, cols)
    
    st.header("Data Preprocessing with discretization")
    st.write(data_recoded.astype('object'))
#     st.table(data_recoded.head())


# ### (Comment) 
# - recoding analysis
#     - there are 11 columns which were recoded only as 'm' for having single unique values
#     - 1 column is recoded only  `low` and `high` 
#     - other 28 columns were recoded as `low`, `medium` and `high`
#     - kmeans clusters need not to be recoded 

# In[ ]:


if(QuestionSB == "Question 7"):
    data_reshaped = reshape_into_mlextend_format(data_recoded)

    # Reshaped dataframe
    st.header("Reshaped dataframe")
    st.write(data_reshaped.astype('object'))
#     st.table(data_reshaped.head())

    print('Shape of dataframe:', data_reshaped.shape)


# In[ ]:


# Apriori Algorithm
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

if(QuestionSB == "Question 7"):
    frequent_itemsets = apriori(data_reshaped.iloc[:,:70],min_support=0.4, use_colnames=True)
    st.header("Apriori Algorithm")
#     st.table(frequent_itemsets)
    st.write(frequent_itemsets)


# In[ ]:


if(QuestionSB == "Question 7"):
    # Association rule with metric as confidence
    rules=association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)
    rules.sort_values('confidence', ascending=False, inplace=True)

    st.header("Association rule with metric as confidence")
    st.write(rules)

    # Association Rule network visualisation
    import networkx as nx
    import matplotlib.pyplot as plt
    plt.figure(figsize=(40,10))
    G = nx.DiGraph()
    G.add_edges_from(
        [('A', 'B'), ('A', 'C'), ('D', 'B'), ('E', 'C'), ('E', 'F'),
         ('B', 'H'), ('B', 'G'), ('B', 'F'), ('C', 'G')])

    val_map = {'A': 1.0,
               'D': 0.5714285714285714,
               'H': 0.0}

    values = [val_map.get(node, 0.25) for node in G.nodes()]

    # Specify the edges you want here
    red_edges = [('A', 'C'), ('E', 'C')]
    edge_colours = ['black' if not edge in red_edges else 'red'
                    for edge in G.edges()]
    black_edges = [edge for edge in G.edges() if edge not in red_edges]

    # Need to create a layout when doing
    # separate calls to draw nodes and edges
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), 
                           node_color = values, node_size = 3500)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, edgelist=red_edges, edge_color='r', arrows=True)
    nx.draw_networkx_edges(G, pos, edgelist=black_edges, font_size=25, arrows=True)
    plt.show()

    st.header("Association Rule network visualisation")
    st.pyplot()


# In[ ]:


if(QuestionSB == "Question 7"):
    # Association rule visualisation
    def visualize_rules(rules, topk=5):
        import networkx as nx
        import matplotlib.pyplot as plt
        plt.figure(figsize=(40,10))
        G = nx.DiGraph()
        for rule in rules.iloc[:topk].itertuples():
            print(rule.antecedents)
            G.add_edges_from([(rule.antecedents,rule.consequents)])

        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), 
                            node_size = 500)
        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_edges(G, pos, arrows=True)
        plt.show()
        st.header("Association rule visualisation")
        st.pyplot()
        for rule in rules.iloc[:topk].itertuples():
            st.write(rule.antecedents)

    visualize_rules(rules)

