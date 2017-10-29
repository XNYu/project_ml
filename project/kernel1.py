# Import the neccessary modules for data manipulation and visual representation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplot
import seaborn as sns
# %matplotlib inline
import scipy.stats as stats
from sklearn import preprocessing

df = pd.read_csv('HR_comma_sep.csv', index_col=None)
# Check to see if there are any missing values in our data set
df.isnull().any()
# Renaming certain columns for better readability
df = df.rename(columns={'satisfaction_level': 'satisfaction',
                        'last_evaluation': 'evaluation',
                        'number_project': 'projectCount',
                        'average_montly_hours': 'averageMonthlyHours',
                        'time_spend_company': 'yearsAtCompany',
                        'Work_accident': 'workAccident',
                        'promotion_last_5years': 'promotion',
                        'sales' : 'department',
                        'left' : 'turnover'
                        })
# Move the reponse variable "turnover" to the front of the table
front = df['turnover']
df.drop(labels=['turnover'], axis=1,inplace = True)
df.insert(0, 'turnover', front)
df.head()

# =======================================
#basic evaluation
#========================================

# Looks like about 76% of employees stayed and 24% of employees left.
# NOTE: When performing cross validation, its important to maintain this turnover ratio
# turnover_rate = df.turnover.value_counts() / len(df)
# print(turnover_rate)
#
# print(df.describe())
#
# # Overview of summary (Turnover V.S. Non-turnover)
# turnover_Summary = df.groupby('turnover')
# print(turnover_Summary.mean())

#=========================================
# Correlation Matrix
#=========================================
# corr = df.corr()
# corr = (corr)
# f, ax = plt.subplots()
#
# # positive(+) correlation or negative(-)
# sns.heatmap(corr,
#             xticklabels=corr.columns.values,
#             yticklabels=corr.columns.values, ax = ax)
# ax.set_xticklabels(ax.get_xticklabels(),fontsize = 7, rotation = 30)
# ax.set_yticklabels(ax.get_yticklabels(),fontsize=7, rotation=30)
# plt.show()
# print(corr)
#
# # Let's compare the means of our employee turnover satisfaction against the employee population satisfaction
# #emp_population = df['satisfaction'].mean()
# emp_population = df['satisfaction'][df['turnover'] == 0].mean()
# emp_turnover_satisfaction = df['satisfaction'][df['turnover']==1].mean()
#
# print( 'The mean satisfaction for the employee population with no turnover is: ' + str(emp_population))
# print( 'The mean satisfaction for employees that had a turnover is: ' + str(emp_turnover_satisfaction) )
#
# # 2-sample T-Test 如果p值大于5%，就接受原假设，否则不接受原假设
# statistic, p = stats.ttest_ind(df[df['turnover']==1]['satisfaction'], df[df['turnover']==0]['satisfaction'])
# if p > 0.05:
#     print('p value = '+str(p) + 'accept the null hypothesis')
# else:
#     print('p value = ' + str(p) + 'reject the null hypothesis')

#=========================================
# Distribution
#=========================================

# # Set up the matplotlib figure
# f, axes = plt.subplots(ncols=3, figsize=(15, 6))
#
# # Graph Employee Satisfaction
# sns.distplot(df.satisfaction, kde=False, color="g", ax=axes[0]).set_title('Employee Satisfaction Distribution')
# axes[0].set_ylabel('Employee Count')
#
# # Graph Employee Evaluation
# sns.distplot(df.evaluation, kde=False, color="r", ax=axes[1]).set_title('Employee Evaluation Distribution')
# axes[1].set_ylabel('Employee Count')
#
# # Graph Employee Average Monthly Hours
# sns.distplot(df.averageMonthlyHours, kde=False, color="b", ax=axes[2]).set_title('Employee Average Monthly Hours Distribution')
# axes[2].set_ylabel('Employee Count')
#
# avh_scale = preprocessing.scale(df['averageMonthlyHours'])
# # avh_scale = preprocessing.minmax_scale(feature_range=(0,1), X = df['averageMonthlyHours'])
# # plt.show()
#
# #put three together
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.hist(df.satisfaction, bins=np.arange(0, 1, 0.025), ls='dashed', alpha = 0.3, lw=3, color= 'g', label = 'Satisfaction')
# ax.hist(df.evaluation, bins=np.arange(0, 1, 0.025), ls='dotted', alpha = 0.3, lw=3, color= 'r', label = 'Evaluation')
# ax.hist(avh_scale, bins=np.arange(0, 1, 0.025), alpha = 0.3, lw=3, color= 'b', label = 'Average Monthly Hours')
# ax.set_ylabel('Employee Count')
#
# # ax.set_xlim(-0.5, 1.5)
# # ax.set_ylim(0, 7)
# plt.show()

