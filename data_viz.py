# -*- coding: utf-8 -*-


import pickle 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor 
from statsmodels.tools.tools import add_constant


#%%
with open(r"FRED_MD.pickle","rb") as file:   
    res = pickle.load(file)

#%%
ax = plt.subplot()
q = [res[0][i][1] for i in range(len(res[0]))]
plt.hist(q, bins=69)
plt.xlim([0,1])
med = round(np.median(q),5)
plt.axvline(x=np.median(q),color='red',label='median')
ax.legend(loc='upper right')
plt.title(f'Distribution of q, median={med}')

#%%
path = r"C:\Users\rayan\Documents\GitHub\hw_bayesian_gibbs_sampler_backup\data_3.csv"
df=pd.read_csv(path,index_col=0)
df.reset_index()

remove= ['sasdate','date','year','Prev_Indpro','Industial monthly growth rate','INDPRO']
regressors=[ i for i in df.columns if i not in remove]
X=df[regressors]

#%%
corr = X.corr()
high_corr_per_var=pd.Series(index=corr.columns,dtype=np.float64)

for var_index in range(len(corr)):
    corr.iloc[var_index,var_index]=0 #Not interested by the correlation between a variable and itself.
    neg_corr = (corr.iloc[var_index]<=-0.9).sum()
    pos_corr = (corr.iloc[var_index]>=0.9).sum()
    high_corr_per_var.iloc[var_index] = neg_corr + pos_corr

xlab=[str(i) for i in range(len(high_corr_per_var))]
ax=plt.subplot()
hcpv = high_corr_per_var.sort_values().reset_index(drop=True)
hcpv.plot(style='.',title = "Number of highly correlated variables for each feature",xticks=[],xlabel="Features",ylabel="Number of highly correlated variables" )
plt.axhline(y=hcpv.mean(),color='r',label='mean = 13')
plt.axhline(y=hcpv.quantile(q=0.75),color='green',label='first and third quartiles')
plt.axhline(y=hcpv.quantile(q=0.25),color='green')

legend = ax.legend(loc='upper left')

#%%
df['INDPROnext'] = df.loc[1:, 'INDPRO'].reset_index( drop =True )

features = list(df.drop( columns =['sasdate', 'INDPROnext']).columns)
number_inclusions = list(res[1])
#little dataframe with name of the feature and the number of time it was included 
df_inc=pd.DataFrame(list(zip(features,number_inclusions)),columns=['Regressors','Number of times included in the model'])
df_inc['Frequency of inclusion']=df_inc['Number of times included in the model']/27000

fig , ax = plt.subplots()
incl=df_inc.sort_values(by='Frequency of inclusion',ascending=False)['Frequency of inclusion']

ax=plt.subplot()
incl.reset_index(drop=True).plot(kind='bar',xlabel='Features',ylabel='Probability of inclusion',xticks=[],label='')
plt.axhline(y=incl.mean(),color='r',label='mean of q')
plt.axhline(y=incl.quantile(q=0.75),color='green',label='first and third quartiles of q')
plt.axhline(y=incl.quantile(q=0.25),color='green')

plt.title("Frequency of inclusion for each feature ")
legend = ax.legend(loc='upper right')