import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# defining X, Y and U
df = pd.read_csv(r'/Users/hamzaab/Desktop/cours/3A/Bayesian stats/Project/current.csv')
df=df.drop(0).reset_index( drop = True )

int_features = df.dtypes.loc[ df.dtypes == 'int64'].index
df[ int_features ] = df[ int_features ].astype ('float')
df['sasdate'] = pd.to_datetime( df['sasdate'])

df['INDPROnext'] = df.loc[1:, 'INDPRO'].reset_index( drop =True )
df=df.loc[df['sasdate'] >= '1993 -01 -01']

rows_na = df [ df.isna().any ( axis =1)]
df = df.loc[~ df.index.isin( rows_na.index )]
# Separate features and variables to predict
X = df.drop( columns =['sasdate', 'INDPROnext']). to_numpy ()
Y = df['INDPROnext']. to_numpy ()
U = np.ones(X. shape [0]) # Intercept

# Standardize data
scaler = StandardScaler ()
X = scaler.fit_transform (X)

# Lasso regression to initialize the parameters beta, z, phi, q and sigma2

from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.1, fit_intercept=False)
lasso_reg.fit(np.c_[U, X], Y)
beta_v = lasso_reg.coef_[1:]
phi_v = lasso_reg.coef_[0]
y_pred = lasso_reg.predict(np.c_[U, X])
residuals = Y - y_pred
sigma2_v = np.var(residuals)
z_v = np.where(beta_v != 0, True, False)
q_v = np.sum(z_v) / len(z_v)


# gibbs sampler
from gibbs_sampling import *
init = z_v, beta_v, phi_v, sigma2_v, q_v
iter=50000
burn_int=5000
res=gibbs_sampler(X, Y, U, init, iter, burn_int)


with open('FREQ_MDQ_2.pickle', 'wb') as handle:
    pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)

# plot q distribution

q=[res[0][i][1] for i in range(len(res[0]))]
plt.hist(q, bins=70)
plt.xlim([0,1])
plt.title('Distribution of q, median='+str(np.median(q)))

# plot kde q distribution 
import seaborn as sns
ax=sns.kdeplot(q,bw_adjust=1,kernel='epanechnikov', color='#1f77b4' )
l1 = ax.lines[0]
x1 = l1.get_xydata()[:,0]
y1 = l1.get_xydata()[:,1]
plt.fill_between(x1,y1, alpha=0.5, color='#1f77b4')
plt.title('Distribution of q'))
plt.xlim([0,1])
plt.show()
