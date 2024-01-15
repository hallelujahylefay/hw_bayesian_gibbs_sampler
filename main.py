import pandas as pd
import numpy as np
from gibbs_ps import gibbs_per_block
from sklearn.linear_model import Lasso

df=pd.read_csv(path)

remove= ['sasdate','date','year','Prev_Indpro','Industial monthly growth rate','INDPRO']
regressors=[ i for i in df.columns if i not in remove]
Y=np.array(df['INDPRO'])
X=np.array(df[regressors])
U=np.ones(len(df))

lasso_reg = Lasso(alpha=0.1, fit_intercept=False)
lasso_reg.fit(np.c_[U, X], Y)
beta = lasso_reg.coef_[1:]
phi = lasso_reg.coef_[0]
y_pred = lasso_reg.predict(np.c_[U, X])
residuals = Y - y_pred
sigma2 = np.var(residuals)
z = np.where(beta != 0, True, False)
q = np.sum(z) / len(z)
z, beta, phi, sigma2, q



BURNIN_period = 1000
ITERATION = 5000

init=z, beta, phi, sigma2, q
res_gibbs = X, Y, gibbs_per_block(X, Y, U, init, ITERATION=ITERATION, BURNIN_period=BURNIN_period)
