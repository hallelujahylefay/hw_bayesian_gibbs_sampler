from scipy.integrate import quad
import numpy as np

def R2q(X, z):
    def joint_pdf(q_v, R2_v, X, z, beta_v, sigma2_v):
        bz = beta_v @ np.diag(z) @ beta_v.T
        sz_v = sz(z)
        return np.exp((-1 / (2 * sigma2_v)) * (k * vbar(X) * q_v * ((1 - R2_v) / R2_v) * bz)) * q_v ** (
                3 / 2 * sz_v + a - 1) * (
                       1 - q_v) ** (k - sz_v + b - 1) * R2_v ** (A - 1 - sz_v / 2) * (1 - R2_v) ** (sz_v / 2 + B - 1)

    grid_q = [i / 1000 for i in range(1, 100)] + [i / 100 for i in range(10, 90)] + [i / 1000 for i in range(900, 1000)]
    grid_R2 = [i / 1000 for i in range(1, 100)] + [i / 100 for i in range(10, 90)] + [i / 1000 for i in
                                                                                      range(900, 1000)]
    
    
    def univariate_pdf(q_v):
        # marginal of q, integrate joint posterior 
        def exp(R2_v):
            return joint_pdf(q_v,R2_v,X,z,beta,sigma2)
        return  quad(exp, 0, 1)

    def conditional_pdf(q_v,R2_v):
        # distribution of q conditional on R2, proportionate to the joint posterior
        bz = beta_v @ np.diag(z) @ beta_v.T
        sz_v = sz(z)
        return np.exp((-1 / (2 * sigma2_v)) * (k * vbar(X) * q_v * ((1 - R2_v) / R2_v) * bz)) * R2_v ** (
            A - 1 - sz_v / 2) * (1 - R2_v) ** (sz_v / 2 + B - 1) 
        

    # initial values for q and R2
    q_ = grid_q[np.random.random_integers(0,len(grid_q))]
    R_ = grid_R2[np.random.random_integers(0,len(grid_R2))]

    def invCDF(pdf, grid, u):
        weights = pdf(grid)
        normalize_constant = np.sum(weights)
        weights /= normalize_constant
        cdf = np.cumsum(weights)
        return grid[np.argmax(cdf > u)]

    def sampleqR(n):
        u=np.random.uniform(0,1)
        q_=invCDF(univariate_pdf(),grid_q,u)
        def pdf_q(R_v):
            return conditional_pdf(q_,R_v)
        R_=invCDF(pdf_q(),grid_R2,u)
        return (q_,R_)
        
        

    return sampleqR  # function that will be looped over to generate samples of (q, R) given X z