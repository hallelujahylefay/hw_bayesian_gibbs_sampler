def vbar(X):
    raise NotImplementedError

def R2q(X, z, n):
    def posteriorR2q(q,R2,X,z,beta,sigma2):
        bz=np.dot(np.dot(beta,np.diag(z)),np.transpose(beta))
        sz=sum(z)
        return exp((-1/(2*sigma2))*(k*vbar(X)*q*((1-R2)/R2)*bz))*q**(3*sz/2)*(1-q)**(k-sz)*R2**(-sz/2)*(1-R2)**(sz/2)
    grid_q =[i/1000 for i in range(1,100)]+[i/100 for i in range(10,90)]+[i/1000 for i in range(900,1000)]
    grid_R2 =[i/1000 for i in range(1,100)]+[i/100 for i in range(10,90)]+[i/1000 for i in range(900,1000)]
    #initial values for q and R2
    q_=0.9
    R_=0.1
    
    q=[]
    R=[]
    
    for j in range(n):
        
        w_r=[posterior(i,q_,X,z) for i in grid_R2]
        s=sum(w_r)
        w_r=[i/s for i in w_r]
        cdf_r=list(np.cumsum(w_r))
        u=np.random.uniform(0,1)
        R_=grid_R2[cdf_r.index(min(n for n in cdf_r  if n>u))]
        R.append(R_)
        
        w_q=[posterior(R_,i,X,z) for i in grid_q]
        s=sum(w_q)
        w_q=[i/s for i in w_q]
        cdf_q=list(np.cumsum(w_q))
        v=np.random.uniform(0,1)
        q_=grid_q[cdf_q.index(min(n for n in cdf_q  if n>v))]
        q.append(q_)
    return list(zip(q,R))

def phi(Y, U, X, z, beta, r2, q, sigma2):
    #yvann
    raise NotImplementedError

def z(Y, U, X, phi, R2, q):
    #yvann
    raise NotImplementedError

def sigma2(Y, U, X, phi, R2, q, z):
    #rayane
    raise NotImplementedError

def betatilde(Y, U, X, phi, R2, q, sigma2, z):
    #rayane
    raise NotImplementedError
