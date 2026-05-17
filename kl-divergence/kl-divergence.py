import numpy as np

def kl_divergence(p, q, eps=1e-12):
    """
    Compute KL Divergence D_KL(P || Q).
    """
    q = np.clip(q,eps,q)
    L = np.sum(np.dot(p,np.log(p/q)))
    return L