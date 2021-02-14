import numpy as np
import math

def generator(sigma_current):
    return np.random.normal(sigma_current,0.5)

def PDF_gauss(di,muy,sigma):
    return math.exp(-(di-muy)**2/(2*sigma**2))*1/math.sqrt(2*math.pi*sigma**2)

def likelihood(D,sigma_new, sigma_current):
    likelihood_new = 0
    likelihood_current = 0
    muy = sum(D)/len(D)
    for d in D:
        likelihood_new += np.log(PDF_gauss(d,muy,sigma_new))
        likelihood_current += np.log(PDF_gauss(d,muy,sigma_current))

    return likelihood_new,likelihood_current

def prior(sigma):
    if sigma <= 0:
        return 0
    return 1

def accept(D, sigma_new, sigma_current):
    likelihood_new,likelihood_current = likelihood(D,sigma_new, sigma_current)
    likelihood_new = likelihood_new + np.log(prior(sigma_new))
    likelihood_current = likelihood_current + np.log(prior(sigma_current))
    if likelihood_new > likelihood_current:
        return True
    else:
        accept = np.random.uniform(0,1)
        return (np.exp(likelihood_new - likelihood_current) > accept)


def MH(interation, sigma_current, D):
    rejected = []
    accepted = []
    for i in range(interation):
        print("iteration:",i,"--> sigma:",sigma_current)
        sigma_new = generator(sigma_current)
        if accept(D,sigma_new,sigma_current):
            sigma_current = sigma_new
            accepted.append(sigma_new)
        else:
            rejected.append(sigma_new)


    return rejected, accepted