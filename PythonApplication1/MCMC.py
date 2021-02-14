import numpy as np
import math

def generator(sigma_current):
    return np.random.normal(sigma_current,0.5)

def PDF_gauss(di,muy,sigma):
    return math.exp(-(di-muy)**2)/(2*sigma**2)*1/math.sqrt(2*math.pi*sigma**2)

def likelihood(D,sigma_new, sigma_current):
    likelihood_new = 1
    likelihood_current = 1
    muy = sum(D)/len(D)
    for d in D:
        likelihood_new *= PDF_gauss(d,muy,sigma_new)
        likelihood_current *= PDF_gauss(d,muy,sigma_current)

    return likelihood_new,likelihood_current

def prior(sigma):
    if simag < 0:
        return 0
    return 1

def accept(D, sigma_new, sigma_current):
    if prior(sigma_new) == 1:
        likelihood_new,likelihood_current = likelihood(D,sigma_new, sigma_current)
        rate = likelihood_new/likelihood_current
        if rate > 1:
            return True
        else:
            accept = np.random.uniform(0,1)
            return (np.exp(rate) > accept)


def MH(interation, sigma_current):
    rejected = []
    accepted = []
    for i in range(interation):
        sigma_new = generator(sigma_current)
        if accept(D,sigma_new,sigma_current):
            sigma_current = sigma_new
            accepted.append(sigma_new)
        else:
            rejected.append(sigma_new)


### sample generator
mod1=lambda t:np.random.normal(10,3,t)

#Form a population of 30,000 individual, with average=10 and scale=3
population = mod1(30000)
#Assume we are only able to observe 1,000 of these individuals.
observation = population[np.random.randint(0, 30000, 1000)]

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1,1,1)
ax.hist( observation,bins=35 ,)
ax.set_xlabel("Value")
ax.set_ylabel("Frequency")
ax.set_title("Figure 1: Distribution of 1000 observations sampled from a population of 30,000 with mu=10, sigma=3")
mu_obs=observation.mean()
mu_obs