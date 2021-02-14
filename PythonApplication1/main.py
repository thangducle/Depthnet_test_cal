from MCMC import *

### sample generator
mod1=lambda t:np.random.normal(10,3,t)

#Form a population of 30,000 individual, with average=10 and scale=3
population = mod1(30000)
#Assume we are only able to observe 1,000 of these individuals.
observation = population[np.random.randint(0, 30000, 1000)]

rejected, accepted = MH(1000, 0.1, observation)
print("avg accept:",sum(accepted)/len(accepted))
print("last accept:", accepted[-1])
print("avg reject:",sum(rejected)/len(rejected))
print("last reject:",rejected[-1])
