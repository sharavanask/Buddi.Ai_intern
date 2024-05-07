import numpy as np
#function to draw sample from the distribution
def draw_sample(s, n):
    p=[]
    values=list(s.values())
    sumv=sum(values)
    #generating probabilistic values 
    for i in values:
        p.append(i/sumv)
    cdf=[]
    sample1=[]
    c=0
    keys=list(s.keys())
    #generating cumulative function
    for i in p:
        c+=i
        cdf.append(c)
    X=[]
    #generating random uniformly distributed values
    for i in range(n):
        x=np.random.uniform(0,1)
        X.append(x)
    #drawing samples
    for i in X:
        j = 0
        found = False
        while j < len(cdf) and not found:
            if i < cdf[j]:
                sample1.append(keys[j])
                found = True
            j += 1
    print("the distribution =",s)
    print("the probability values =",p)
    print("the cumulative values =",cdf)
    return sample1
s = {'Apple': 10, 'Banana': 6, 'Carrot': 4}
n = 10
#passing the dictionary values and n into random sampler functions
samples = draw_sample(s, n)
print(samples)
