import numpy as np
s=[10,6,4]
p=[]
n=10
cp=[]
sus=sum(s)
for i in s:
    p.append(i/sus)
cpi=0
for i in p:
    cpi+=i
    cp.append(cpi)
def getsample(x):
    for i in range(len(cp)):
        if x<cp[i]:
            return cp[i]
sample=[]
for i in range(n):
    x=np.random.uniform(0,1)
    #x=0.7
    sam=getsample(x)
    sample.append(sam)
sample1=[]
for i in sample:
    sample1.append(s[cp.index(i)])
sample2=[]
for i in sample1:
    if i ==6:
        sample2.append('Apple')
    elif i ==10:
        sample2.append('Banana')
    elif i ==4:
        sample2.append('Carrot')
print(p)
print(cp)        
print(sample2)
