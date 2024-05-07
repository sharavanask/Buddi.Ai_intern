#finding the epdilon value by finding the absolute difference between the original value and estimated valuie


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
x = [-3, -2, -1, 0, 1, 2, 3]
y = [7, 2, 0,0,0, 2, 7]
b1 = [-3, -2, -1, 0, 1, 2, 3]
b2 = [-3, -2, -1, 0, 1, 2, 3]
y1 = []
e = []
B1 = []
B2 = []

for l in x:
    for i in b1:
        for j in b2:
            y1.append((l * i) + (j * (l * l)))

for i in y1:
  c=0
  for j in y:
    c+=abs(i - j)
  e.append(c)
E=[]
B1=[]
B2=[]
for i in b1:
  for j in b2:
    B1.append(i)
    B2.append(j)
for i in range(0,len(e),7):
  a=list(e[i:i+7])
  print(a)
  E.append(sum(a))
print(B1)
print(B2)
print(E)
plot = plt.figure()
ax = plot.add_subplot(111, projection='3d')
ax.plot_trisurf(B1,B2,E, cmap='viridis', edgecolor='none')
ax.set_xlabel('Beta1')
ax.set_ylabel("Beta2")
ax.set_zlabel('Epsilon')
ax.set_title('Surface Plot')
plt.show()
