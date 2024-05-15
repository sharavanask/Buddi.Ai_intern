import numpy as np
import matplotlib.pyplot as plt



def indicator(y):
    I=[]
    for i in y: 
        if i =='G':
            I.append(0)
        else:
            I.append(1)
    return I


def closedform(x,y):
    x1=[[1]*9,[]]
    for i in x:
        x1[1].append(i)
    X1=np.transpose(x1)
    # b=np.matmul(np.linalg.inv(np.matmul(x1,X1)),np.matmul(x1,y))
    b=np.matmul(np.linalg.inv(np.matmul(np.transpose(X1),X1)),np.matmul(np.transpose(X1),y))

    return b

def linear_regressor(x,Iy,beta):
    linY=[]
    b0=beta[0]
    b1=beta[1]
    for i in x:
        fx=b0+(b1*i)
        s=1/(1+((np.e)**-fx))
        linY.append(s)
    return linY

def classifier(xi,b0,b1):
    fx=b0+(b1*xi)
    if fx > 0.55:
        return 'B'
    else:
        return 'G'



x=[-4,-3,-2,-1,0,1,2,3,4]
y=['G','G','G','G','B','B','B','B','B']
Iy= indicator(y)
print(Iy)
beta=closedform(x,Iy)
Y=linear_regressor(x,Iy,beta)
b0=beta[0]
b1=beta[1]
Yp=[]
for i in x:
    Yp.append((-1/b0)*i + 0.55)
classi=classifier(2,b0,b1)
print(classi)
plt.scatter(x[:4],Iy[:4],color="yellow")
plt.scatter(x[4:],Iy[4:],color="purple")
plt.plot(x,Y)
plt.plot(x,Yp)
plt.plot()
plt.show()