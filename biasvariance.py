import numpy as np
import matplotlib.pyplot as plt

x=[[1]*101,[],[],[],[]]
x1 =[]
y=[]
for i in range(-50,51):
    i=i/10
    x1.append(i)
    x.append
    x[1].append(i)
    x[2].append(i**2)
    x[3].append(i**3)
    x[4].append(i**4)
    n=np.random.normal(0,3)
    fx=((2*(i**4))-(3*(i**3))+(7*(i**2))-(23*i)+8+n)
    y.append(fx)
X=np.transpose(x)
Y=np.transpose(y)
b=np.dot(np.linalg.inv(np.dot(np.transpose(X),X)),np.dot(np.transpose(X),y))
y1=[]
y2=[]
y3=[]
y4=[]
for i in x1:
    f1=b[0] + b[1]*i
    y1.append(f1)
    f2 = b[0] + b[1]*i + b[2] * (i**2)
    y2.append(f2)
    f3= b[0] + b[1]*i + b[2] * (i**2)  + b[3]*(i**3)
    y3.append(f3)
    f4= b[0] + b[1]*i + b[2] * (i**2)  + b[3]*(i**3) + b[4]*(i**4)
    y4.append(f4)


def lagrangeInterpolation(x, y, xInterp):
    n = len(x)
    m = len(xInterp)
    yInterp = np.zeros(m)
    
    for j in range(m):
        p = 0
        for i in range(n):
            L = 1
            for k in range(n):
                if k != i:
                    L *= (xInterp[j] - x[k]) / (x[i] - x[k])
            p += y[i] * L
        yInterp[j] = p
    return yInterp
yInte=lagrangeInterpolation(x1,y,x1)
plt.figure(figsize=(8, 4))
plt.plot(x1,y1,label="linear")
plt.plot(x1,y2,label="quadratic")
plt.plot(x1,y3,label="cubic")
plt.plot(x1,y4,label="biquadratic")
plt.plot(x1,yInte,label="legrange")
plt.xlabel('X')
plt.ylabel('Y=F(X)')
plt.title("Different models and its plot")
plt.figtext(0.5, 0.01, "Let us consider a biquadratic polynomial and 101 x values in the range(-5,5) and generated y values for different models such as linear,quadratic,cubic and biquadratic and their corresponding f(x) is plotted above", ha="center", fontsize=10, bbox={"facecolor":"brown", "alpha":0.5, "pad":5})
plt.legend()
#plt.show()

X1=x1[:81]
X2=x1[81:]
y1_train=[]
y2_train=[]
y3_train=[]
y4_train=[]
for i in X1:
    f1=b[0] + b[1]*i
    y1_train.append(f1)
    f2 = b[0] + b[1]*i + b[2] * (i**2)
    y2_train.append(f2)
    f3= b[0] + b[1]*i + b[2] * (i**2)  + b[3]*(i**3)
    y3_train.append(f3)
    f4= b[0] + b[1]*i + b[2] * (i**2)  + b[3]*(i**3) + b[4]*(i**4)
    y4_train.append(f4)
# print(y1_train)
# print(y2_train)
# print(y3_train)
# print(y4_train)
e1train=[]
e2train=[]
e3train=[]
e4train=[]
com=[1,2,3,4]
for i in range(81):
    e1train.append(abs(y1_train[i]-y[i]))
    e2train.append(abs(y2_train[i]-y[i]))
    e3train.append(abs(y3_train[i]-y[i]))
    e4train.append(abs(y4_train[i]-y[i]))
Etrain=[]
Etrain.append(sum(e1train))
Etrain.append(sum(e2train))
Etrain.append(sum(e3train))
Etrain.append(sum(e4train))

ytest1=[]
ytest2=[]
ytest3=[]
ytest4=[]
Yorig=Y[81:]
e1test=[]
e2test=[]
e3test=[]
e4test=[]
Etest=[]
for i in X2:
    f1=b[0] + b[1]*i
    ytest1.append(f1)
    f2 = b[0] + b[1]*i + b[2] * (i**2)
    ytest2.append(f2)
    f3= b[0] + b[1]*i + b[2] * (i**2)  + b[3]*(i**3)
    ytest3.append(f3)
    f4= b[0] + b[1]*i + b[2] * (i**2)  + b[3]*(i**3) + b[4]*(i**4)
    ytest4.append(f4)
# fxt=((2*(i**4))-(3*(i**3))+(7*(i**2))-(23*i)+8+n)
for i in range(len(ytest1)):
    e1test.append(abs(ytest1[i]-Yorig[i]))
    e2test.append(abs(ytest2[i]-Yorig[i]))
    e3test.append(abs(ytest3[i]-Yorig[i]))
    e4test.append(abs(ytest4[i]-Yorig[i]))
Etest.append(sum(e1test))
Etest.append(sum(e2test))
Etest.append(sum(e3test))
Etest.append(sum(e4test))
plt.figure(figsize=(8, 4))
plt.plot(com,Etest,label="Variance")
plt.plot(com,Etrain,label="Bias")
plt.xlabel('Complexity')
plt.ylabel('Error')
plt.title("Bias Variance TradeOff")
plt.figtext(0.5, 0.01, "The above graph shows the bias variance tradeoff for models such as linear,quadratic,cubic and biquadratic models", ha="center", fontsize=10, bbox={"facecolor":"brown", "alpha":0.5, "pad":5})
plt.legend()
plt.show()