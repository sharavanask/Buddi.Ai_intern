import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#creatinginput and output list ias x and y
x=[[1]*101,[],[],[],[]]
x1 =[]
y=[]
# Append x values to the respective columns in x matrix
for i in range(-50,51):
    i=i/10
    x1.append(i)
    x[1].append(i)
    x[2].append(i**2)
    x[3].append(i**3)
    x[4].append(i**4)
    #gettingnoise as random values
    n=np.random.normal(0,3)
    fx=((2*(i**4))-(3*(i**3))+(7*(i**2))-(23*i)+8+n)
    y.append(fx)
x_train, x_test, ytrain, y_test = train_test_split(x1, y, test_size=0.20, random_state=42)
# Transpose the  'x and 'y' list to prepare it for matrix operations
X=np.transpose(x)
Y=np.transpose(y)
# Calculate the coefficients 'b' using linear regression formula
b=np.dot(np.linalg.inv(np.dot(np.transpose(X),X)),np.dot(np.transpose(X),y))
y1=[]
y2=[]
y3=[]
y4=[]
#Calculating the output of each models 
for i in x1:
    f1=b[0] + b[1]*i 
    y1.append(f1)
    f2 = b[0] + b[1]*i + b[2] * (i**2)
    y2.append(f2)
    f3= b[0] + b[1]*i + b[2] * (i**2)  + b[3]*(i**3) 
    y3.append(f3)
    f4= b[0] + b[1]*i + b[2] * (i**2)  + b[3]*(i**3) + b[4]*(i**4)
    y4.append(f4)
#funtion to find legrange interpolates
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
#calling legrange funtion
yInte=lagrangeInterpolation(x1,y,x1)
#plottting different models and thier outputs plot
plt.figure(figsize=(8, 4))
plt.plot(x1,y1,label="linear")
plt.plot(x1,y2,label="quadratic")
plt.plot(x1,y3,label="cubic")
plt.plot(x1,y4,label="biquadratic")
plt.plot(x1,yInte,marker='^',label="legrange")
plt.xlabel('X')
plt.ylabel('Y=F(X)')
plt.title("Different models and its plot")
plt.figtext(0.5, 0.01, "Let us consider a biquadratic polynomial and 101 x values in the range(-5,5) and generated y values for different models such as linear,quadratic,cubic and biquadratic and their corresponding f(x) is plotted above", ha="center", fontsize=10, bbox={"facecolor":"brown", "alpha":0.5, "pad":5})
plt.legend()
#creating empty list to store trainning outputs
y1_train=[]
y2_train=[]
y3_train=[]
y4_train=[]
#finding training outout for each and evry models
for i in x_train:
    f1=b[0] + b[1]*i
    y1_train.append(f1)
    f2 = b[0] + b[1]*i + b[2] * (i**2)
    y2_train.append(f2)
    f3= b[0] + b[1]*i + b[2] * (i**2)  + b[3]*(i**3)
    y3_train.append(f3)
    f4= b[0] + b[1]*i + b[2] * (i**2)  + b[3]*(i**3) + b[4]*(i**4)
    y4_train.append(f4)
#creating empty lists to store trainnig errror of each of the models
e1train=[]
e2train=[]
e3train=[]
e4train=[]
#finding training error forf the different models 
for i in range(len(y1_train)):
    e1train.append((y1_train[i]-ytrain[i])**2)
    e2train.append((y2_train[i]-ytrain[i])**2)
    e3train.append((y3_train[i]-ytrain[i])**2)
    e4train.append((y4_train[i]-ytrain[i])**2)
#creating training error list
Etrain=[]
com=[1,2,3,4]
#appending each opf the trainnig error of each models
Etrain.append((sum(e1train))/len(y1_train))
Etrain.append((sum(e2train))/len(y2_train))
Etrain.append((sum(e3train))/len(y3_train))
Etrain.append((sum(e4train))/len(y4_train))
print(len(Etrain))
#creating empty list to store trainning outputs
y1test=[]
y2test=[]
y3test=[]
y4test=[]
#finding training error forf the different models 

for i in x_test:
    f1=b[0] + b[1]*i 
    y1test.append(f1)
    f2 = b[0] + b[1]*i + b[2] * (i**2) 
    y2test.append(f2)
    y3test.append(f3)
    f4= b[0] + b[1]*i + b[2] * (i**2)  + b[3]*(i**3) + b[4]*(i**4) 
    y4test.append(f4)
#creating empty lists to store trainnig errror of each of the models

e1test=[]
e2test=[]
e3test=[]
e4test=[]
#creating training error list
Etest=[]
#appending each opf the trainnig error of each models
for i in range(len(y1test)):
    e1test.append(((y1test[i] - y_test[i]))**2)
    e2test.append(((y2test[i] - y_test[i]))**2)
    e3test.append(((y3test[i] - y_test[i]))**2)
    e4test.append(((y4test[i] - y_test[i]))**2)
#appending mean error of eachh model to training error
Etest.append(sum(e1test)/len(y1test))
Etest.append(sum(e2test)/len(y2test))
Etest.append(sum(e3test)/len(y3test))
Etest.append(sum(e4test)/len(y4test))

print(y_test)
print(y1test)
#plotting  the Bias Variance tradeoff for different kinds of models and their errrors
plt.figure(figsize=(8, 4))
plt.plot(com,Etrain,label="bias")
plt.plot(com,Etest,label='variance')
plt.legend()
plt.show()
