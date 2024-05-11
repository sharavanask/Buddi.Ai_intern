import numpy as np
import matplotlib.pyplot as plt

# Generating random y values with respect to x values
x = []
y = []
x1=[[1]*100,[]]
for i in range(-50, 50):
    x_val = i / 100
    x.append(x_val)
    x1[1].append(i)
    n = np.random.normal(0, 5)
    y_val = 2 * x_val - 3 + n
    y.append(y_val)
X1=np.transpose(x1)
Y1=np.transpose(y)
# Calculate coefficients using matrix multiplication/

b=np.matmul(np.linalg.inv(np.matmul(x1,X1)),np.matmul(x1,Y1))
B0=b[0]
B1=b[1]


# Gradient Descent processing
b0f = np.random.normal(0, 1)
b1f = np.random.normal(0, 1)

errorf = np.mean((y - (b0f + b1f*np.array(x)))**2)
lr = 0.01

error = errorf
b0 = b0f
b1 = b1f
epoch = 0

Epoch = [0]
# Epoch.append(0)
E = []
E.append(errorf)
Gb0=[]
Gb1=[]
out = False
while not out:
    y_pred = b0 + b1*np.array(x)
    db0=[]
    db1=[]
    # grad_b0 = -2*np.mean((y - y_pred))
    # grad_b1 = -2*np.mean((y - y_pred)*np.array(x))
    for i in range(len(x)):
        db0.append(-2*((y[i] - b0 + b1*x[i])))
        db1.append(-2*((y[i] - b0+b1*(x[i]))*x[i]))
    grad_b0=sum(db0)/len(db0)
    grad_b1=sum(db1)/len(db1)


    b0 -= lr * grad_b0
    b1 -= lr * grad_b1
    
    # new_error = np.mean((y - (b0 + b1*np.array(x)))**2)
    ne=[]
    for i in range(len(x)):
        ne.append((y[i] - (b0 + b1*(x[i])))**2)
    new_error = sum(ne)/len(ne)
    epoch += 1
    
    Epoch.append(epoch)
    E.append(new_error)
    Gb0.append(b0)
    Gb1.append(b1)
    if abs(error - new_error) < 10e-6:
        out = True
    else:
        error = new_error

print("Closed Form: b0:", B0, "b1:", B1, "error:", errorf)
print("Gradient Descent: b0:", b0, "b1:", b1, "error:", error, "epoch:", epoch)

# Plotting the error convergence
plt.figure(figsize=(8, 4))
plt.plot(Epoch, E)
plt.xlabel('NUmber of Epoch Cycles')
plt.ylabel('Mean Squared Error')
plt.title('Gradient Descent')
plt.figtext(0.5, 0.01, "In the above graph the MeanSquare Error is plotted with respect to number of Epoch cycle is executed", ha="center", fontsize=10, bbox={"facecolor":"brown", "alpha":0.5, "pad":5})
# plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(Gb0,Gb1,Epoch, cmap='viridis')

ax.set_xlabel('Beta1(B1)')
ax.set_ylabel('Beta2(B2)')
ax.set_zlabel('Epoc')
ax.set_title('Surface Plot')
plt.show()