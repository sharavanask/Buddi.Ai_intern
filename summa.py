import numpy as np
import matplotlib.pyplot as plt

X = [[1] * 101, [], [], [], []]
x_values = []
y_values = []

# Generate x and y values
for i in range(-50, 51):
    x_val = i / 10
    x_values.append(x_val)
    X[1].append(x_val)
    X[2].append(x_val ** 2)
    X[3].append(x_val ** 3)
    X[4].append(x_val ** 4)
    noise = np.random.normal(0, 3)
    f_x = 2 * (x_val ** 4) - 3 * (x_val ** 3) + 7 * (x_val ** 2) - 23 * x_val + 8 + noise
    y_values.append(f_x)

X = np.transpose(X)
y_values = np.transpose(y_values)

# Compute coefficients using linear regression
coefficients = np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.dot(np.transpose(X), y_values))

# Generate y values for different models
y_linear = coefficients[0] + coefficients[1] * np.array(x_values)
y_quadratic = coefficients[0] + coefficients[1] * np.array(x_values) + coefficients[2] * (np.array(x_values) ** 2)
y_cubic = coefficients[0] + coefficients[1] * np.array(x_values) + coefficients[2] * (np.array(x_values) ** 2) + \
          coefficients[3] * (np.array(x_values) ** 3)
y_biquadratic = coefficients[0] + coefficients[1] * np.array(x_values) + coefficients[2] * (np.array(x_values) ** 2) + \
                coefficients[3] * (np.array(x_values) ** 3) + coefficients[4] * (np.array(x_values) ** 4)


def lagrange_interpolation(x, y, x_interp):
    n = len(x)
    m = len(x_interp)
    y_interp = np.zeros(m)
    
    for j in range(m):
        p = 0
        for i in range(n):
            L = 1
            for k in range(n):
                if k != i:
                    L *= (x_interp[j] - x[k]) / (x[i] - x[k])
            p += y[i] * L
        y_interp[j] = p
    return y_interp


y_interpolated = lagrange_interpolation(x_values, y_values, x_values)

# Plotting
plt.figure(figsize=(8, 4))
plt.plot(x_values, y_linear, label="Linear")
plt.plot(x_values, y_quadratic, label="Quadratic")
plt.plot(x_values, y_cubic, label="Cubic")
plt.plot(x_values, y_biquadratic, label="Biquadratic")
plt.plot(x_values, y_interpolated, label="Lagrange Interpolation")
plt.xlabel('X')
plt.ylabel('Y = F(X)')
plt.title("Different Models and Their Plots")
plt.figtext(0.5, 0.01, "Consider a biquadratic polynomial and 101 x values in the range (-5, 5) with generated y values for different models such as linear, quadratic, cubic, and biquadratic, and their corresponding f(x) is plotted above.", ha="center", fontsize=10, bbox={"facecolor": "brown", "alpha": 0.5, "pad": 5})
plt.legend()

# Bias-Variance Tradeoff
x_train = x_values[:81]
x_test = x_values[81:]
y_train_linear = coefficients[0] + coefficients[1] * np.array(x_train)
y_train_quadratic = coefficients[0] + coefficients[1] * np.array(x_train) + coefficients[2] * (np.array(x_train) ** 2)
y_train_cubic = coefficients[0] + coefficients[1] * np.array(x_train) + coefficients[2] * (np.array(x_train) ** 2) + \
               coefficients[3] * (np.array(x_train) ** 3)
y_train_biquadratic = coefficients[0] + coefficients[1] * np.array(x_train) + coefficients[2] * (np.array(x_train) ** 2) + \
                     coefficients[3] * (np.array(x_train) ** 3) + coefficients[4] * (np.array(x_train) ** 4)

y_test_linear = coefficients[0] + coefficients[1] * np.array(x_test)
y_test_quadratic = coefficients[0] + coefficients[1] * np.array(x_test) + coefficients[2] * (np.array(x_test) ** 2)
y_test_cubic = coefficients[0] + coefficients[1] * np.array(x_test) + coefficients[2] * (np.array(x_test) ** 2) + \
               coefficients[3] * (np.array(x_test) ** 3)
y_test_biquadratic = coefficients[0] + coefficients[1] * np.array(x_test) + coefficients[2] * (np.array(x_test) ** 2) + \
                     coefficients[3] * (np.array(x_test) ** 3) + coefficients[4] * (np.array(x_test) ** 4)

errors_train = [sum(abs(y_train_linear - y_values[:81])),
                sum(abs(y_train_quadratic - y_values[:81])),
                sum(abs(y_train_cubic - y_values[:81])),
                sum(abs(y_train_biquadratic - y_values[:81]))]

errors_test = [sum(abs(y_test_linear - y_values[81:])),
               sum(abs(y_test_quadratic - y_values[81:])),
               sum(abs(y_test_cubic - y_values[81:])),
               sum(abs(y_test_biquadratic - y_values[81:]))]

plt.figure(figsize=(8, 4))
plt.plot([1, 2, 3, 4], errors_test, label="Variance")
plt.plot([1, 2, 3, 4], errors_train, label="Bias")
plt.xlabel('Complexity')
plt.ylabel('Error')
plt.title("Bias-Variance Tradeoff")
plt.figtext(0.5, 0.01, "The graph shows the bias-variance tradeoff for linear, quadratic, cubic, and biquadratic models.", ha="center", fontsize=10, bbox={"facecolor": "brown", "alpha": 0.5, "pad": 5})
plt.legend()
plt.show()
