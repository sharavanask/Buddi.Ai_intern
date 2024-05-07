import numpy as np
import matplotlib.pyplot as plt
m = 0
sd = 1
x = np.linspace(-5*sd, 5*sd)
print(x)
f = (1 / (sd * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - m) / sd) ** 2)
plt.plot(x, f, color='blue', label='Normal Distribution')
m = 0
sd1 = 2
x = np.linspace(-5*sd, 5*sd)
f = (1 / (sd1 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - m) / sd1) ** 2)


plt.plot
plt.plot(x, f, color='yellow', label='Normal Distribution')
plt.title("Normal Distribution (Bell Curve)")
plt.xlabel("Value")
plt.ylabel("Probability Density")
plt.legend()
plt.figtext(0.5, 0.01, "This graph represents a standard normal distribution curve with same mean and standard deviation=1.", ha="center", fontsize=10, bbox={"facecolor":"lightgrey", "alpha":0.5, "pad":5})
plt.show()



