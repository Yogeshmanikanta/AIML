import numpy as np 
import matplotlib.pyplot as plt

def f(x):
    return x**2

# derivative 

def gradient(x):
    return 2*x

# starting point
x=10

#learning point
lr=0.1

# Store path
x_history =[]

for i in range(20):
    x_history.append(x)

    grad=gradient(x)

    x=x - lr*grad

    print(f"Itreatin {i+1}: x= {x}")


curve_x=np.linspace(-10,10,200)
curve_y=f(curve_x)

path_y = [f(x) for x in x_history]

plt.plot(curve_x, curve_y,label="y = x²")
plt.scatter(x_history,path_y,label="Gradient Descent Path")

plt.legend()
plt.show()