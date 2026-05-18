'''
Visualize Gradient Descent in Code

Plot f(x) = x² using matplotlib — a simple parabola
Start at x = 10. Apply gradient descent manually: derivative of x² is 2x, so update x = x − 0.1 * 2x
Run this for 20 iterations, print x each time — watch it move toward 0
Try learning rate = 0.01, then 0.9, then 1.1 — observe what happens in each case
Plot the path of x on the parabola — see it descend visually

'''

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