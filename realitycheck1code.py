
import matplotlib.pyplot as plt
import numpy as np

L1 = 2 
L2 = np.sqrt(2)
L3 = np.sqrt(2)
gamma = np.pi /2
x1 = 4 
x2 = 0
y2 = 4 
p1 = np.sqrt(5)
p2 = p1
p3 = p1
# Question 1 
def f(theta):
    A2 = L3*np.cos(theta)-x1 
    B2= L3*np.sin(theta)
    A3 = L2*np.cos(theta + gamma) - x2
    B3 = L2*np.sin(theta+ gamma) -y2
    D = 2 * (A2* B3 - B2*A3)
    N1 = B3*(p2**2-p1**2-A2**2-B2**2)-B2*(p3**2-p1**2-A3**2-B3**2)
    N2 = -A3*(p2**2-p1**2-A2**2-B2**2)+A2*(p3**2-p1**2-A3**2-B3**2)
    return N1**2+N2**2-p1**2*D**2


theta = np.pi/4 
x_array = np.linspace(-np.pi, np.pi,400)
plt.plot(x_array, f(x_array))
plt.grid()
plt.show()


#def triangle(p1, p2, gamma, theta):
#
#    A2 = L3*np.cos(theta)-x1 
#    B2= L3*np.sin(theta)
#    A3 = L2*np.cos(theta + gamma) - x2
#    B3 = L2*np.sin(theta+ gamma) -y2
#    D = 2 * (A2* B3 - B2*A3)
#    N1 = B3*(p2**2-p1**2-A2**2-B2)-B2*(p3**2-p1**2-A3**2-B3**2)
#    N2 = -A3*(p2**2-p1**2-A2**2-B2)+A2*(p3**2-p1**2-A3**2-B3**2)
#
#    x = N1 / D
#    y = N2 / D
#
#
#    answ = [x,y, theta]
#    return answ
#

def plot_triangle(point1, point2, point3, x1, x2 , y2):
    
    
    # Prepare x and y coordinates
    x = [point1[0], point2[0], point3[0], point1[0]]  # Closing the triangle
    y = [point1[1], point2[1], point3[1], point1[1]]  # Closing the triangle
    strut1x = [0, point1[0]]
    strut1y= [0, point1[1]]
    strut2x = [0, point2[0]]
    strut2y = [x1,point2[1]]
    strut3x = [x2, point3[0]]
    strut3y = [y2, point3[1]]
        
    # Create the plot
    plt.figure()
    
    plt.plot(x, y, linestyle='-', color='red')  
    plt.plot(strut1x, strut1y, linestyle='-', color='red') 
    plt.plot(strut2x, strut2y, linestyle='-', color='red')  
    plt.plot(strut3x, strut3y, linestyle='-', color='red')  


# Scatter with small open circles
    plt.scatter(x, y, marker='o', edgecolor='blue', s=50)
    plt.scatter(strut1x, strut1y, marker='o', edgecolor='blue', s=50 )
    plt.scatter(strut2x, strut2y, marker='o', edgecolor='blue', s=50 )
    plt.scatter(strut3x, strut3y, marker='o', edgecolor='blue', s=50 )


    # Set aspect ratio and limits
    plt.axis('equal')
    plt.grid()

  
#testing my triangle function
    
plot_triangle((1,2),(2,3),(2,1), 4,4,0)
plot_triangle((2,1),(1,2),(3,2), 4,4,0)

from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt

import plotly_express as px

L1 = 3
L2 = 3 * np.sqrt(2)
L3 = 3
gamma = np.pi /4
x1 = 5 
x2 = 0
y2 = 6
p1 = 5
p2 = 5
p3 = 3

# Define the function f(θ) from earlier
def f(theta):
    A2 = L3 * np.cos(theta) - x1 
    B2 = L3 * np.sin(theta)
    A3 = L2 * np.cos(theta + gamma) - x2
    B3 = L2 * np.sin(theta + gamma) - y2
    D = 2 * (A2 * B3 - B2 * A3)
    N1 = B3 * (p2**2 - p1**2 - A2**2 - B2**2) - B2 * (p3**2 - p1**2 - A3**2 - B3**2)
    N2 = -A3 * (p2**2 - p1**2 - A2**2 - B2**2) + A2 * (p3**2 - p1**2 - A3**2 - B3**2)
    return N1**2 + N2**2 - p1**2 * D**2

x_array = np.linspace(-np.pi, np.pi,400)
plt.plot(x_array, f(x_array))
plt.grid()
plt.show()

def secant(f, x0, x1, k):
    for i in range(1,k):
        x2 = x1 - f(x1)*(x1-x0)/(f(x1)-f(x0))
        x0 =x1 
        x1 = x2
    return x2


#def rootfinder(start, stop, num):
#    initial_guesses = np.linspace(start, stop, num)  # Multiple guesses for better coverage
#    roots = []
#
#    for i in initial_guesses:
#        root = fsolve(f, i)
#        # Add unique roots only
#        if root not in roots :
#            roots.append(root)
#
#    roots = np.array(roots).flatten()
#    print("Roots found:", roots)
#
#    # Convert roots to a more usable format
#    return roots
#
## Print the roots
#rootfinder(-np.pi, np.pi, 5)


import numpy as np 
import matplotlib.pyplot as plt 

L1 = 3
L2 = 3 * np.sqrt(2)
L3 = 3
gamma = np.pi / 4
x1 = 5 
x2 = 0
y2 = 6
p1 = 5
p2 = 5
p3 = 3


# Define the function f(θ) from earlier
def f(theta):
    A2 = L3 * np.cos(theta) - x1 
    B2 = L3 * np.sin(theta)
    A3 = L2 * np.cos(theta + gamma) - x2
    B3 = L2 * np.sin(theta + gamma) - y2
    D = 2 * (A2 * B3 - B2 * A3)
    N1 = B3 * (p2**2 - p1**2 - A2**2 - B2**2) - B2 * (p3**2 - p1**2 - A3**2 - B3**2)
    N2 = -A3 * (p2**2 - p1**2 - A2**2 - B2**2) + A2 * (p3**2 - p1**2 - A3**2 - B3**2)
    return N1**2 + N2**2 - p1**2 * D**2

def triangleplotting(theta):

    A2 = L3*np.cos(theta)-x1 
    B2= L3*np.sin(theta)
    A3 = L2*np.cos(theta + gamma) - x2
    B3 = L2*np.sin(theta+ gamma) -y2
    D = 2 * (A2* B3 - B2*A3)
    N1 = B3*(p2**2-p1**2-A2**2-B2**2)-B2*(p3**2-p1**2-A3**2-B3**2)
    N2 = -A3*(p2**2-p1**2-A2**2-B2**2)+A2*(p3**2-p1**2-A3**2-B3**2)
    x = N1/D
    y = N2/D
    
    #coordinate points for the triangle 
    u1 = N1/D
    u2 = x + L3 * np.cos(theta)
    u3 = x + L2 * np.cos(theta + gamma)
    m1 = N2/D
    m2 = y + L3 * np.sin(theta)
    m3 = y + L2 * np.sin(gamma + theta)

    plt.grid()
    plt.autoscale()
    #plots the inner triangle.
    plt.plot([x,u3, x + L3 * np.cos(theta),x ],[y, y + L2 * np.sin(gamma + theta), y + L3 * np.sin(theta),y ])
    
    #plots strut 1
    plt.plot([0, x],[0, y])

    #plots strut 2
    plt.plot([x1,u2 ],[0, m2])

    #plots strut 3
    plt.plot([x2, u3],[y2, m3])
    plt.title('Theta at '+ str(theta))

#tested Triangle plot
#triangleplotting( np.pi /4)

def secant(f, x0, x1, k):
    for i in range(1,k):
        x2 = x1 - f(x1)*(x1-x0)/(f(x1)-f(x0))
        x0 =x1 
        x1 = x2
    return x2

#fig,axes = plt.subplots(2,2, figsize = 10)
plt.figure()
firsttheta = secant(f, -1, -0.6, 10)
triangleplotting(firsttheta)
#print(firsttheta)

plt.figure()
secondtheta = secant(f, -.4, -.3, 5)
#print(secondtheta)
triangleplotting(secondtheta)
#
plt.figure()
thirdtheta = secant(f, 1, 1.2, 5)
triangleplotting(thirdtheta)
#
plt.figure()
fourththeta = secant(f, 2, 2.2, 5)
triangleplotting(fourththeta)


p2 = 7

x_array = np.linspace(-np.pi, np.pi,400)
plt.plot(x_array, f(x_array))
plt.grid()
plt.show()

g1 = secant(f,-.7, -.6, 3)
g2 = secant(f,-.4, -.3, 3)
g3 = secant(f,0, .1, 3)
g4 = secant(f,.2, .5, 3)
g5 = secant(f,.9, 1.1, 3)
g6 = secant(f,2.3, 2.6, 3)

plt.figure()
triangleplotting(g1)
plt.figure()
triangleplotting(g2)
plt.figure()
triangleplotting(g3)
plt.figure()
triangleplotting(g4)
plt.figure()
triangleplotting(g5)
plt.figure()
triangleplotting(g6)

p2 = 4

x_array = np.linspace(-np.pi, np.pi,400)
plt.plot(x_array, f(x_array))
plt.grid()
plt.show()

g1 = secant(f,1,1.5, 5)
#print(g1)
#print(g2)
g2 = secant(f, 1.5, 2, 5)

plt.figure()
triangleplotting(g1)
plt.figure()
triangleplotting(g2)


p2 = 0
p2_f = 10
d_p2 =  0.01

# Creating an interval to define where there are exactly 4 roots in our f function. 
print('0')
Theta = np.linspace(-np.pi, np.pi, 1000)
zeros_prev= 0 
while p2< p2_f:
    zeros = 0
    function = f(Theta)
    value1= 1 
    for value in function: 
       if value1*value < 0:
           zeros += 1
       value1 = value 
    if zeros != zeros_prev:
       print(p2)
   
    zeros_prev = zeros 
    p2 += d_p2

print('0')
