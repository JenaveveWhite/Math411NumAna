
## IMPORTS
import numpy as np 
import matplotlib as mlt
import plotly.express as px 
from scipy.integrate import quad
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.animation as animation

## FUNCTIONS

# ACTIVITY 1
def P(t): 
    x = 0.5 + 0.3 * t + 3.9 * (t ** 2) - 4.7 * (t ** 3)
    y = 1.5 + 0.3 * t + 0.9 * (t ** 2) - 2.7 * (t ** 3)
    return x,y 
def derP(t):
    x = 0.3 + 2 * 3.9 * t - 3 * 4.7 * (t ** 2)
    y = 0.3 + 2 * 0.9 * t - 3 * 2.7 * (t ** 2)
    return x,y

# Derivatives of x and y with respect to t
def dx_dt(t):
    return 0.3 + 2 * 3.9 * t - 3 * 4.7 * (t ** 2)
def dy_dt(t):
    return 0.3 + 2 * 0.9 * t - 3 * 2.7 * (t ** 2)

# Integrand for the arc length
def integrand(t):
    return np.sqrt(dx_dt(t)**2 + dy_dt(t)**2) 
def arc_length(e):
    # Calculate the arc length from 0 to s
    length, _ = quad(integrand, 0, e)
    return length

# s is the start of t and e is the end of t
e = 1

length = arc_length(e)

print(f"The arc length from 0 to t={e} is: {length}")

# ACTIVITY 2
def bisect(f, a, b, tol, s):
    fa = f(a,s)
    fb = f(b,s)
    n = 0
    if np.sign(fa*fb) >= 0:
        print('f(a)f(b) < 0 not satisfied!')
        quit()
    while (b-a)/2. > tol:
        n = n + 1
        c = (a+b)/2.
        fc = f(c,s)
        if fc == 0:
            return c
        if np.sign(fc*fa) < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    return [(a+b)/2., n]

def obj(tstar,s):
    return arc_length(tstar) - s * arc_length(1)

value, n = bisect(obj, 0, 1, .0005, 1/2)

# ACTIVITY 
def equipar(n):
    partition_points = [0]  # Start at t=0
    total_length = arc_length(1)
    segment_length = total_length / n
    for i in range(1, n):
        s = i / n
        t, _ = bisect(obj, 0, 1, 0.0005, s)
        partition_points.append(t)
    partition_points.append(1)  # End at t=1
    return np.array(partition_points)

def plot_equipartitions(n, func):
    t_vals = np.linspace(0, 1, 1000)
    x_vals, y_vals = func(t_vals)
    partition_points = equipar(n)
    partition_coords = [func(t) for t in partition_points]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines'))
    for coord in partition_coords:
        fig.add_trace(go.Scatter(x=[coord[0]], y=[coord[1]], mode='markers'))
    fig.update_layout(title=f'Equipartition of Path into {n} Subpaths',showlegend=False, width = 500, height = 500)
    fig.update_xaxes(range=[-0.1, 1.8])
    fig.update_yaxes(range=[-0.1, 1.8])
    fig.show()


plot_equipartitions(4, P)
plot_equipartitions(20, P)

# ACTIVTY 4
def newton(f, fp, x0, k):
    xc = x0
    for i in range(1, k):
        xc = xc - f(xc)/fp(xc)
    return xc

def obj(tstar,s):
    return arc_length(tstar) - s * arc_length(1)

## What is the derivative needed to use Newton's Method?
def equipar_newton(n):
    partition_points = [0] 
    total_length = arc_length(1)
    segment_length = total_length / n

    for i in range(1, n):
        s = i / n
        t_guess = 0.5 
        # Use Newton's method to find the value of t for the given s
        t = newton(lambda t: obj(t, s), lambda t: np.sqrt(dx_dt(t)**2 + dy_dt(t)**2), t_guess, 10)
        partition_points.append(t)
    partition_points.append(1) 
    return partition_points

def plot_newequipartitions(n, func):
    t_vals = np.linspace(0, 1, 1000)
    x_vals, y_vals = func(t_vals)
    partition_points = equipar_newton(n)
    partition_coords = [func(t) for t in partition_points]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines'))
    for coord in partition_coords:
        fig.add_trace(go.Scatter(x=[coord[0]], y=[coord[1]], mode='markers'))
    fig.update_layout(title=f'Equipartition of Path into {n} Subpaths',showlegend=False, width = 500, height = 500)
    fig.update_xaxes(range=[-0.1, 1.8])
    fig.update_yaxes(range=[-0.1, 1.8])
    fig.show()


plot_newequipartitions(4, P)
plot_newequipartitions(20, P)

# ACTIVITY 5

x4, y4 = P(equipar(4))
x20, y20 = P(equipar(20))

data4 = np.vstack([x4,y4])
data20 = np.vstack([x20,y20])

#altdata4 = P(np.linspace(0,1,5))
altdata4 = np.vstack(P(np.linspace(0,1,5)))
#print(altdata4)

altdata20 = np.vstack(P(np.linspace(0,1,21)))
#print(altdata20)


# Create the figure, axis, and plot
fig, ax = plt.subplots(ncols=2)

left, = ax[0].plot([], [], 'r:o')
right, = ax[1].plot([], [], 'r:o')

for axis in ax: 
    axis.set_xlim(-0.1, 2)
    axis.set_ylim(-0.1, 2)
    axis.set_xlabel('x')

ax[0].set_title('Constant-Speed traversal')
ax[1].set_title('Original Speed')


#update = lambda fnum: left.set_data(data20[..., :fnum]); right.set_data(altdata20[..., :fnum])
 
def update(fnum):
    left.set_data(data20[..., :fnum])
    right.set_data(altdata20[..., :fnum])
    return left, right

# run a frame for every column
r,c = np.shape(data20)

# Set up the animation
ani = animation.FuncAnimation(fig, update, frames=c+1, interval=500)
plt.show()

# ACTIVITY 6


def newpath(t):
    x = np.cos(t)
    y = np.sin(t)
    return x,y

def dx_dt2(t):
    return -np.sin(t)

def dy_dt2(t):
    return np.cos(t)

# Integrand for the arc length
def integrand2(t):
    return np.sqrt(dx_dt2(t)**2 + dy_dt2(t)**2)
   
def arc_length2(e):
    # Calculate the arc length from 0 to s
    length, _ = quad(integrand2, 0, e)
    return length


def obj2(tstar,s):
    return arc_length2(tstar) - s * 6.283

def equipar2(n):
    partition_points = [0]  # Start at t=0
    total_length = arc_length2(2 * np.pi)
    #print(total_length)
    segment_length = total_length / n
    for i in range(1, n):
        s = i / n
        t, _ = bisect(obj2, 0, 2 * np.pi, 0.0005, s)
        partition_points.append(t)
    partition_points.append(2 * np.pi)
    partition_points.append(0)  # End at t=1
    return np.array(partition_points)

def plot_equipartitions2(n, func):
    t_vals = np.linspace(0, 2 * np.pi , 1000)
    x_vals, y_vals = func(t_vals)
    partition_points = equipar2(n)
    partition_coords = [func(t) for t in partition_points]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines'))
    for coord in partition_coords:
        fig.add_trace(go.Scatter(x=[coord[0]], y=[coord[1]], mode='markers'))
    fig.update_xaxes(range=[-1.5, 1.5])
    fig.update_yaxes(range=[-1.5, 1.5])
    fig.update_layout(title=f'Equipartition of Path into {n} Subpaths',showlegend=False, width = 500, height = 500)
    fig.show()


#plot_equipartitions2(20, newpath)

# ANIMATION PLOT
#print(equipar2(20))
x20, y20 = newpath(equipar2(20))

data20 = np.vstack([x20,y20])


# Create the figure, axis, and plot
fig, ax = plt.subplots()
l, = ax.plot([], [], 'r:o')
# adjust axis parameters to make it "pretty"
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.xlabel('x')
plt.ylabel('y')
ax.set_box_aspect(1)
plt.title('Parametric Circle Graph')
plt.grid()

#update = lambda fnum: left.set_data(data20[..., :fnum]); right.set_data(altdata20[..., :fnum])
update = lambda fnum: l.set_data(data20[..., :fnum])

# run a frame for every column
r,c = np.shape(data20)

# Set up the animation
ani = animation.FuncAnimation(fig, update, frames=c, interval= 300)
plt.show()

