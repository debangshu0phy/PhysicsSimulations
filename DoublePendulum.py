import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
import matplotlib as mpl
mpl.rcParams['animation.embed_limit'] = 40 #Memory assigned is 40 MB

# Parameters (user can choose these)
l1 = float(input("Enter length of first pendulum (m1): "))
l2 = float(input("Enter length of second pendulum (m2): "))
m1 = float(input("Enter mass of first bob (kg): "))
m2 = float(input("Enter mass of second bob (kg): "))
theta1_deg = float(input("Enter initial angle of pendulum 1 (in degrees): "))
theta2_deg = float(input("Enter initial angle of pendulum 2 (in degrees): "))
theta1_0 = np.radians(theta1_deg)
theta2_0 = np.radians(theta2_deg)

# start from rest unless user wants to modify
omega1_0 = 0.0 
omega2_0 = 0.0
g = 9.80

# Define equations of motion for double pendulum
def deriv(t, y):
    θ1, ω1, θ2, ω2 = y
    δ = θ2 - θ1
    c, s = np.cos(δ), np.sin(δ)
    den1 = (m1 + m2)*l1 - m2*l1*c*c
    den2 = (l2/l1)*den1

    dθ1 = ω1
    dθ2 = ω2
    dω1 = (m2*l1*ω1*ω1*s*c + m2*g*np.sin(θ2)*c + m2*l2*ω2*ω2*s - (m1+m2)*g*np.sin(θ1))/den1
    dω2 = (-m2*l2*ω2*ω2*s*c + (m1+m2)*(g*np.sin(θ1)*c - l1*ω1*ω1*s - g*np.sin(θ2)))/den2
    return np.array([dθ1, dω1, dθ2, dω2], dtype=float)

# Time span and frame rate
t0, t1, fps = 0.0, 20.0, 45
t_eval = np.linspace(t0, t1, int((t1 - t0)*fps))

y0 = np.array([theta1_0, omega1_0, theta2_0, omega2_0], dtype=float)

# Solve ODE and extract solutions
sol = solve_ivp(deriv, (t0, t1), y0, t_eval=t_eval, rtol=1e-9, atol=1e-9)
θ1, θ2 = sol.y[0], sol.y[2]

#Check if initial starting angles have been taken in properly or not
print("Initial starting angle of pendulums:", sol.y[:, 0])

# Convert to Cartesian coordinates
x1 = l1*np.sin(θ1)
y1 = -l1*np.cos(θ1)
x2 = x1 + l2*np.sin(θ2)
y2 = y1 - l2*np.cos(θ2)

# Setting up the animation
fig, ax = plt.subplots()
ax.set_aspect('equal')
R = 1.2*(l1 + l2)
ax.set_xlim(-R, R); ax.set_ylim(-R, R)
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title(f"Double Pendulum: L1={l1}m, L2={l2}m, M1={m1}kg, M2={m2}kg")

(line,) = ax.plot([], [], "o-", lw=2)   # rods + bobs
(trace,) = ax.plot([], [], "-", alpha=0.5)  # trace of mass 2
trace_x, trace_y = [], []

def init():
    line.set_data([], [])
    trace.set_data([], [])
    trace_x.clear()
    trace_y.clear()
    return line, trace

def update(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]
    line.set_data(thisx, thisy)
    if i%2 == 0:
        trace_x.append(x2[i])
        trace_y.append(y2[i])
    trace.set_data(trace_x, trace_y)
    return line, trace

# --- Create animation ---
ani = FuncAnimation(fig, update, frames=len(t_eval), init_func=init, blit=True, interval=1000/fps)

from IPython.display import HTML
HTML(ani.to_jshtml())