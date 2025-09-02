import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
mpl.rcParams['animation.embed_limit'] = 40 #Memory assigned is 40 MB
try:
    # If SciPy is available, use it
    from scipy.integrate import solve_ivp
    use_scipy = True
except Exception:
    use_scipy = False

# Starting parameters -- length, mass, angles and velocity
l1, l2 = 1.0, 1.0       # lengths
m1, m2 = 1.0, 1.0       # masses
g = 9.81                # gravity
theta1_0, theta2_0 = np.pi/2, np.pi/2  # initial angles (from vertical, positive CCW)
omega1_0, omega2_0 = 0.0, 0.0          # initial angular velocities

# Equations of motion (state: [θ1, ω1, θ2, ω2]) ---
def deriv(t, y):
    θ1, ω1, θ2, ω2 = y
    δ = θ2 - θ1
    c, s = np.cos(δ), np.sin(δ)
    den1 = (m1 + m2)*l1 - m2*l1*c*c
    den2 = (l2/l1)*den1

    dθ1 = ω1
    dθ2 = ω2
    dω1 = (m2*l1*ω1*ω1*s*c + m2*g*np.sin(θ2)*c + m2*l2*ω2*ω2*s - (m1+m2)*g*np.sin(θ1)) / den1
    dω2 = (-m2*l2*ω2*ω2*s*c + (m1+m2)*(g*np.sin(θ1)*c - l1*ω1*ω1*s - g*np.sin(θ2))) / den2
    return np.array([dθ1, dω1, dθ2, dω2], dtype=float)

# Time scale and interval of data collection
t0, t1, fps = 0.0, 20.0, 60
t_eval = np.linspace(t0, t1, int((t1 - t0)*fps))

y0 = np.array([theta1_0, omega1_0, theta2_0, omega2_0], dtype=float)

if use_scipy:
    sol = solve_ivp(deriv, (t0, t1), y0, t_eval=t_eval, rtol=1e-9, atol=1e-9)
    θ1, θ2 = sol.y[0], sol.y[2]
else:
    # Tiny RK4 fallback so the example still runs without SciPy
    def rk4(f, y, t):
        dt = t[1] - t[0]
        Y = np.empty((len(y), len(t)))
        Y[:, 0] = y
        for i in range(len(t)-1):
            k1 = f(t[i], Y[:, i])
            k2 = f(t[i]+dt/2, Y[:, i] + dt*k1/2)
            k3 = f(t[i]+dt/2, Y[:, i] + dt*k2/2)
            k4 = f(t[i]+dt,   Y[:, i] + dt*k3)
            Y[:, i+1] = Y[:, i] + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        return Y
    Y = rk4(deriv, y0, t_eval)
    θ1, θ2 = Y[0], Y[2]

#  Cartesian coodinate of bobs
x1 = l1*np.sin(θ1);  y1 = -l1*np.cos(θ1)
x2 = x1 + l2*np.sin(θ2);  y2 = y1 - l2*np.cos(θ2)

# animation setup
fig, ax = plt.subplots()
ax.set_aspect('equal')
R = 1.2*(l1 + l2)
ax.set_xlim(-R, R); ax.set_ylim(-R, R)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title("Double Pendulum")

(line,) = ax.plot([], [], "o-", lw=2)   # rods + bobs
(trace,) = ax.plot([], [], "-", alpha=0.5)  # trace of mass 2
trace_x, trace_y = [], []

def init():
    line.set_data([], [])
    trace.set_data([], [])
    trace_x.clear(); trace_y.clear()
    return line, trace

def update(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]
    line.set_data(thisx, thisy)
    trace_x.append(x2[i]); trace_y.append(y2[i])
    trace.set_data(trace_x, trace_y)
    return line, trace

ani = FuncAnimation(fig, update, frames=len(t_eval), init_func=init, blit=True, interval=1000/fps)

try:
    from IPython.display import HTML
    plt.rcParams["animation.html"] = "jshtml"
    display(HTML(ani.to_jshtml()))
except Exception:
    plt.show()
