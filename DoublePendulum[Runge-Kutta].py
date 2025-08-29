{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import numpy as np\
import matplotlib.pyplot as plt\
from matplotlib.animation import FuncAnimation\
import matplotlib as mpl\
mpl.rcParams['animation.embed_limit'] = 40 #Memory assigned is 40 MB\
try:\
    # If SciPy is available, use it\
    from scipy.integrate import solve_ivp\
    use_scipy = True\
except Exception:\
    use_scipy = False\
\
# Starting parameters -- length, mass, angles and velocity\
l1, l2 = 1.0, 1.0       # lengths\
m1, m2 = 1.0, 1.0       # masses\
g = 9.81                # gravity\
theta1_0, theta2_0 = np.pi/2, np.pi/2  # initial angles (from vertical, positive CCW)\
omega1_0, omega2_0 = 0.0, 0.0          # initial angular velocities\
\
# Equations of motion (state: [\uc0\u952 1, \u969 1, \u952 2, \u969 2]) ---\
def deriv(t, y):\
    \uc0\u952 1, \u969 1, \u952 2, \u969 2 = y\
    \uc0\u948  = \u952 2 - \u952 1\
    c, s = np.cos(\uc0\u948 ), np.sin(\u948 )\
    den1 = (m1 + m2)*l1 - m2*l1*c*c\
    den2 = (l2/l1)*den1\
\
    d\uc0\u952 1 = \u969 1\
    d\uc0\u952 2 = \u969 2\
    d\uc0\u969 1 = (m2*l1*\u969 1*\u969 1*s*c + m2*g*np.sin(\u952 2)*c + m2*l2*\u969 2*\u969 2*s - (m1+m2)*g*np.sin(\u952 1)) / den1\
    d\uc0\u969 2 = (-m2*l2*\u969 2*\u969 2*s*c + (m1+m2)*(g*np.sin(\u952 1)*c - l1*\u969 1*\u969 1*s - g*np.sin(\u952 2))) / den2\
    return np.array([d\uc0\u952 1, d\u969 1, d\u952 2, d\u969 2], dtype=float)\
\
# Time scale and interval of data collection\
t0, t1, fps = 0.0, 20.0, 60\
t_eval = np.linspace(t0, t1, int((t1 - t0)*fps))\
\
y0 = np.array([theta1_0, omega1_0, theta2_0, omega2_0], dtype=float)\
\
if use_scipy:\
    sol = solve_ivp(deriv, (t0, t1), y0, t_eval=t_eval, rtol=1e-9, atol=1e-9)\
    \uc0\u952 1, \u952 2 = sol.y[0], sol.y[2]\
else:\
    # Tiny RK4 fallback so the example still runs without SciPy\
    def rk4(f, y, t):\
        dt = t[1] - t[0]\
        Y = np.empty((len(y), len(t)))\
        Y[:, 0] = y\
        for i in range(len(t)-1):\
            k1 = f(t[i], Y[:, i])\
            k2 = f(t[i]+dt/2, Y[:, i] + dt*k1/2)\
            k3 = f(t[i]+dt/2, Y[:, i] + dt*k2/2)\
            k4 = f(t[i]+dt,   Y[:, i] + dt*k3)\
            Y[:, i+1] = Y[:, i] + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)\
        return Y\
    Y = rk4(deriv, y0, t_eval)\
    \uc0\u952 1, \u952 2 = Y[0], Y[2]\
\
#  Cartesian coodinate of bobs\
x1 = l1*np.sin(\uc0\u952 1);  y1 = -l1*np.cos(\u952 1)\
x2 = x1 + l2*np.sin(\uc0\u952 2);  y2 = y1 - l2*np.cos(\u952 2)\
\
# animation setup\
fig, ax = plt.subplots()\
ax.set_aspect('equal')\
R = 1.2*(l1 + l2)\
ax.set_xlim(-R, R); ax.set_ylim(-R, R)\
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title("Double Pendulum")\
\
(line,) = ax.plot([], [], "o-", lw=2)   # rods + bobs\
(trace,) = ax.plot([], [], "-", alpha=0.5)  # trace of mass 2\
trace_x, trace_y = [], []\
\
def init():\
    line.set_data([], [])\
    trace.set_data([], [])\
    trace_x.clear(); trace_y.clear()\
    return line, trace\
\
def update(i):\
    thisx = [0, x1[i], x2[i]]\
    thisy = [0, y1[i], y2[i]]\
    line.set_data(thisx, thisy)\
    trace_x.append(x2[i]); trace_y.append(y2[i])\
    trace.set_data(trace_x, trace_y)\
    return line, trace\
\
ani = FuncAnimation(fig, update, frames=len(t_eval), init_func=init, blit=True, interval=1000/fps)\
\
try:\
    from IPython.display import HTML\
    plt.rcParams["animation.html"] = "jshtml"\
    display(HTML(ani.to_jshtml()))\
except Exception:\
    plt.show()\
}