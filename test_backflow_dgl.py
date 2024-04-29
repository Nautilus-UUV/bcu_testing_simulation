import numpy as np
import scipy
from matplotlib import pyplot as plt

# everything in SI units
p_0 = 0.5 * 10 ** 5  # =1bar
V_0 = 3.5 * 10 ** (-3)  # = 3.5l
A = 0.2 * 0.2
V_t_i = 3 * 10 ** (-3)  # 3 liter initial tank filling with N_2
h_i = 0.5 + 11.8
V_c_i = A * h_i
rho = 0.86 * 10 ** 3
g = 9.81
r = 0.004#0.0023
eta = 0.774
l = 0.25


def h_t_dgl(h, t):
    return ((p_0 * V_0) / (h * A + V_t_i - V_c_i) - h * rho * g) * (np.pi * r ** 4) / (8 * A * eta * l)


t = np.linspace(0, 500, 1000)
sol = scipy.integrate.odeint(h_t_dgl, h_i, t, atol=1e-9, rtol=1e-9)

# height
plt.plot(t, sol)
plt.title('height (m) over t (s)')
plt.legend()
# plt.savefig('height.png')
plt.show()



# volume tank

plt.plot(t, (sol * A + V_t_i - V_c_i) * 10 ** 3)
plt.title('volume tank (l) over t (s), t')
plt.legend()
# plt.savefig('volume.png')

plt.show()

# pressure
plt.plot(t, p_0 * V_0 / (sol * A + V_t_i - V_c_i) / 10 ** 5, label='pressure tank')
plt.plot(t, sol * rho * g / 10 ** 5, label='pressure container')
plt.title('pressure (bar) over t(s)')
plt.legend()
# plt.savefig('pressure.png')

plt.show()
