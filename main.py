# python
import numpy as np
import matplotlib.pyplot as plt

# notation difference from handwritten notes: I dropped the _r subscript for the dimensionless values.

# parameters
beta = 0.3
T_a = 2 # this is not used
p_a = 5
tau = 0.2
f = 3 # Degrees of Motion

# Q = 0 # heating is 0. How is this enforced if $T_a$ is used? Will there be inconsistency or divergence? Where is $T_a$ used? GGG

# simulation parameters
dt = 0.0001 # to maintain stability dt <<< tau
N = 100000
Time = N*dt

# eqs of state
def p_rev(T, v): 
    return ((8*T) / ((3*v)-1)) - (3 / (v**2))

# thermo relations. 
# Q: are these also eqs of state?
def e_rev(T, v): # these two are the same
    return (4/3)*f*T - (3/v)
def T(e_rev, v): # these two are the same
    return (3/4/f)*(e_rev+3/v)
def e(e_rev, T, p_irr):
    return e_rev + tau*(p_irr**2)/(2*beta)
def p(p_rev, p_irr):
    return p_rev + p_irr

# diff eqs
# notation: the dt_ means time derivative
def dt_p_irr(b, p_irr):
    return -(beta*b + p_irr) / tau
def dt_v(b):
    return b
def dt_e(p, b):
    return -p*b
def dt_b(p, p_a):
    return p-p_a



# initialize 
# notation: 
# _n: the physically meaningful values
# _m: values shifted by -1/2

# diferential
p_irr_n = np.zeros(N) 
p_irr_m = np.zeros(N)  

e_n = np.zeros(N) 
e_m = np.zeros(N)  

# b_n = np.zeros(N)  # it was curious to me that we did not need the b_n values, only the ones at (1/2)*dt shifted times. this is because it is the time-derivative of v_n. 
b_m = np.zeros(N)  

v_n = np.zeros(N) 
v_m = np.zeros(N)  

# algebraic
p_n = np.zeros(N) 
p_m = np.zeros(N)  

e_rev_n = np.zeros(N) 
e_rev_m = np.zeros(N)  

p_rev_n = np.zeros(N) 
p_rev_m = np.zeros(N)  

T_n = np.zeros(N) 
T_m = np.zeros(N)  


# initial conditions
T_n[0] = 1.1 # slightly hypercritical
v_n[0] = 1.1 # slightly hypercritical
b_n0 = 0 # initially static
p_irr_n[0] = 0 # I think it makes sense since the idea is that you start close to critical point. 
# Q: is the critical point a stable point? GGG

p_rev_n[0] = p_rev(T_n[0], v_n[0]) 
e_rev_n[0] = e_rev(T_n[0], v_n[0])
# extra
e_n[0] = e(e_rev_n[0], T_n[0], p_irr_n[0])
p_n[0] = p(p_rev_n[0], p_irr_n[0]) 

# these "pre-initial values" are calculated using the euler method in reverse. a crude approximation. GGG
# notation: the _m means values shifted by -1/2
b_m[0] = b_n0 - (dt/2) * dt_b(p_n[0], p_a) # reverse euler
v_m[0] = v_n[0] - (dt/2) * dt_v(b_n0)

p_irr_m[0] = p_irr_n[0] - (dt/2) * dt_p_irr(b_n0, p_irr_n[0])
e_m[0] = e_n[0] - (dt/2) * dt_e(p_n[0], b_n0) # reverse euler


# I need p_rev_m[0], how do I get it? It is a state variable p_rev(T, v)

# Compute T_m[0] from e_rev_m[0] and v_m[0]
e_rev_m[0] = e_m[0] - (tau/2/beta) * (p_irr_m[0])**2 # this is obtained from e(e_rev, T, p_irr)
T_m[0] = T(e_rev_m[0], v_m[0]) # I could have done T_m[0] = T_n[0] - (dt/2) * dt_T(...) but I don't have dt_T ! So I calculated e_rev_m[0]
p_rev_m[0] = p_rev(T_m[0], v_m[0])

p_rev_n[-1] = 2*p_rev_m[0] - p_rev_n[0] # I know that [-1] refers to the last element of the array, but this will just get overwritten at the end with the correct physical value. => no trouble. clever.

# # idk. handwritten notation. GGG
# t_u = m*X_p*v_c/p_c
# t_r = t / t_u

# I could have vectorized this for numerical optimization if performance had been a problem. Luckily, using a simple for-loop was fast enough for my use-case.
for i in range(N-1):

    b_m[i+1] = b_m[i] + dt * (p_n[i] - p_a)
    # p_irr_m[i+1] = (1 / ((tau/dt)+(1/2)) ) * ( (-beta * (1/2) * (b_m[i+1] + b_m[i])) + (((tau/dt)-(1/2)) * p_irr_m[i]) ) # variant 1: here we use (b_m[i+1] + b_m[i]), but we don't use b_n[i], because we don't calculate the b_n[i] array
    p_irr_m[i+1] = np.exp(-dt/tau) * p_irr_m[i] + beta * ( (((b_m[i+1]-b_m[i]) * (tau/dt)) - b_m[i+1]) - (((b_m[i+1]-b_m[i]) * (tau/dt)) - b_m[i])*np.exp(-dt/tau) ) # variant 2. The past stress decays exponentially, so it has some memory and is not instantaneous.

    # diff
    # linear interpolation. use better
    p_irr_n[i+1] = 2 * p_irr_m[i+1] - p_irr_n[i]
    p_rev_m[i+1] = 2 * p_rev_n[i] - p_rev_m[i]
    # parabolic interpolation AAA. causes instability GGG
    # p_irr_n[i+1] = 3*p_irr_m[i+1] -3*p_irr_n[i] + p_irr_m[i] 
    # p_rev_m[i+1] = 3*p_rev_n[i] -3*p_rev_m[i] + p_rev_n[i-1]
    
    #  GGG delete
    # p_irr_n[i+1] = (3/8) * p_irr_n[i] + (3/4) * p_irr_m[i+1] - (1/8) * p_irr_m[i]
    # p_rev_m[i+1] = (3/8) * p_rev_m[i] + (3/4) * p_rev_n[i] - (1/8) * p_rev_n[i-1]

    # Also, honestly, the linear interpolation you currently have is the standard and recommended approach for staggered leapfrog schemes. It's second-order accurate and doesn't introduce additional instabilities. Parabolic interpolation can sometimes cause spurious oscillations, especially in dissipative systems like yours with the irreversible pressure term.

    # algebraic
    p_m[i+1] = p(p_rev_m[i+1], p_irr_m[i+1]) # p_rev_m[i+1] + p_irr_m[i+1] 

    # diff
    v_n[i+1] = v_n[i] + dt * dt_v(b_m[i+1])
    e_n[i+1] = e_n[i] + dt * dt_e(p_m[i+1], b_m[i+1])

    # algebraic
    # e_rev_n[i+1] = e_rev(T_n[i+1], v_n[i+1]) # this is what (naively) I would have done, but we havent calculated T_n[i+1] yet! it is 0! so we do:
    e_rev_n[i+1] = e_n[i+1] - (tau/2/beta) * (p_irr_n[i+1])**2 
    # now effectively we calculate T_n[i+1]
    T_n[i+1] = T(e_rev_n[i+1], v_n[i+1])  
    p_rev_n[i+1] = p_rev(T_n[i+1], v_n[i+1])
    p_n[i+1] = p(p_rev_n[i+1], p_irr_n[i+1])    

t = np.linspace(start=0, stop=Time, num=N) # GGG Q: time does it need any normalization by the critical values?

# visualization
fig, ax1 = plt.subplots()

var_colors = {
    'v_n': '#0072B2',
    'T_n': '#D55E00',
    'b_m': '#009E73',
    # 'e_n': '#CC79A7',
}

# Base axis
ax1.plot(t, v_n, color=var_colors['v_n'], label='v_n')
ax1.set_ylabel('v_n', color=var_colors['v_n'])
ax1.tick_params(axis='y', labelcolor=var_colors['v_n'])

# Second axis
ax2 = ax1.twinx()
ax2.plot(t, T_n, color=var_colors['T_n'], label='T_n')
ax2.set_ylabel('T_n', color=var_colors['T_n'])
ax2.tick_params(axis='y', labelcolor=var_colors['T_n'])

# Third axis (offset)
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('axes', 1.1))
ax3.plot(t, b_m, color=var_colors['b_m'], label='b_m')
ax3.set_ylabel('b_m', color=var_colors['b_m'])
ax3.tick_params(axis='y', labelcolor=var_colors['b_m'])

# # Fourth axis (offset further)
# ax4 = ax1.twinx()
# ax4.spines['right'].set_position(('axes', 1.2))
# ax4.plot(t, e_n, color=var_colors['e_n'], label='e_n')
# ax4.set_ylabel('e_n', color=var_colors['e_n'])
# ax4.tick_params(axis='y', labelcolor=var_colors['e_n'])

ax1.set_xlabel('t')
plt.title('Variables vs t')
plt.tight_layout()
plt.show()


# ---------- extra: entropy

s_rev_n = (f/2) * np.log(T_n) + np.log(3*v_n - 1)
s_n = s_rev_n - (tau / (2 * beta * T_n)) * p_irr_n**2
# Entropy production rate
s_dot = -p_irr_n * b_m / T_n # (p_irr_n**2) / (beta * T_n)




fig, ax1 = plt.subplots()

# Base axis
ax1.plot(t, s_rev_n, color=var_colors['v_n'], label='s_rev_n')
ax1.set_ylabel('s_rev_n', color=var_colors['v_n'])
ax1.tick_params(axis='y', labelcolor=var_colors['v_n'])

# Second axis
ax2 = ax1.twinx()
ax2.plot(t, s_n, color=var_colors['T_n'], label='s_n')
ax2.set_ylabel('s_n', color=var_colors['T_n'])
ax2.tick_params(axis='y', labelcolor=var_colors['T_n'])

# Third axis (offset)
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('axes', 1.1))
ax3.plot(t, s_dot, color=var_colors['b_m'], label='s_dot')
ax3.set_ylabel('s_dot', color=var_colors['b_m'])
ax3.tick_params(axis='y', labelcolor=var_colors['b_m'])

ax1.set_xlabel('t')
plt.title('Variables vs t')
plt.tight_layout()
plt.show()


# ------------ show pressure, then settles to p_a

plt.plot(t, p_n, label="p")
plt.hlines(y=[5], xmin=np.min(t), xmax=np.max(t), label="p_a", linestyles="dashed", colors="red")
plt.title('Pressure vs t')
plt.xlabel("t")
plt.xlabel("p")
plt.legend()
plt.tight_layout()
plt.show()
# The gas pressure settles to the atmospheric pressure



"""
Comments:

Explicitly writing the solver foced me to handle each variable and parameter and gave me a richer understanding of the system. For example, when first reading the project description, the Q=0 condition was curious to me and I wondered how this could be "enforced" given that there is a T_a (I even wrote it in my notes at the time)
After writing this program, I realized that T_a is not used at all, which explains how this Q=0 makes sense.

From a software development perspective, it was very benefitial to start with the minimal possible implementation (using the simpler variantes, like linear extrapolation), and then increasing the complexity step-by-step.
"""