# Question 2 - Calibrating the model

#%% Import packages
import numpy as np
import matplotlib.pyplot as plt
from sequence_jacobian import simple, create_model

#%%
from sequence_jacobian.utilities.drawdag import drawdag


#%% Question 2.1 - Calibrating the baseline RBC model
# Define parameters
eta = 1
alpha = 0.33
rho = 0.9
sigma = 0.01
beta = 0.99 
b = 6.57 
r = 0.01
delta = 0.02


#%%
#%% Set up model

@simple
def firm(K, N, A, alpha, delta):
    r = alpha * A * (K(-1) / N) ** (alpha-1) - delta
    w = (1 - alpha) * A * (K(-1) / N) ** alpha
    Y =  A * K(-1) ** alpha * N ** (1 - alpha)
    return r, w, Y

@simple
def household(K, N, w, eta, b, delta):
    C = w /(b * N**eta)
    I = K - (1 - delta) * K(-1)
    return C, I

@simple
def mkt_clearing(r, C, Y, I, K, N, w, beta):
    goods_mkt = Y - C - I
    euler = 1/ C - beta * (1 + r(+1)) * (1/C(+1) )
    walras = C + K - (1 + r) * K(-1) - w * N
    return goods_mkt, euler, walras


@simple
def AR1(A, epsilon_Z, rho, epsilon_news):
    epsilon_Z_target = (A).apply(np.log) - rho * (A(-1)).apply(np.log) - epsilon_Z - epsilon_news(-4)
    return  epsilon_Z_target

# %%
rbc_base = create_model([firm, household, mkt_clearing, AR1], name = "RBC_v1")

#drawdag(rbc, inputs, unknowns, targets)

# %% Solving for SS
calibration = {"epsilon_Z":0., "epsilon_news":0., "A": 1., "r": r, "eta": eta,
               "delta": delta, "alpha": alpha, "beta": beta, "b": b, "rho": rho}
unknowns_ss = {"K": 14., "N": 1.}
targets_ss = {"goods_mkt": 0., "euler": 0.}

ss = rbc_base.solve_steady_state(calibration, unknowns_ss, targets_ss, solver="hybr")
#%%

# %%
unknowns = ['K', 'N', 'A']
targets = ['euler', 'goods_mkt','epsilon_Z_target']
inputs = ['epsilon_Z', 'epsilon_news']

jacob = rbc_base.solve_jacobian(ss, unknowns, targets, inputs, T = 300) 


# %%
print(f"Goods market clearing: {ss['goods_mkt']}, Euler equation: {ss['euler']}, Walras: {ss['walras']}")
print(f"Capital: {ss['K']}, Output: {ss['Y']}, Investment: {ss['I']}, Consumption: {ss['C']}")
print(f"Interest rate: {ss['r']}, Wage rate: {ss['w']}, Productivity: {ss['A']}")
print(f"N: {ss['N']}")

#%%
T, impact, rho = 300, 0.01, 0.9
dA_news = np.zeros((T, 2))
dA_news[0, 0] = np.exp(sigma) -1
dA_news[:, 1] = np.exp(impact*ss['A']*rho**np.arange(T))-1

# %% consumption response to TFP shock
dC = 100 * jacob['C']['epsilon_news'] @ dA_news / ss['C']
# IRF for output, employment, consumption, investment, capital, the wage rate, and the interest rate 
# shock to production
dY = 100 * jacob['Y']['epsilon_news'] @ dA_news / ss['Y']
# shock to employment
dN = 100 * jacob['N']['epsilon_news'] @ dA_news / ss['N']
# shock to interest rate
dI = 100 * jacob['I']['epsilon_news'] @ dA_news / ss['I']
# shock to capital
dK = 100 * jacob['K']['epsilon_news'] @ dA_news / ss['K']
# shock to wage rate
dw = 100 * jacob['w']['epsilon_news'] @ dA_news / ss['w']
# shock to interest rate
dr = 100 * jacob['r']['epsilon_news'] @ dA_news / ss['r']
#TFP shock  
dA = jacob['A']['epsilon_news'] @ dA_news / ss['A']

# %%
# combine all the IRFs into 8 plots in one graph
fig, ax = plt.subplots(2, 4, figsize=(15, 8))
# add subplot for consumption
#ax[0, 0].plot(dC[:50, 0], label='iid shock', linewidth=2.5, color = 'red')
ax[0, 0].plot(dC[:50,0], linewidth=2.5)   
ax[0, 0].set_title(r'Consumption')
ax[0, 0].set_ylabel('Percentage')

# add subplot for output
#ax[0, 1].plot(dY[:50, 0], label='iid shock', linewidth=2.5, color = 'red')
ax[0, 1].plot(dY[:50,0], linewidth=2.5)
ax[0, 1].set_title(r'Output')

# add subplot for employment
#ax[0, 2].plot(dN[:50, 0], label='iid shock', linewidth=2.5, color = 'red')
ax[0, 2].plot(dN[:50,0], linewidth=2.5)
ax[0, 2].set_title(r'Employment')

# add subplot for investment
#ax[0, 3].plot(dI[:50, 0], label='iid shock', linewidth=2.5, color = 'red')
ax[0, 3].plot(dI[:50,0], linewidth=2.5)
ax[0, 3].set_title(r'Investment')

# add subplot for capital
#ax[1, 0].plot(dK[:50, 0], label='iid shock', linewidth=2.5, color = 'red')
ax[1, 0].plot(dK[:50,0],  linewidth=2.5)
ax[1, 0].set_title(r'Capital')
ax[1, 0].set_xlabel(r'quarters')
ax[1, 0].set_ylabel('Percentage')

# add subplot for wage rate
#ax[1, 1].plot(dw[:50, 0], label='iid shock', linewidth=2.5, color = 'red')
ax[1, 1].plot(dw[:50,0], linewidth=2.5)
ax[1, 1].set_title(r'Wage')
ax[1, 1].set_xlabel(r'quarters')

# add subplot for interest rate
#ax[1, 2].plot(dr[:50, 0], label='iid shock', linewidth=2.5, color = 'red')
ax[1, 2].plot(dr[:50,0], linewidth=2.5)
ax[1, 2].set_title(r'Interest Rate')
ax[1, 2].set_xlabel(r'quarters')

# add subplot for shock
#ax[1, 3].plot(dZ[:50, 0], label='iid shock', linewidth=2.5, color = 'red')
ax[1, 3].plot(dA[:50,0],  linewidth=2.5)
ax[1, 3].legend()
ax[1, 3].set_title(r'TFP shock')
ax[1, 3].set_xlabel(r'quarters')

plt.tight_layout()
fig.savefig('IRFs2a.pdf')



#%% Question 2b) - adjusted RBC model

# Define parameters
eta = 1
alpha = 0.33
rho = 0.9
sigma = 0.01
beta = 0.99 
b = 6.57 
delta = 0.02
phi = 1.0
kappa = 0.75
#%% Set up model

#%%
@simple
def firm(K, L, Z, alpha, delta):
    r = alpha * Z * (K(-1) / L) ** (alpha-1) - delta
    w = (1 - alpha) * Z * (K(-1) / L) ** alpha
    Y = Z * K(-1) ** alpha * L ** (1 - alpha)
    return r, w, Y

@simple
def household(L, w, eta, b):
    lbda = (b * L ** (eta) / w)
    return lbda

@simple
def mkt_clearing(r, C, Y, I, K, beta, lbda,q):
    goods_mkt = Y - C - I
    q_eq = q - 1/(1-phi*(I/K(-1) - delta))
    euler = (beta * lbda(+1)/lbda *((r(+1) + delta) + q(+1)*(1 - delta + phi*(I(+1)/K - delta)*I(+1)/K - phi/2 * (I(+1)/K - delta)**2 )))/q -1
    consumption = lbda - 1/(C - kappa*C(-1)) + beta*kappa/(C(+1) - kappa * C)
    inv_eq = K - I - K(-1)*(1-delta) + phi/2 * K(-1)* (I/K(-1) - delta)**2
    return goods_mkt, euler, consumption, inv_eq, q_eq

@simple
def shock(Z, epsilon_Z, epsilon_news):
    epsilon_Z_target = (Z).apply(np.log) - 0.9 * (Z(-1)).apply(np.log) - epsilon_Z - epsilon_news(-4)
    return  epsilon_Z_target


rbc2 = create_model([household, firm, mkt_clearing, shock], name="RBC2")


#drawdag(rbc, inputs, unknowns, targets)

# %% Solving for SS

calibration2 = {"epsilon_Z":0., "epsilon_news":0., "Z": 1., "r": r, "eta": eta,
               "delta": delta, "alpha": alpha, "beta": beta, "b": b, 
               "phi": phi, "kappa": kappa}
unknowns_ss2 = {"K": 14., "L": 1., "C": 1., "I": 0.5, "q": 1.}
targets_ss2 = {"goods_mkt": 0., "euler": 0., "consumption": 0., "inv_eq": 0., "q_eq": 0.}

ss2 = rbc2.solve_steady_state(calibration2, unknowns_ss2, targets_ss2, solver="hybr")

for key, value in ss2.items():
    print(key, value)
#%%


unknowns2 = ['K', 'L', 'Z', 'C', 'I', 'q']
targets2 = ['euler', 'goods_mkt', 'epsilon_Z_target', 'consumption', 'inv_eq', 'q_eq']
inputs2 = ['epsilon_Z', 'epsilon_news']


jacob = rbc2.solve_jacobian(ss2, unknowns2, targets2, inputs2, T=300)

#%%
T, impact, rho = 300, 0.01, 0.9
dA_news = np.zeros((T, 2))
dA_news[0, 0] = np.exp(sigma) -1
dA_news[:, 1] = np.exp(impact*ss2['Z']*rho**np.arange(T))-1

# %% consumption response to TFP shock
dC = 100 * jacob['C']['epsilon_news'] @ dA_news / ss2['C']
# IRF for output, employment, consumption, investment, capital, the wage rate, and the interest rate 
# shock to production
dY = 100 * jacob['Y']['epsilon_news'] @ dA_news / ss2['Y']
# shock to employment
dN = 100 * jacob['L']['epsilon_news'] @ dA_news / ss2['L']
# shock to interest rate
dI = 100 * jacob['I']['epsilon_news'] @ dA_news / ss2['I']
# shock to capital
dK = 100 * jacob['K']['epsilon_news'] @ dA_news / ss2['K']
# shock to wage rate
dw = 100 * jacob['w']['epsilon_news'] @ dA_news / ss2['w']
# shock to interest rate
dr = 100 * jacob['r']['epsilon_news'] @ dA_news / ss2['r']
#TFP shock  
dA = jacob['Z']['epsilon_news'] @ dA_news / ss2['Z']

# %%
# combine all the IRFs into 8 plots in one graph
fig, ax = plt.subplots(2, 4, figsize=(15, 8))
# add subplot for consumption
#ax[0, 0].plot(dC[:50, 0], label='iid shock', linewidth=2.5, color = 'red')
ax[0, 0].plot(dC[:50,0], linewidth=2.5)   
ax[0, 0].set_title(r'Consumption')
ax[0, 0].set_ylabel('Percentage')

# add subplot for output
#ax[0, 1].plot(dY[:50, 0], label='iid shock', linewidth=2.5, color = 'red')
ax[0, 1].plot(dY[:50,0], linewidth=2.5)
ax[0, 1].set_title(r'Output')

# add subplot for employment
#ax[0, 2].plot(dN[:50, 0], label='iid shock', linewidth=2.5, color = 'red')
ax[0, 2].plot(dN[:50,0], linewidth=2.5)
ax[0, 2].set_title(r'Employment')

# add subplot for investment
#ax[0, 3].plot(dI[:50, 0], label='iid shock', linewidth=2.5, color = 'red')
ax[0, 3].plot(dI[:50,0], linewidth=2.5)
ax[0, 3].set_title(r'Investment')

# add subplot for capital
#ax[1, 0].plot(dK[:50, 0], label='iid shock', linewidth=2.5, color = 'red')
ax[1, 0].plot(dK[:50,0],  linewidth=2.5)
ax[1, 0].set_title(r'Capital')
ax[1, 0].set_xlabel(r'quarters')
ax[1, 0].set_ylabel('Percentage')

# add subplot for wage rate
#ax[1, 1].plot(dw[:50, 0], label='iid shock', linewidth=2.5, color = 'red')
ax[1, 1].plot(dw[:50,0], linewidth=2.5)
ax[1, 1].set_title(r'Wage')
ax[1, 1].set_xlabel(r'quarters')

# add subplot for interest rate
#ax[1, 2].plot(dr[:50, 0], label='iid shock', linewidth=2.5, color = 'red')
ax[1, 2].plot(dr[:50,0], linewidth=2.5)
ax[1, 2].set_title(r'Interest Rate')
ax[1, 2].set_xlabel(r'quarters')

# add subplot for shock
#ax[1, 3].plot(dZ[:50, 0], label='iid shock', linewidth=2.5, color = 'red')
ax[1, 3].plot(dA[:50,0],  linewidth=2.5)
ax[1, 3].legend()
ax[1, 3].set_title(r'TFP shock')
ax[1, 3].set_xlabel(r'quarters')

plt.tight_layout()
fig.savefig('IRFs2b.pdf')
