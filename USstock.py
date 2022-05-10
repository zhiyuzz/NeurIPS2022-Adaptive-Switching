from matplotlib import pyplot as plt
from Algorithm import *
import glob

dates = np.loadtxt("Data/AAPL_data.csv", dtype=np.datetime64, delimiter=',', skiprows=1, usecols=0)

data_temp = []
for file_name in glob.glob('Data/'+'*.csv'):
    data_temp.append(np.loadtxt(file_name, delimiter=',', skiprows=1, usecols=4))
closing_price = np.stack(data_temp)
price_diff = np.diff(closing_price)
D, T = price_diff.shape

# Compute the largest gain / loss for any stock on any day; round it up and assign it to G
G = np.ceil(np.amax(np.abs(price_diff)))
lam = 0.1     # Weight of the transaction cost

C1 = 1   # Hyperparameter for the algorithms
C2 = 200

# Create the list of base algorithms, starting from our Algorithm 1
alg_ours = []
for d in range(D):
    alg_ours.append(Ours(G, lam, C1 / 5))
profit_ours = np.empty(T)
invest_ours = np.empty(T)
pred_ours = np.zeros(D)

# Run our algorithm
for t in range(T):
    # Get prediction
    prev_pred = pred_ours.copy()
    for d in range(D):
        pred_ours[d] = alg_ours[d].get_prediction()

    print(pred_ours-prev_pred)
    # Compute cumulative losses
    gt = price_diff[:, t]
    profit_ours[t] = pred_ours @ gt - lam * np.sum(np.abs(prev_pred - pred_ours))
    if t != 0:
        profit_ours[t] += profit_ours[t - 1]

    # Compute the amount of new investment
    invest_ours[t] = (pred_ours - prev_pred) @ closing_price[:, t]

    # Update
    for d in range(D):
        alg_ours[d].update(-gt[d])

# Create the list of base algorithms, starting from our Algorithm 1
alg_baseline = []
for d in range(D):
    alg_baseline.append(Baseline(G, lam, C2 / 5))
profit_baseline = np.empty(T)
invest_baseline = np.empty(T)
pred_baseline = np.zeros(D)

# Run our algorithm
for t in range(T):
    # Get prediction
    prev_pred = pred_baseline
    for d in range(D):
        pred_baseline[d] = alg_baseline[d].get_prediction()

    # Compute cumulative losses
    gt = price_diff[:, t]
    profit_baseline[t] = pred_baseline @ gt - lam * np.sum(np.abs(prev_pred - pred_baseline))
    if t != 0:
        profit_baseline[t] += profit_baseline[t - 1]

    # Compute the amount of new investment
    invest_baseline[t] = (pred_baseline - prev_pred) @ closing_price[:, t]

    # Update
    for d in range(D):
        alg_baseline[d].update(-gt[d])

plt.figure()
plt.rcParams.update({'font.size': 14})
plt.plot(dates[1:], profit_ours, '-', label=r"Ours, $C=1$, $\lambda=0.1$")
plt.plot(dates[1:], profit_baseline, '-', label=r"Baseline, $C=200$, $\lambda=0.1$")
plt.xlabel('Date')
plt.ylabel('Cumulative return')
plt.legend(loc='upper left')

plt.savefig("Figures/US_stock.pdf", bbox_inches='tight')

plt.figure()
plt.rcParams.update({'font.size': 14})
plt.plot(dates[:-1], invest_ours, '-', label=r"Ours, $C=1$, $\lambda=0.1$")
plt.plot(dates[:-1], invest_baseline, '-', label=r"Baseline, $C=200$, $\lambda=0.1$")
plt.xlabel('Date')
plt.ylabel('Cumulative return')
plt.legend(loc='upper left')
