from matplotlib import pyplot as plt
from Algorithm import *

# Problem constants
G = 1   # Lipschitz constant
lam = 1   # Switching cost weight
D = 5   # Number of assets

# Time horizon
T = 3000

# Hyperparameter for the algorithms
C1 = 20
C2 = 20

# Setting for repeated experiment
N = 50  # Number of repeated trials
rng = np.random.default_rng(2022)   # Initialize the random number generator
all_profit = np.empty([N, 2, T])    # The three dimensions are number of random seeds, number of algorithms and time horizon


# Definition of the adversary; each entry of x is a sample from U[-1,1]
def adversary(x):
    # The gradients are combinations of a purely random term, a "trend" and a bias
    a = 0.2 * np.ones(D)
    a[0] = 0.4 * x[0] + 0.4 * np.sin((t / 500) * np.pi)
    a[1] = 0.5 * x[1] + 0.3 * np.sin((t / 500 + 1 / 2) * np.pi)
    a[2] = 0.6 * x[2] + 0.2 * np.sin((t / 500 + 1) * np.pi)
    a[3] = 0.7 * x[3] + 0.1 * np.sin((t / 500 + 3 / 2) * np.pi)
    a[4] = 0.8 * x[4]
    return a


for n in range(N):

    # Generate a noise sequence
    random_seq = rng.uniform(-1, 1, [D, T])

    # Create the list of base algorithms, starting from our Algorithm 1
    alg_ours = []
    for d in range(D):
        alg_ours.append(Ours(G, lam, C1 / 5))
    profit_ours = np.empty(T)
    prediction = np.zeros(D)

    # Run our algorithm
    for t in range(T):
        # Get prediction
        prev_prediction = prediction
        for d in range(D):
            prediction[d] = alg_ours[d].get_prediction()

        # Compute cumulative losses
        gt = adversary(random_seq[:, t])
        profit_ours[t] = prediction @ gt - lam * np.sum(np.abs(prev_prediction - prediction))
        if t != 0:
            profit_ours[t] += profit_ours[t - 1]

        # Update
        for d in range(D):
            alg_ours[d].update(-gt[d])

    all_profit[n, 0, :] = profit_ours

    # Repeat for the baseline
    alg_baseline = []
    for d in range(D):
        alg_baseline.append(Baseline(G, lam, C2 / 5))
    profit_baseline = np.empty(T)
    prediction = np.zeros(D)

    # Run our algorithm
    for t in range(T):
        # Get prediction
        prev_prediction = prediction
        for d in range(D):
            prediction[d] = alg_baseline[d].get_prediction()

        # Compute cumulative losses
        gt = adversary(random_seq[:, t])
        profit_baseline[t] = prediction @ gt - lam * np.sum(np.abs(prev_prediction - prediction))
        if t != 0:
            profit_baseline[t] += profit_baseline[t - 1]

        # Update
        for d in range(D):
            alg_baseline[d].update(-gt[d])

    all_profit[n, 1, :] = profit_baseline


mean = np.mean(all_profit, axis=0)
std = np.std(all_profit, axis=0)

plt.figure()
plt.rcParams.update({'font.size': 14})
plt.plot(np.arange(1, T + 1), mean[0, :], '-', label=r"Ours, $C=20$, $\lambda=1$")
plt.fill_between(np.arange(1, T + 1), mean[0, :] - std[0, :], mean[0, :] + std[0, :], color='C0', alpha=0.2)
plt.plot(np.arange(1, T + 1), mean[1, :], '-', label=r"Baseline, $C=20$, $\lambda=1$")
plt.fill_between(np.arange(1, T + 1), mean[1, :] - std[1, :], mean[1, :] + std[1, :], color='C1', alpha=0.2)
plt.xlabel('t')
plt.ylabel('Cumulative return')
plt.legend(loc='upper left')

plt.savefig("Figures/Synthetic_tuneC_2.pdf", bbox_inches='tight')
