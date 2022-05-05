from matplotlib import pyplot as plt
from Algorithm import *

# Problem constants
G = 1   # Lipschitz constant
lam = 1   # Switching cost weight

# Time horizon
T = 2000

# Hyperparameter for the algorithms, C = 1 is the most natural choice that matches the essence of "parameter-freeness"
C = 1

# Setting for repeated experiment
N = 100  # Number of independent runs
rng = np.random.default_rng(2022)   # Initialize the random number generator
all_losses = np.empty([N, 2, T])    # The three dimensions are number of random seeds, number of algorithms and time horizon


# Definition of the adversary; x is a sample from U[-1,1]
def adversary(x):
    # The gradient is the combination of a purely random term, a "trend" and a bias
    return 0.6 * x + 0.3 * np.cos(t * np.pi * 4 / T) - 0.1


for n in range(N):

    # Generate a noise sequence
    random_seq = rng.uniform(-1, 1, T)

    # Create the algorithm, starting from our algorithm
    alg_ours = Ours(G, lam, C)
    losses_ours = np.empty(T)
    prediction = 0

    # Run our algorithm
    for t in range(T):
        # Get prediction
        prev_prediction = prediction
        prediction = alg_ours.get_prediction()

        # Compute cumulative losses
        gt = adversary(random_seq[t])
        losses_ours[t] = prediction * gt + lam * np.abs(prev_prediction - prediction)
        if t != 0:
            losses_ours[t] += losses_ours[t-1]

        # Update
        alg_ours.update(gt)

    all_losses[n, 0, :] = losses_ours

    # Repeat for the baseline
    alg_baseline = Baseline(G, lam, C)
    losses_baseline = np.empty(T)
    prediction = 0

    # Run the baseline
    for t in range(T):
        # Get prediction
        prev_prediction = prediction
        prediction = alg_baseline.get_prediction()

        # Compute cumulative losses
        gt = adversary(random_seq[t])
        losses_baseline[t] = prediction * gt + lam * np.abs(prev_prediction - prediction)
        if t != 0:
            losses_baseline[t] += losses_baseline[t-1]

        # Update
        alg_baseline.update(gt)

    all_losses[n, 1, :] = losses_baseline

mean = np.mean(all_losses, axis=0)
std = np.std(all_losses, axis=0)

plt.figure()
plt.plot(np.arange(1, T + 1), mean[0, :], '-', label="ours")
plt.fill_between(np.arange(1, T + 1), mean[0, :] - std[0, :], mean[0, :] + std[0, :], color='C0', alpha=0.2)
plt.plot(np.arange(1, T + 1), mean[1, :], '-', label="baseline")
plt.fill_between(np.arange(1, T + 1), mean[1, :] - std[1, :], mean[1, :] + std[1, :], color='C1', alpha=0.2)
plt.legend()
