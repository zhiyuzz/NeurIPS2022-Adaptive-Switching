import numpy as np
from scipy import special


class Ours:
    # Algorithm 1 in our paper
    def __init__(self, G, lam, C):
        self.t = 1
        self.C = C
        self.G = G
        self.alpha = 2 + 4 * lam / G
        self.S = 0

    # Define the potential function
    def __potential(self, z):
        return self.C * np.sqrt(self.alpha * self.t) * (np.sqrt(np.pi) * z * special.erfi(z) - np.exp(z ** 2))

    def get_prediction(self):
        Z_high = (self.S + 1) / np.sqrt(4 * self.alpha * self.t)
        Z_low = (self.S - 1) / np.sqrt(4 * self.alpha * self.t)
        prediction = (self.__potential(Z_high) - self.__potential(Z_low)) / 2   # Discrete derivative
        return prediction

    def update(self, gt):
        self.S -= gt / self.G
        self.t += 1


class Baseline:
    # Algorithm 1 of [ZCP22a], "Adversarial Tracking Control via Strongly Adaptive Online Learning with Memory"
    # Version here is surveyed as Algorithm 2 in Appendix A of our paper
    def __init__(self, G, lam, C):

        # Problem constants
        self.lam = lam  # lambda
        self.K = lam + G

        # Initialize internal variables; for the "current time" in the main loop
        self.t = 1  # time
        self.wealth_past = C * self.K
        self.beta_raw = 0
        self.beta = 0
        self.prediction = 0

        # Initialize other interval variables; for the "next time" in the loop
        self.beta_next_raw = 0
        self.beta_next = 0
        self.wealth = 0

    def get_prediction(self):
        return self.prediction

    def update(self, gt):

        # Line 4, 5, 6 - compute the next betting fraction
        self.beta_next_raw = (1 - 1 / self.t) * self.beta_raw - gt / (2 * self.t * self.K ** 2)
        bound = 1 / (self.K * np.sqrt(2 * self.t))
        if self.beta_next_raw < - bound:
            self.beta_next = - bound
        elif self.beta_next_raw > bound:
            self.beta_next = bound
        else:
            self.beta_next = self.beta_next_raw

        # Line 7 - compute the wealth
        wealth_temp = (1 - (gt + self.lam) * self.beta) * self.wealth_past / (1 - self.lam * self.beta_next)
        if self.prediction >= self.beta_next * wealth_temp:
            self.wealth = wealth_temp
        else:
            self.wealth = (1 - (gt - self.lam) * self.beta) * self.wealth_past / (1 + self.lam * self.beta_next)

        # Update the clock
        self.t += 1
        self.wealth_past = self.wealth
        self.beta_raw = self.beta_next_raw
        self.beta = self.beta_next
        self.prediction = self.beta * self.wealth_past

