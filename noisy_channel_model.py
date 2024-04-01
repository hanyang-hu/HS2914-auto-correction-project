from textdistance import damerau_levenshtein
import math

"""
Implementation of the Noisy Channel Model.
"""
class NoisyChannelModel():
    def __init__(self, candidates, priors, surface_word):
        self.candidates = candidates
        self.log_priors = {candidate: math.log(prior) for candidate, prior in priors.items()}
        self.surface_word = surface_word

    """
    Given the candidate, the likelihood is the inverse of the Damerau-Levenshtein distance from the surface word.
    We subtract a small constant (< 1) to prefer candidates with lower edit distance.
    """
    def log_likelihood(self, candidate):
        alpha = 3 # preference for the original word compared to those with edit distance 1
        gamma = 7 # the higher the gamma >= 1, the more the model prefers the candidates with higher prior probability (i.e. lower edit distance)
        if candidate == self.surface_word:
            return alpha # more preference for the original word
        return -math.log(damerau_levenshtein(candidate, self.surface_word)) * gamma

    """
    Using Bayes' rule to compute the posterior probability.
    Return the MAP estimatation.
    """
    def correct(self):
        best_candidate, best_log_posterior = None, float('-inf')
        for candidate in self.candidates:
            log_posterior = self.log_likelihood(candidate) + self.log_priors[candidate]
            if log_posterior > best_log_posterior:
                best_candidate, best_log_posterior = candidate, log_posterior
        return best_candidate
