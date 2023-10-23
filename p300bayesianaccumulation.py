import numpy as np


class P300BayesianAcc:
    """Classification of characters of the P300-speller,
    using a cumulative-trial MAP based classifier, processing target and
    non-target responses to the flash and giving a probability after each flash.

    Parameters
    ----------
    n_characters : int, (default 36)
        The number of characters flashed in the interface.
    class_prior : list, (default None)
        The prior probability on characters.
    """

    def __init__(self, n_characters, character_prior=None):
        self.n_characters = n_characters

        self.reset(character_prior)

    def reset(self, character_prior=None):
        """Reset the probabilities of characters.
        It must be called at the beginning of each group of flashes.

        Parameters
        ----------
        class_prior : list, (default None)
            The prior probability on characters.

        Returns
        -------
        self : P300BayesianAcc instance
            The P300BayesianAcc instance.
        """
        if character_prior is None:
            self.character_prior = np.ones(self.n_characters)
        else:
            if len(character_prior) != self.n_characters:
                raise ValueError(
                    "Length of character_prior is different from n_characters"
                )
            self.character_prior = np.asarray(character_prior)
        self.character_prior /= self.character_prior.sum()

        self.character_proba = self.character_prior.copy()

        return self

    def predict_proba(self, erp_likelihood, character_flash):
        """Predict probability of each character after a new trial.

        Parameters
        ----------
        erp_likelihood : array-like, shape (2,)
            array-like containing the likelihoods of ERP (target and non-target
            responses) on this new trial.
        character_flash : array-like, shape (n_characters,)
            array-like containing 1 is the character has been flashed during
            this trial, 0 otherwise.

        Returns
        -------
        character_proba : ndarray, shape (n_characters,)
            probability for each character cumulated across trials.
        """
        if len(erp_likelihood) != 2:
            raise ValueError("erp_likelihood must contain 2 values")

        for c in range(self.n_characters):
            if character_flash[c] == 1:
                # character c has been flashed: likelihood is the distance to
                # the target response
                likelihood = erp_likelihood[1]
            elif character_flash[c] == 0:
                # character c has not been flashed: likelihood is the distance
                # to the non-target response
                likelihood = erp_likelihood[0]
            else:
                raise ValueError("character_flash must contain only binary numbers")

            self.character_proba[c] *= likelihood

        sum = self.character_proba.sum()
        if sum == 0:
            # Just in case (avoid division by 0)
            sum = self.n_characters
            self.reset()

        character_proba = self.character_proba / sum

        return character_proba


class P300_Scorer:
    def __init__(self, symbols, c2l, l2c):
        self.symbols = symbols
        self.c2l = c2l
        self.l2c = l2c
        self.reset()

    def reset(self):
        self.scores = np.zeros(len(self.symbols))

    def get_proba(self):
        return self.scores

    def predict(self):
        return self.c2l(self.scores.argmax())

    def update(self, proba, group):
        pass


class Scorer_DA(P300_Scorer):
    """pMDM+BA, but with a direct accumulation instead of a Bayesian one."""

    def update(self, proba, group):
        for char in group:
            self.scores[self.l2c(char)] += proba[1]


class Scorer_BA(P300_Scorer):
    """pMDM+BA classifier for ERP datasets, based on Bayesian accumulation (BA)"""

    def reset(self):
        self.scores = np.zeros(len(self.symbols))
        self.scorer = P300BayesianAcc(len(self.symbols))

    def update(self, proba, group):
        mask = np.zeros(len(self.symbols))
        for char in group:
            mask[self.l2c(char)] = 1
        self.scores = self.scorer.predict_proba(proba, mask)


class Scorer_OM(P300_Scorer):
    """MDM+OM classifier for ERP, based on occurrences maximizing (OM)"""

    def update(self, proba, group):
        if proba[1] > proba[0]:
            for char in group:
                self.scores[self.l2c(char)] += 1
