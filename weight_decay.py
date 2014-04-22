
from random import shuffle

from pybrain.supervised.trainers.backprop import BackpropTrainer

class WeightDecayTrainer(BackpropTrainer):
    def train(self):
        """Just like the BackpropTrainer, but the weight decay is added after
        each epoch, not after every sample.
        Useful in our case since we're training after every word.
        """
        assert len(self.ds) > 0, "Dataset cannot be empty."
        self.module.resetDerivatives()
        errors = 0
        ponderation = 0.
        shuffledSequences = []
        for seq in self.ds._provideSequences():
            shuffledSequences.append(seq)
        shuffle(shuffledSequences)
        for seq in shuffledSequences:
            e, p = self._calcDerivs(seq)
            errors += e
            ponderation += p

        if self.verbose:
            print("Total error:", errors / ponderation)
        newderivs = self.module.derivs - (self.weightdecay*self.module.params)
        self.module._setParameters(self.descent(newderivs))
        self.epoch += 1
        self.totalepochs += 1
        return errors / ponderation
