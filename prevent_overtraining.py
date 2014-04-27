
from random import shuffle

from pybrain.supervised.trainers.backprop import BackpropTrainer

class PreventOverTrainer(BackpropTrainer):
    def train(self):
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
        
        #print("Total error:", errors)
	if errors > 0.1:
            self.module._setParameters(self.descent(self.module.derivs))
        self.epoch += 1
        self.totalepochs += 1
        return errors / ponderation
