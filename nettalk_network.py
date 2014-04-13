
import sys
import logging
from itertools import izip
from optparse import make_option, OptionParser

from pybrain.shortcuts import buildNetwork
from pybrain.structure import SigmoidLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackPropTrainer

import numpy as np

import nettalk_data as nd

HELP="""
Does stuff. Hope you know what you're doing!
"""


OPTIONS = [
    make_option('-n', '--hidden_layers', type=int, action='store', 
                dest='nhidden', help="the number of hidden units to use",
                default=80),
    make_option('-l', '--logging', type=str, action='store',
                default='INFO', 
                help="set logging verbosity: DEBUG INFO WARN ERROR, CRITICAL")
    make_option('-t', '--training_data', type=str, action='store',
                default=nd.topKDatafile),
    make_option('-w', '--window_size', type=int, action='store', default=7)
]


log = logging.getLogger(__name__)


def handle_cli():
    return OptionParser(option_list=OPTIONS,
                        description=HELP).parse_args()

def main():
    
    opts, _ = handle_cli()

    log.setLevel(getattr(logging, opts.logging))
    log.debug("Enabled logging")

    networkshape = [ 
        nd.letterToPos*opts.window_size, # input
        opts.nhidden,                    # hidden
        nd.NUMOUTPUTS,                   # output
    ]
    log.debug("Constructing network of shape %s", networkshape)
    network = buildNetwork( 
        networkshape, 
        bias=True, 
        outputbias=True,
        hiddenclass=SigmoidLayer,
        outclass=SigmoidLayer
    )
    log.debug("Instantiating trainer")
    trainer = BackPropTrainer(
        network, 
        dataset=None, 
        learningrate=1.0,
        batchlearning=True, 
        weightdecay=0.0,
        verbose=True
    )

    entries = nd.dictionary(datafile=opts.training_data)
    inputstream = nd.binarystream(windowsize=opts.windowsize,
                                  input_entries=entries)
    outputstream = iter(outputUnits(e) for e in entries)

    for target, input_vectors in izip(outputstream, inputstream):
        dataset = SupervisedDataSet(networkshape[0], networkshape[-1])
        for input_vector in input_vectors:
            dataset.addSample(input_vector)




if __name__ == '__main__':
    ret = main()
    sys.exit(ret)
