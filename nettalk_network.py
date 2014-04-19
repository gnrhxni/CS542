
import sys
import logging

logging.basicConfig(
    level=logging.CRITICAL,
    format=r"[%(levelname)s %(created)f]: %(message)s"
)

import cPickle as pickle
from itertools import izip, tee
from optparse import make_option, OptionParser

from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import SigmoidLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

import numpy as np

import nettalk_data as nd


HELP="""Runs the network described in the Sejnowski paper and computes its
accuracy.  Accuracy results are printed to stdout """


OPTIONS = [
    make_option('-n', '--hidden_layers', type=int, action='store', 
                dest='nhidden', help="the number of hidden units to use",
                default=80),
    make_option('-l', '--logging', type=str, action='store',
                default='INFO', 
                help="set logging verbosity: DEBUG INFO WARN ERROR, CRITICAL"),
    make_option('-t', '--training_data', type=str, action='store',
                default=nd.topKDatafile),
    make_option('-w', '--windowsize', type=int, action='store', default=7),
    make_option('-N', '--accuracy_interval', type=int,
                default=100, action='store',
                help="print accuracy measurements after "+ 
                "so many training iterations"),
    make_option('-p', '--passes', type=int,
                default=1000, action='store', dest='npasses',
                help="number of passes through the dataset to train"),
    make_option('-s', '--save_to', default=None, action='store', 
                help="Where to save the trained network")

]

def handle_cli():
    return OptionParser(option_list=OPTIONS,
                        description=HELP).parse_args()

def datastreams(opts):
    input_entries, output_entries, debug_entries = tee(
        nd.dictionary(datafile=opts.training_data), 3) 
    inputstream = nd.binarystream(windowsize=opts.windowsize,
                                  input_entries=input_entries)
    outputstream = iter(nd.outputUnits(e) for e in output_entries)

    return inputstream, outputstream, debug_entries


def generate_datasets(opts, networkshape):
    inputstream, outputstream, debug_entries = datastreams(opts)

    for     target,       input_vectors, entry in izip(
            outputstream, inputstream,   debug_entries):
        logging.debug("Generating dataset for %s", entry)
        dataset = SupervisedDataSet(networkshape[0], networkshape[-1])
        for i, input_vector in enumerate(input_vectors):
            dataset.addSample(input_vector, target[i])
        yield dataset

def hits_and_misses(opts, network):
    inputstream, _, raw_entries = datastreams(opts)

    for input_bin, entry in izip(inputstream, raw_entries):
        for     letter_bin, phoneme,        stress in zip(
                input_bin,  entry.phonemes, entry.stress):
            answer = network.activate(letter_bin)
            guessed_phoneme = nd.closestByDotProduct(
                answer[:nd.MINSTRESS], nd.articFeatures)
            guessed_stress = nd.closestByDotProduct(
                answer[nd.MINSTRESS:], nd.stressFeatures)
            yield bool(guessed_stress == stress), \
                  bool(guessed_phoneme == phoneme)
            

def calculate_accuracy(opts, network):
    logging.debug("Calculating accuracy")
    stress_hits, phoneme_hits = zip( *hits_and_misses(opts, network) )
    stress_accuracy = np.mean(stress_hits)
    phoneme_accuracy = np.mean(phoneme_hits)
    logging.debug("Accuracy calculation complete")
    return stress_accuracy, phoneme_accuracy


def main():
    opts, _ = handle_cli()
    logging.getLogger().setLevel(
        getattr(logging, opts.logging.upper())
    )
    logging.debug("Enabled logging")

    networkshape = [ 
        len(nd.letterToPos)*opts.windowsize, # input
        opts.nhidden,                         # hidden
        nd.NUMOUTPUTS,                        # output
    ]
    logging.debug("Constructing network of shape %s", networkshape)
    network = buildNetwork( 
        *networkshape, 
        bias=True, 
        outputbias=True,
        hiddenclass=SigmoidLayer,
        outclass=SigmoidLayer
    )
    logging.debug("Instantiating trainer")
    trainer = BackpropTrainer(
        network, 
        dataset=None, 
        learningrate=1.0,
        batchlearning=True, 
        weightdecay=0.0
    )
    if opts.accuracy_interval:
        logging.debug("Determining base accuracy")
        s_accuracy, p_accuracy = calculate_accuracy(opts, network)
        print '%d\t%.3f\t%.3f' %(0, s_accuracy, p_accuracy)

    i = 0
    for pass_number in range(1, opts.npasses+1):
        logging.debug("Running pass %d", pass_number)

        datasets = generate_datasets(opts, networkshape)
        for dataset in datasets:
            i += 1
            trainer.setData(dataset)
            logging.debug("Training dataset %d", i)
            err = trainer.train()
            logging.debug("Trained to err %f", err)
            
            if opts.accuracy_interval and i % opts.accuracy_interval == 0:
                s_accuracy, p_accuracy = calculate_accuracy(opts, network)
                print '%d\t%.3f\t%.3f' %(i, s_accuracy, p_accuracy)
        
    if opts.save_to:
        pickle.dump(network, open(opts.save_to, 'wb'))

if __name__ == '__main__':
    ret = main()
    sys.exit(ret)
