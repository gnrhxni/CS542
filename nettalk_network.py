
import sys
import logging

logging.basicConfig(
    level=logging.CRITICAL,
    format=r"[%(levelname)s %(created)f]: %(message)s"
)

import cPickle as pickle
from itertools import izip, tee
from optparse import make_option, OptionParser

from pybrain.structure.modules import SigmoidLayer, LinearLayer, BiasUnit
from pybrain.structure.networks.feedforward import FeedForwardNetwork
from pybrain.structure import FullConnection
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
                help="Where to save the trained network"),
    make_option('-L', '--load_from', default=None, action='store', 
                dest="load",
                help="Where load a saved network")
]

def handle_cli():
    return OptionParser(option_list=OPTIONS,
                        description=HELP).parse_args()

def build_that_network(opts, networkshape):

    if opts.load:
        with open(opts.load, 'rb') as pickle_file:
            params = pickle.load(pickle_file)
        (inlayer, hiddenlayer, outlayer, biasunit,
         in2hidden, hidden2out, bias2hidden, bias2out) = params
    else:
        inlayer = LinearLayer(networkshape[0], name='in')
        hiddenlayer = SigmoidLayer(networkshape[1], name='hidden')
        outlayer = SigmoidLayer(networkshape[2], name='out')
        biasunit = BiasUnit(name='bias')

        in2hidden = FullConnection(inlayer, hiddenlayer)
        hidden2out = FullConnection(hiddenlayer, outlayer)
        bias2hidden = FullConnection(biasunit, hiddenlayer)
        bias2out = FullConnection(biasunit, outlayer)

        params = (inlayer, hiddenlayer, outlayer, biasunit,
                  in2hidden, hidden2out, bias2hidden, bias2out) 

    that_network = FeedForwardNetwork()
    that_network.addInputModule(inlayer)
    that_network.addModule(hiddenlayer)
    that_network.addOutputModule(outlayer)
    that_network.addModule(biasunit)
    that_network.addConnection(in2hidden)
    that_network.addConnection(hidden2out)
    that_network.addConnection(bias2hidden)
    that_network.addConnection(bias2out)
    that_network.sortModules()

    return that_network, params


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
        len(nd.letterToPos)*opts.windowsize,  # input
        opts.nhidden,                         # hidden
        nd.NUMOUTPUTS,                        # output
    ]

    logging.debug("Constructing network of shape %s", networkshape)
    network, to_save = build_that_network(opts, networkshape)
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
        logging.info("Running pass %d", pass_number)

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
        with open(opts.save_to, 'wb') as save_file:
            pickle.dump(to_save, save_file)


if __name__ == '__main__':
    ret = main()
    sys.exit(ret)