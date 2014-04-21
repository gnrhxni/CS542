#!/usr/bin/python

from constants import *
import numpy
from pybrain.structure.modules import LinearLayer, SigmoidLayer, BiasUnit 
from pybrain.structure.networks.feedforward import FeedForwardNetwork
from pybrain.structure import FullConnection
from sigmoidsparselayer import SigmoidSparseLayer

def buildModules(inputs=NUMINPUTS, hidden=80, outputs=NUMOUTPUTS, beta=0, sparsity=0.5):
    mod = dict();
    mod['in'] = LinearLayer(inputs, name='in')
    mod['hidden'] = SigmoidSparseLayer(dim=hidden, beta=beta, sparsity=sparsity, name='hidden')
    mod['out'] = SigmoidLayer(outputs, name='out')
    mod['bias'] = BiasUnit(name='bias')
    mod['in_to_hidden'] = FullConnection(mod['in'], mod['hidden'])
    mod['hidden_to_out'] = FullConnection(mod['hidden'], mod['out'])
    mod['bias_to_hidden'] = FullConnection(mod['bias'], mod['hidden'])
    mod['bias_to_out'] = FullConnection(mod['bias'], mod['out'])
    return mod

def buildnet(modules):
    net = FeedForwardNetwork(name='mynet');
    net.addInputModule(modules['in'])
    net.addModule(modules['hidden'])
    net.addOutputModule(modules['out'])
    net.addModule(modules['bias'])
    net.addConnection(modules['in_to_hidden'])
    net.addConnection(modules['hidden_to_out'])
    net.addConnection(modules['bias_to_hidden'])
    net.addConnection(modules['bias_to_out'])
    net.sortModules()
    return net

