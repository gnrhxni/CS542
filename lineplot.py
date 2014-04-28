#!/usr/bin/env python

import sys
import itertools
from optparse import OptionParser, make_option

import numpy as np
import matplotlib.pyplot as plt


markers = ['.', 'o', 'v', '2', 'o', '<', '>', 'v']
colors  = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

# symbols = itertools.cycle(iter(
#     ''.join(tuple(items))
#     for items in itertools.product(markers,colors)
# ))

symbols = itertools.cycle(colors)

OPTIONS = [
    make_option('-L', '--legend', type=str, action='store',
                default="", 
                help="comma separated list of legend names"),
    make_option('-X', '--xlabel', type=str, action='store'),
    make_option('-Y', '--ylabel', type=str, action='store'),
    make_option('-T', '--title', type=str, action='store'),
    make_option('-o', '--save', type=str, action='store')
]


def plot(data, xlabel=None, ylabel=None, legend=list(), title=None, save=""):
    args = list()
    x = data[:,0]
    for axis_y in xrange(1, data.shape[1]):
        y = data[:,axis_y]
        symbol = symbols.next()
        args.extend([x, y, symbol])
        
    plt.plot(*args)
    plt.figure(num=1, figsize=(20,15))
    if xlabel:
        plt.xlabel(xlabel, fontsize=14)
    if ylabel:
        plt.ylabel(ylabel, fontsize=14)
    if legend:
        aucs = np.sum(data[:,1:], axis=0)
        ideal = data.shape[0]
        assert len(aucs) == len(legend)
        legend = [ "%s (%0.2f/%i)" %(l, auc, ideal)
                   for l, auc in zip(legend, aucs) ]
        plt.legend(legend, loc="best")#, fontsize=10)
    if title:
        plt.title(title)
    plt.grid(True)
    if save:
        plt.savefig(save)
    else:
        plt.show()


def load_data(datafname_list):
    data = np.loadtxt(datafname_list.pop(0))
    for filename in datafname_list:
        data = np.hstack(
            (data,
             np.loadtxt(filename)[:,1:])
        )
    return data


def main():
    opts, args = OptionParser(option_list=OPTIONS).parse_args()

    if not args:
        print >> sys.stderr, "Need some arguments, please"
        return 1

    if opts.legend:
        opts.legend = opts.legend.split(',')
    else:
        opts.legend = list()

    data = load_data(args)

    plot(data, **opts.__dict__)

    return 0

if __name__ == '__main__':
    ret = main()
    sys.exit(ret)


