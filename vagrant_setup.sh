#!/bin/bash

sudo apt-get -y update
sudo apt-get -y upgrade

sudo apt-get -y install build-essential python \
    python-dev python-virtualenv \
    python-distribute python-pip \
    python-numpy python-scipy \
    python-matplotlib \
    git language-pack-en

sudo easy_install -U distribute

cd ~vagrant

function vagrantdo() {
    sudo -iu vagrant bash -c "$1";
}
vagrantdo 'git clone https://github.com/gnrhxni/CS542.git /home/vagrant/nettalk'

sudo pip install -r /home/vagrant/nettalk/requirements.txt
