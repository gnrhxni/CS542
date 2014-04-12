# -*- mode: ruby -*-

VAGRANTFILE_API_VERSION = "2"

Vagrant.configure(VAGRANTFILE_API_VERSION) do |config|

  config.vm.box     = "ubuntu_13.04"
  config.vm.box_url = "http://cloud-images.ubuntu.com/vagrant/raring/current/raring-server-cloudimg-amd64-vagrant-disk1.box"

  config.vm.provider :virtualbox do |vb|
    vb.customize ["modifyvm", :id, "--ioapic", "on" ]
    vb.customize ["modifyvm", :id, "--memory", 2048, "--cpus", "2"]
  end

  config.vm.provision :shell, :path => "vagrant_setup.sh"


end
