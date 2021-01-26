#!/bin/bash

set -o errexit

# main website:
# <http://multiword.sourceforge.net/PHITE.php?sitesig=CONF&page=CONF_04_LAW-MWE-CxG_2018___lb__COLING__rb__&subpage=CONF_40_Shared_Task>

mkdir -p parseme2018
cd parseme2018

# from <https://gitlab.com/parseme/sharedtask-data/tree/master/1.1>
wget -O sharedtask-data-master.tar.gz https://gitlab.com/parseme/sharedtask-data/-/archive/master/sharedtask-data-master.tar.gz?path=1.1
tar xzf sharedtask-data-master.tar.gz
rm -f sharedtask-data-master.tar.gz
