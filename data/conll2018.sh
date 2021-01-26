#!/bin/bash

set -o errexit

# from <https://universaldependencies.org/conll18/>
mkdir -p conll2018
cd conll2018

curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2885{/conll2018-test-runs.tgz}
tar xzf conll2018-test-runs.tgz
rm -f conll2018-test-runs.tgz
