#!/bin/bash

set -o errexit

mkdir -p conll2009
cd conll2009

# system outputs, from <http://ufal.mff.cuni.cz/conll2009-st/results/results.php>
wget http://ufal.mff.cuni.cz/conll2009-st/eval-data.zip
unzip eval-data.zip
rm -f eval-data.zip

# gold data, from <http://ufal.mff.cuni.cz/conll2009-st/eval-data.html>
mkdir gold && cd gold
wget http://ufal.mff.cuni.cz/conll2009-st/eval-gold/CoNLL2009-ST-Gold-Both_tasks.zip
unzip CoNLL2009-ST-Gold-Both_tasks.zip
rm -f CoNLL2009-ST-Gold-Both_tasks.zip

http://ufal.mff.cuni.cz/conll2009-st/eval-gold/CoNLL2009-ST-Gold-Joint.zip
unzip CoNLL2009-ST-Gold-Joint.zip
rm -f CoNLL2009-ST-Gold-Joint.zip
