# Error Analysis and the Role of Morphology

This code accompanies the following research paper:

+ Marcel Bollmann and Anders Søgaard (2021).  Error Analysis and the Role of
  Morphology.  Accepted at EACL 2021.


## Prerequisites

+ Python 3.6 or higher

+ Install dependencies via `pip install -r requirements.txt`

+ Install dependencies for modified `pyconll` library, included within this
  repo, via `pip install -r requirements_pyconll.txt`

+ Trained UDPipe models from: <http://hdl.handle.net/11234/1-2998>
  - Edit `config.yaml` with the path to these files

Generally, all Python scripts can be run with the `-h` or `--help` flag to get
more usage information and a detailed list of options.


## How-to

1. **Download data files** via scripts in the `data/` directory.  For example,
   `data/wmt19-qe.sh` will download the dataset from the WMT19 quality
   estimation shared task.

2. **Extract morphological features** from the dataset.  This is done via the
   corresponding scripts in the `scripts/` directory.

   For example, `scripts/process_wmt19_qe.py` will process the downloaded
   WMT19-QE dataset.

   Concretely, processing will find applicable UDPipe models (that you should
   have downloaded and placed in the path given in `config.yaml`) and use them
   to run a morphological tagger.  The output files will be written to
   `processed/<dataset>/` in CoNLL-U format.  Annotations about errors in the
   dataset are added to the last (`misc`) column of the file.

3. **Add token frequency features** by calling `scripts/add_frequencies.py`.

4. **Run the analysis** by calling `scripts/analyze.py` with the desired
   processed file in `processed/` as an argument.

   This will compute a bunch of scores on the data, including correlation
   coefficients and feature importances of a random forest classifier.  The
   output is in CSV format.

   Some notes about how the script was invoked for the experiments reported in the paper:

   - Use the flags `-n -I -L -w 0 --method drop-category-upos`
   - The control setting (without the morphological features) is run by adding `-M`
   - The log file (produced by `--log filename.log`) is used to store classifier
     performance (like F1-score), while the output of the script contains the
     feature importance scores.
   - A concrete example of how to run the full analysis on all files can be seen
     in `scripts/batch_run_analysis.sh`.

   (`scripts/analyze_ss.py` contains an older version of the analysis that
   computes stability selection scores with randomized logistic regression.
   Note that this needs additional requirements, most notably the [stability
   selection module from
   scikit-learn-contrib](https://github.com/scikit-learn-contrib/stability-selection).)

### Generate further stats and analyses

- **Extract classifier performance from log files** (produced by step 4 above)
  and output it in CSV format via `scripts/parse_logfiles.py`.

- **Add feature frequency information** to the analyzed CSV files via
  `scripts/add_feature_freqs.py`, invoked with the same flags as the analysis
  script in step 4 above.  (This is a separate script for historical reasons
  only...)

- **Calculate corpus-level stats** such as type-token ratio, frequency of error
  label, etc. via `scripts/make_corpus_stats.py`.  (Additionally,
  `make_train_stats.py` adds information about training set sizes from the CoNLL
  2018 shared task.)


## Data files

The `data/` directory is used for data files produced by the scripts, but also
contains the following:

- `datastats_plus.csv` contains the corpus-level stats computed from
  `make_corpus_stats.py`.

- `mlcstats.tsv` contains the morphological complexity measures calculated with
  the scripts at
  [https://github.com/coltekin/mlc2018/](https://github.com/coltekin/mlc2018/).

- `ud2.5-frequencies.json.gz` contains token frequencies computed from UD2.5
  treebanks; they are used for step 3 in the analysis pipeline.

### Experimental results

- `analyzed_rf.tar.gz` contains the outputs from `analyze.py`, i.e. the feature
  importances for all data points in our analysis.

- The `logstats_*.csv` files contain the results from running
  `parse_logfiles.py`, i.e. classifier performance for all data points in our
  analysis.

- We also provide an unedited Jupyter notebook of plots and other calculations
  in the `notebooks/` directory.


## Questions

For further questions about this work, feel free to create an issue on the
GitHub repository or email Marcel Bollmann <marcel@bollmann.me> directly.
