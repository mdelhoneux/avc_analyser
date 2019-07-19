# AVC analyser

Code for reproducing the experiments in the following paper:  
What Should/Do/Can LSTMs Learn When Parsing Auxiliary Verb Constructions?
by Miryam de Lhoneux, Sara Stymne and Joakim Nivre  
available on [arXiv](https://arxiv.org/abs/1907.07950)

# Training the parser

The uuparser folder contains a modified version of [UUParser](https://github.com/UppsalaNLP/uuparser) where we save intermediate representations in different parts of the network.
See that repository for more details on how to train models.

# AVC transformation
To transform the representation of AVCs, see, [this repository](https://github.com/mdelhoneux/oDETTE).

# AVC analysis

First, the config.py file needs to be modified to specify the path of trained models as well as datasets.  
For running everything from data collection to the diagnostic classifiers, there is one main script: avc_analyser.py Some options are relevant to all steps:
* `--include`: list of iso codes
* `--word_types`: list of word types from [main_verb, finite_verb, aux, punct]
* `--vec_types`: list of vector types from [contextual, type, char, word2vec, composed]  
Those lists can be specified as a string with spaces or in a file with one word per line.

The different steps can be run with the different options in turn, as follows:  
* `--create_avc_gold`: create the data
* `--train_word2vec`: train a word2vec model on the training sets
* `--dump_vecs`: dump vectors from all models involved
* `--predict`: do the classification
* `--evaluate`: evaluate the results

It will create a lot of different folders and files and the final results table will be written to 'res.csv'.

To see all options:
```
python avc_analyser.py --help
```

By default, the scripts expects a UD representation for all parts of the pipeline, but the option `--style ms` can be used to use a MS representation style. 


#### License

This software is released under the terms of the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).

#### Contact

For questions and usage issues, please contact miryam dot de underscore lhoneux at lingfil dot uu dot se


