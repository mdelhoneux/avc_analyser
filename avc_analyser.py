from optparse import OptionParser
from uuparser.src import utils
import os, copy
from avc.src.options_manager import OptionsManager
from avc.src.utils import (collect_avc_info,
                           dump_all_vecs,
                           train_word2vec,
                           collect_finite_verb_info,
                           write_avcs)
from avc.src.task_manager import TaskManager
import config

def run(treebank, i, options, n_tot, all_results={}):
    word_types = options.word_types
    vec_types = options.vec_types
    #empty if we do not check for lemma
    lemma_aux_list = config.lemma_auxiliaries[treebank.iso_id]
    lemma_aux_list = [item.encode('utf-8') for item in lemma_aux_list]
    if not options.tasks:
        if treebank.testfile:
            testdata = list(utils.read_conll(treebank.testfile, False,
                                             treebank.iso_id))
        else:
            testdata= None

        traindata = list(utils.read_conll(treebank.trainfile, False,
                                          treebank.iso_id))

    if options.create_avc_gold:
        if word_types != ['finite_verb']:
            if 'finite_verb' in word_types:
                traindata_copy = copy.deepcopy(traindata)
                if testdata:
                    testdata_copy = copy.deepcopy(testdata)
            avcs = collect_avc_info(traindata, style=options.style,
                                    lemma_aux_list=lemma_aux_list)
            print "writing to " + treebank.task_train
            write_avcs(avcs, treebank.task_train)

            if testdata:
                avcs = collect_avc_info(testdata, style=options.style,
                                        lemma_aux_list=lemma_aux_list)
                print "writing to " + treebank.task_test
                write_avcs(avcs, treebank.task_test)

        if 'finite_verb' in word_types:
            if len(word_types) > 1:
                traindata = traindata_copy
                if testdata:
                    testdata = testdata_copy

            finite_verbs = collect_finite_verb_info(traindata)
            print "writing to " + treebank.fv_train
            write_avcs(finite_verbs, treebank.fv_train)

            finite_verbs = collect_finite_verb_info(testdata)
            print "writing to " + treebank.fv_test
            write_avcs(finite_verbs, treebank.fv_test)


    elif options.dump_vecs:
        if options.style == 'ud':
            base_model, compos_model = config.models
        elif options.style == 'ms':
            base_model, compos_model = config.transformed_models


        if 'composed' in vec_types:
            traindata_copy = copy.deepcopy(traindata)
            testdata_copy = copy.deepcopy(testdata)
            traindata_copy, testdata_copy = utils.parser_process_data(treebank,
                                                                      compos_model,
                                                                      traindata_copy,
                                                                      testdata_copy,
                                                                      predict=True)
            collect_avc_info(traindata_copy, options.style, lemma_aux_list=lemma_aux_list)
            collect_avc_info(testdata_copy, options.style, lemma_aux_list=lemma_aux_list)
            if options.style == 'ud':
                word_type = 'main_verb'
            else:
                word_type = 'aux'
            dump_all_vecs(options,traindata_copy,testdata_copy,treebank,'composed',word_type,
                         options.style)
            vec_types.remove('composed')

        if not (vec_types == []): #any left
            traindata, testdata = utils.parser_process_data(treebank,
                                                            base_model,
                                                            traindata,
                                                            testdata)
            collect_avc_info(traindata, options.style, lemma_aux_list=lemma_aux_list)
            collect_avc_info(testdata, options.style, lemma_aux_list=lemma_aux_list)
            #TODO: this could also be done in parallel potentially
            for vec_type in vec_types:
                for word_type in word_types:
                    if word_type == 'finite_verb' or word_type =='punct':
                        collect_finite_verb_info(traindata)
                        collect_finite_verb_info(testdata)
                    dump_all_vecs(options,traindata,testdata,treebank,vec_type,word_type,
                                 options.style)

    elif options.train_word2vec:
        train_word2vec(traindata,treebank)

    elif options.predict or options.evaluate:
        tm = TaskManager(treebank,options,i,om.task_list,vec_types,word_types)
        print "Working on %s"%treebank.iso_id
        results = {}
        if options.predict and options.parallel:
                from joblib import Parallel, delayed
                import multiprocessing
                num_cores = multiprocessing.cpu_count()
                results = Parallel(n_jobs=num_cores)(delayed(tm.run)(results,
                                                                     vec_type,
                                                                     word_type)
                                                     for vec_type in options.vec_types
                                                     for word_type in options.word_types )
        else:
            for vec_type in options.vec_types:
                for word_type in options.word_types:
                    tm.run(results, vec_type, word_type)
            if options.evaluate:
                tm.write_results(results)

if __name__ == '__main__':
    parser = OptionParser()
    #maybe some of these should also to the config file
    parser.add_option("--outdir", type="string", dest="output", default="taskEXP")
    parser.add_option("--task_prediction_output", type="string",
                      dest="taskpred_out", default="taskpred")
    parser.add_option("--style", type="string", dest="style", default="ud")
    parser.add_option("--wembed_dir", type="string", dest="wembed_dir",
                      default="wembed")
    #TODO: these should all be mutually exclusive
    parser.add_option("--create_avc_gold", action="store_true", dest="create_avc_gold", default=False)
    parser.add_option("--predict", action="store_true", dest="predict", default=False)
    parser.add_option("--evaluate", action="store_true", dest="evaluate", default=False)
    parser.add_option("--train_word2vec", action="store_true", dest="train_word2vec", default=False)
    parser.add_option("--parallel", action="store_true", dest="parallel", default=False)
    parser.add_option("--test_tasks", dest="tasks", default=None)
    parser.add_option("--classifier", dest="classifier", default='mlp')
    parser.add_option("--word_types", dest="word_types", default="main_verb finite_verb")
    parser.add_option("--vec_types", dest="vec_types", default="contextual type")
    parser.add_option("--dump_vecs", action="store_true", dest="dump_vecs", default=False)
    parser.add_option("--results-file", type="string", dest="results_file",\
                      default="res.csv")
    parser.add_option("--datadir", dest="datadir", help="UD Dir -obligatory if\
                      using include", default=None)
    parser.add_option("--n_seed", type="int", dest="n_seed", default=1)

    parser.add_option("--include", dest="include", default =None,\
                      help="The languages to be run if using UD - None\
                      by default - if None - need to specify dev,train,test.\
                      \n Used in combination with multiling: trains a common \
                      parser for all languages. Otherwise, train monolingual \
                      parsers for each")

    (options, args) = parser.parse_args()

    om = OptionsManager(options)
    n_tot = len(om.languages) - 1
    all_results = {}
    for i, treebank in enumerate(om.languages):
        run(treebank, i, options, n_tot, all_results)

