import os,sys
from uuparser.src import utils
import copy
import config

class OptionsManager(object):
    def __init__(self,options):
        language_list = utils.parse_list_arg(options.include)
        if options.tasks:
            self.task_list = utils.parse_list_arg(options.tasks)
        if options.style == 'ud':
            options.datadir = config.datadir
        elif options.style == 'ms':
            options.datadir = config.transformed_datadir
        options.word_types = utils.parse_list_arg(options.word_types)
        if not options.datadir:
            raise Exception("You need to specify the data directory in\
                            config.py")
        options.vec_types = utils.parse_list_arg(options.vec_types)
        #TODO: not hard-encode this
        options.json_isos = 'uuparser/src/utils/ud2.2_iso.json'
        options.shared_task = False
        options.testdir = False
        options.golddir = False
        json_treebanks = utils.get_all_treebanks(options)
        self.languages = [lang for lang in json_treebanks if lang.iso_id in language_list]
        for language in self.languages:
            language.task_train="%s/avc_data/%s_train.avc"%(options.output,language.iso_id)
            language.task_test="%s/avc_data/%s_test.avc"%(options.output,language.iso_id)
            language.fv_train="%s/avc_data/fv/%s_train.avc"%(options.output,language.iso_id)
            language.fv_test="%s/avc_data/fv/%s_test.avc"%(options.output,language.iso_id)
            language.word2vec_model="%s/wembed/word2vec/%s_model"%(options.output,language.iso_id)
            #TODO: omg this is sooo ugly
            dirs="%s/avc_data/fv/"%options.output
            if options.create_avc_gold and not os.path.exists(dirs):
                os.makedirs(dirs)
            if options.train_word2vec:
                #OUCH ugly
                w2vec_dir = options.output + '/wembed/word2vec/'
                if not os.path.exists(w2vec_dir):
                    os.makedirs(w2vec_dir)
