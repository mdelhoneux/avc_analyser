from classifier import Classifier
from utils import read_avc, read_vecs, vect_id
import numpy as np
import time
import os

class TaskManager(object):
    def __init__(self, treebank, options, treebank_number, task_list, vec_types, word_types):
        self.treebank = treebank
        self.tasks = task_list
        self.options = options
        self.treebank_number = treebank_number

    def run(self, results, vec_type, word_type):
        if self.options.style == 'ud' and vec_type == 'composed' and word_type != 'main_verb':
            return
        if self.options.style == 'ms' and vec_type == 'composed' and word_type != 'aux':
            return
        print "Word type: %s - Vector type: %s"%(word_type,vec_type)

        #TODO: bit of an overkill to do this several times
        if word_type == 'finite_verb' or word_type == 'punct':
            avc_train = read_avc(self.treebank.fv_train)
            avc_test = read_avc(self.treebank.fv_test)
            print "Reading gold train file from %s"%self.treebank.fv_train
        else:
            avc_train = read_avc(self.treebank.task_train)
            avc_test = read_avc(self.treebank.task_test)
            print "Reading gold train file from %s"%self.treebank.task_train

        task_train_vecs, task_test_vecs = read_vecs(self.options,self.treebank,vec_type,word_type)
        acc = self.run_tasks(avc_train, avc_test, task_train_vecs,task_test_vecs,
                             vec_type,word_type)
        if self.options.evaluate:
            results[(vec_type,word_type)] = acc

    def run_tasks(self, avc_train, avc_test, task_train_vecs,task_test_vecs,
                  vec_type,word_type):
        acc = {}
        for task in self.tasks:
            print "Task %s"%task
            outdir = "%s/%s/%s/%s/%s/"%(self.options.output, self.options.taskpred_out, task, vec_type, word_type)
            if not os.path.exists(outdir):
                os.makedirs(outdir)


            if self.options.predict:
                for i in range(self.options.n_seed):
                    outfile = outdir + self.treebank.iso_id + '_' + str(i) + '.txt'
                    cl = Classifier(task, classifier_type=self.options.classifier)
                    beg = time.time()
                    cl.train(task_train_vecs,avc_train)
                    x_ids, pred = cl.predict(task_test_vecs, avc_test)
                    print "Training took: %.2gs"%(time.time()-beg)

                    #hacky!
                    if x_ids is not None and pred is not None:
                        out = open(outfile,'w')
                        for x,y in zip(x_ids, pred):
                            line = '\t'.join([str(i) for i in x]) + '\t' + str(y) + '\n'
                            out.write(line)
                        out.close()

            elif self.options.evaluate:
                acc[task] = {}
                acc[task]['acc'] = []
                cl = Classifier(task, classifier_type=self.options.classifier)
                x_ids,x_train, y_train = cl.get_xys(task_train_vecs,avc_train)
                cl.get_train_info(x_train,y_train)
                for seed_n in range(self.options.n_seed):
                    outfile = outdir + self.treebank.iso_id + '_' + str(seed_n) + '.txt'
                    #this could also be parallelized...

                    #bit of an overkill to read training data but no clean other way
                    #to get info about train size
                    results = [line.strip('\n').split('\t') for line in open(outfile, 'r')]
                    x_ids = [tuple([int(i) for i in line[:3]]) for line in results]
                    pred = [int(line[3]) if line[3].isdigit() else line[3] for line in results]

                    if seed_n == 0:
                        acc[task]['training_size'] = cl.trainsize
                        acc[task]['test_size'] = len(pred)

                        #TODO: bit of an overkill to calculate it each time
                        if not cl.task_undefined:
                            acc[task]['maj'] = cl.majority_baseline(task_train_vecs, avc_train,
                                                                    task_test_vecs, avc_test)*100
                        else:
                            acc[task]['maj'] = np.nan

                    seed_accuracy =cl.evaluate(x_ids,pred,avc_test) *100
                    acc[task]['acc'].append(seed_accuracy)
                acc[task]['acc'] = np.average(acc[task]['acc'])

        return acc

    def write_results(self,results):
        out = open(self.options.results_file,'a')

        if self.treebank_number == 0:
            line1=';;'
            line2=';;'
            for word_type in self.options.word_types:
                for i, vec_type in enumerate(self.options.vec_types):
                    if (vec_type,word_type) in results:
                        #this assume that composed comes last!
                        if i == 0:
                            line1 += ";;;"
                            line2 += "train;test;maj;"
                        line1 += "%s_%s;"%(vect_id[vec_type],vect_id[word_type])
                        line2 += 'acc;'
            out.write("%s\n"%line1)
            out.write("%s\n"%line2)

        for task in self.tasks:
            line = "%s;%s;"%(self.treebank.iso_id,task)
            for word_type in self.options.word_types:
                for i, vec_type in enumerate(self.options.vec_types):
                    if (vec_type,word_type) in results:
                        if i == 0:
                            line += "%d;"%results[(vec_type,word_type)][task]['training_size']
                            line += "%d;"%results[(vec_type,word_type)][task]['test_size']
                            line += "%.2f;"%results[(vec_type,word_type)][task]['maj']
                        line += "%.2f;"%(results[(vec_type,word_type)][task]['acc'])
            out.write("%s\n"%line)
            out.flush()

