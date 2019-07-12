# -*- coding: utf-8 -*-
from uuparser.src.utils import ConllEntry
import os
import numpy as np
import gensim

class AuxiliaryVerbConstruction(object):
    def __init__(self,senID, avcID, has_dobj=None, has_iobj=None, subj_num=None,\
                 subj_pers=None):
        self.senID = senID
        self.avcID = avcID
        self.has_dobj = has_dobj
        self.has_iobj = has_iobj
        self.subj_num = subj_num if subj_num != '_' else None
        self.subj_pers = subj_pers if subj_pers != -1 else None
        self.aux_ids = []

    def find_avc_args(self,conll_sentence,verbs):
        verb_ids = [v.id for v in verbs]
        self.has_subj = int(bool([token for token in conll_sentence if token.parent_id in \
                                  verb_ids and token.relation.split(":")[0] == 'nsubj']))
        self.has_dobj = int(bool([token for token in conll_sentence if token.parent_id in \
                              verb_ids and token.relation.split(":")[0] == 'obj']))
        self.has_iobj = int(bool([token for token in conll_sentence if token.parent_id in \
                              verb_ids and token.relation.split(":")[0] == 'iobj']))

    def get_agreement_feats(self,token):
        if 'Number' in token.feats:
            self.subj_num = token.feats['Number']
        #this happens for Finnish
        if 'Person' in token.feats and token.feats['Person'] != 0:
            self.subj_pers = token.feats['Person']

    def __str__(self):
        values = [self.senID, self.avcID, \
                  self.has_dobj, self.has_iobj, self.subj_num,\
                  self.subj_pers
                  ]
        return '\t'.join(['_' if v is None else str(v) for v in values])

def init_token(token):
    token.avcID = None
    token.is_aux = False
    token.is_main_verb = False
    token.is_main_aux = False
    token.is_punct = False
    token.is_finite_verb = False
    return token

def collect_avc_info(sentences, style='ud', lemma_aux_list=[]):
    if style == 'ud':
        return _collect_avc_info(sentences, lemma_aux_list)
    elif style == 'ms':
        return collect_avc_info_ms(sentences, lemma_aux_list)

def _collect_avc_info(sentences, lemma_aux_list=[]):
    senID = -1
    all_avcs = []
    for sentence in sentences:
        avcs = {}
        senID += 1
        avcID = 0
        conll_sentence = [init_token(entry) for entry in sentence if isinstance(entry, ConllEntry)]
        main_verbs = []
        avc = AuxiliaryVerbConstruction(senID,avcID)
        for token in conll_sentence:
            if is_aux_dependency(token, conll_sentence[token.parent_id]) and\
            (token.lemma.encode('utf-8') in lemma_aux_list or lemma_aux_list == []):
                if token.parent_id not in main_verbs:
                    avcID += 1
                    avc = AuxiliaryVerbConstruction(senID,avcID)
                    avc.main_verb_id = token.parent_id
                    main_verbs.append(token.parent_id)
                    avcs[token.parent_id] = avc
                    main_verb = conll_sentence[avc.main_verb_id]
                    main_verb.is_main_verb = True
                    main_verb.avcID = avc.avcID
                    avc.find_avc_args(conll_sentence,[main_verb])
                token.avcID = avcs[token.parent_id].avcID
                token.is_aux = True
                avcs[token.parent_id].aux_ids.append(token.id)

        #annotate avc with leftmost aux
        for avc in sorted(avcs, key = lambda x: avcs[x].avcID):
            avc = avcs[avc]
            leftmost_aux_id = avc.aux_ids[0]
            avc.get_agreement_feats(conll_sentence[leftmost_aux_id])
            all_avcs.append(avc)
    return all_avcs

def collect_avc_info_ms(sentences, lemma_aux_list):
    senID = -1
    all_avcs = []
    for sentence in sentences:
        all_aux = []
        avcs = []
        senID += 1
        avcID = 0
        conll_sentence = [init_token(entry) for entry in sentence if isinstance(entry, ConllEntry)]
        avc = AuxiliaryVerbConstruction(senID,avcID)
        for token in conll_sentence:
            if is_aux_dependency(token, conll_sentence[token.parent_id]):
                aux_id = token.parent_id
                if aux_id not in all_aux:
                    # save prev
                    if len(avc.aux_ids) >0:
                        avcs.append(avc)
                    avcID += 1
                    avc = AuxiliaryVerbConstruction(senID,avcID)
                    recurse_aux_chain(token, avc, conll_sentence)
                    if lemma_aux_list:
                        filtered_aux = filter_lemma(avc, conll_sentence, lemma_aux_list)
                        all_aux += filtered_aux
                        if avc.aux_ids == []:
                            avcID -= 1

                    all_aux += avc.aux_ids

        #saving last
        if len(avc.aux_ids) >0:
            avcs.append(avc)

        for avc in avcs:
            verbs = [tok for tok in conll_sentence if tok.avcID == avc.avcID]
            leftmost_aux= conll_sentence[min(avc.aux_ids)]
            avc.get_agreement_feats(leftmost_aux)
            avc.find_avc_args(conll_sentence, verbs)
            all_avcs.append(avc)
    return all_avcs

def recurse_aux_chain(token, avc, conll_sentence):
    recurse_head(conll_sentence[token.parent_id], avc, conll_sentence)
    recurse_dependent(token, avc, conll_sentence)

def recurse_head(tok, avc, conll_sentence):
    if is_aux_dependency(tok, conll_sentence[tok.parent_id]):
        tok.is_aux = True
        tok.avcID = avc.avcID
        avc.aux_ids.append(tok.id)
        recurse_head(conll_sentence[tok.parent_id], avc, conll_sentence)
    else:
        tok.is_aux = True
        tok.is_main_aux = True
        tok.avcID = avc.avcID
        avc.aux_ids.append(tok.id)

def recurse_dependent(tok, avc, conll_sentence):
    next_aux = is_head_of_aux_dep(tok, conll_sentence)
    #the dependent is itself the head of an aux dependency relation
    if next_aux:
        tok.is_aux = True
        tok.avcID = avc.avcID
        avc.aux_ids.append(tok.id)
        recurse_dependent(next_aux, avc, conll_sentence)
    else:
        tok.is_aux = False
        tok.is_main_verb = True
        tok.avcID = avc.avcID

def is_head_of_aux_dep(token, conll_sentence):
    aux_id = [tok for tok in conll_sentence if (tok.parent_id == token.id and
                                                is_aux_dependency(tok, token))]
    if len(aux_id) > 0:
        return aux_id[0]
    else:
        return None

def filter_lemma(avc, conll_sentence, lemma_aux_list):
    filtered_aux = []
    for tok in conll_sentence:
        if tok.avcID == avc.avcID:
            if tok.is_aux and tok.lemma.encode('utf-8') not in lemma_aux_list:
                filtered_aux.append(tok.id)
                tok.is_aux = False
                tok.is_main_aux = False
                tok.avcID = None
                avc.aux_ids.remove(tok.id)

    if filtered_aux:
        #unannotate main verb
        if avc.aux_ids == []:
            for tok in conll_sentence:
                if tok.avcID == avc.avcID:
                    tok.avcID = None
                    tok.is_main_verb = False
    return filtered_aux

def is_finite_verb(token, conll_sentence):
    if token.cpos == 'VERB' and not is_aux_dependency(token, conll_sentence[token.parent_id]):
        if 'VerbForm' in token.feats and token.feats['VerbForm'] == 'Fin':
            # check it has no aux or cop as dep 
            children_rel = [tok.relation.split(":")[0] for tok in conll_sentence if tok.parent_id == token.id]
            #but also as head / could still be cop
            children_rel += [token.relation.split(":")[0]]
            if 'aux' not in children_rel and 'cop' not in children_rel:
                return True
    return False

def collect_finite_verb_info(sentences):
    senID = -1
    finite_verbs = []
    for sentence in sentences:
        senID += 1
        conll_sentence = [entry for entry in sentence if isinstance(entry, ConllEntry)]
        avcID = -1
        for token in conll_sentence:
            if is_finite_verb(token, conll_sentence):
                token.is_finite_verb = True
                avcID += 1
                avc = AuxiliaryVerbConstruction(senID, avcID)
                avc.main_verb = token.id
                token.avcID = avcID
                avc.find_avc_args(conll_sentence,[token])
                avc.get_agreement_feats(token)
                get_closest_punct(token,conll_sentence)
                finite_verbs.append(avc)
    return finite_verbs

def get_closest_punct(tok,conll_sentence):
    #orders children from closest right then closest left #this is hacky
    reordered_sentence = conll_sentence[tok.id:] + list(reversed(conll_sentence[:tok.id]))
    children = [token for token in reordered_sentence if token.parent_id == tok.id]
    for child in children:
        if child.relation == 'punct':
            child.is_punct = True
            child.avcID = tok.avcID
            return

def is_aux_dependency(dependent, head):
    return dependent.relation.split(":")[0]== 'aux' and dependent.cpos in ['AUX','VERB'] and head.cpos in ['AUX','VERB']

def dump_all_vecs(options,traindata,testdata, treebank,vec_type,word_type,
                  style='ud'):
    #TODO: this should be generalised to include other vec options
    iso_id = treebank.iso_id
    vec_dir = "%s/%s/%s/%s"%(options.output,options.wembed_dir,vec_type,word_type)
    if not os.path.exists(vec_dir):
        os.makedirs(vec_dir)
    treebank.train_vec="%s/%s_train.vec"%(vec_dir,iso_id)
    treebank.test_vec="%s/%s_test.vec"%(vec_dir,iso_id)
    dump_vecs(traindata, treebank, treebank.train_vec, vec_type, word_type,
              style)
    dump_vecs(testdata, treebank, treebank.test_vec, vec_type, word_type, style)

def dump_vecs(sentences, treebank, outfile, vec_type="contextual",
              word_type="main_verb", style='ud'):
    if vec_type == 'word2vec':
        word2vecModel = gensim.models.Word2Vec.load(treebank.word2vec_model)
    else:
        word2vecModel=None

    out = open(outfile,'w')
    print "dumping vectors to: " + outfile
    senID = -1
    for sen in sentences:
        senID += 1
        conll_sentence = [entry for entry in sen if isinstance(entry, ConllEntry)]
        aux_num = 1
        avcID = -1
        for token in conll_sentence:
            if word_type=="aux":
                if token.is_aux:
                    if token.avcID == avcID:
                        aux_num +=1
                    else:
                        avcID += 1
                        aux_num = 1
                    if token.is_main_aux or style == 'ud':
                        dump_vec(token,vec_type,out,senID,token.avcID,aux_num,word2vecModel)
            elif word_type=="main_verb" and token.is_main_verb:
                if token.avcID != avcID:
                    avcID += 1
                dump_vec(token,vec_type,out,senID,token.avcID,0,word2vecModel)
            elif word_type =='punct' and token.is_punct:
                vec = dump_vec(token,vec_type,out,senID,token.avcID,0,word2vecModel)
            elif word_type == 'finite_verb':
                if token.is_finite_verb:
                    dump_vec(token,vec_type,out,senID,token.avcID,0,word2vecModel)

    out.close()

def dump_vec(token,vec_type,out,senID,avcID,aux_num=0,word2vecModel=None):
    vec = "%s;%s;%s\t"%(str(senID),str(avcID),str(aux_num))
    if vec_type == "contextual":
        vec += ', '.join([str(i) for i in token.contextual_vector])
    elif vec_type == "type":
        vec += ', '.join([str(i) for i in token.vecs["word"].value()])
    elif vec_type == "char":
        vec += ', '.join([str(i) for i in token.vecs["char"].value()])
    elif vec_type == 'word2vec':
        if token.form in word2vecModel.wv:
            vec += ', '.join([str(i) for i in word2vecModel.wv[token.form]])
        elif token.norm in word2vecModel.wv:
            vec += ', '.join([str(i) for i in word2vecModel.wv[token.norm]])
        else:
            vec += ', '.join([str(0) for i in range(100)])
    elif vec_type == 'composed':
        vec += ', '.join([str(i) for i in token.composed_rep])
    out.write(vec)
    out.write("\n")

def read_vecs(options,treebank,vec_type,word_type):
    #change for finite verbs
    iso_id = treebank.iso_id
    vec_dir = "%s/%s/%s/%s"%(options.output,options.wembed_dir,vec_type,word_type)
    print "reading vec from: " + vec_dir
    treebank.train_vec="%s/%s_train.vec"%(vec_dir,iso_id)
    treebank.test_vec="%s/%s_test.vec"%(vec_dir,iso_id)
    train_vec = file_to_vecs(treebank.train_vec)
    test_vec = file_to_vecs(treebank.test_vec)
    return train_vec, test_vec

def file_to_vecs(infile):
    """
    Input: file with structure:
    sentence ID; verb group ID; aux number\t vec (num1, num2, ...)
    Output: dictionary with (sentence ID, verb group ID) as key and vec as value
    """
    d = {}
    for line in open(infile,'r'):
        vals,vec = line.strip('\n').split("\t")
        senID,avcID,aux_num = vals.split(";")
        vec = vec.split(", ")
        vec = np.array(vec,dtype='float64')
        d[(int(senID),int(avcID),int(aux_num))] = vec
    return d

def read_avc(infile):
    """
    Input: file with avc format
    Output: dictionary with (sentence ID, verb group ID) as key and verb group
    object as value
    """
    avcs = {}
    for line in open(infile,'r'):
        tok = line.strip('\n').split("\t")
        none_indices = []
        for i,item in enumerate(tok):
            if item == '_':
                tok[i] = None
        avc = AuxiliaryVerbConstruction(int(tok[0]), int(tok[1]), int(tok[2]), int(tok[3]),\
                                        tok[4], int(tok[5]) if str(tok[5]).isdigit()
                                        else -1)
        avcs[(avc.senID,avc.avcID)] = avc
    return avcs

def write_avcs(avcs,outfile):
    out = open(outfile,'w')
    for avc in avcs:
        out.write(str(avc))
        out.write("\n")
    out.close()


vect_id = {
    'contextual' : 'tok',
    'type' : 'type',
    'char': 'char',
    'word2vec' : 'w2v',
    'main_verb' : 'nfmv',
    'aux' : 'maux',
    'punct':'punct',
    'composed':'+c',
    'finite_verb':'fmv',
}

def train_word2vec(traindata,treebank):
    sentences = []
    for sentence in traindata:
        sen = [entry.form for entry in sentence if isinstance(entry, ConllEntry)]
        sentences.append(sen)
    model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
    print "saving w2v model to " + treebank.word2vec_model
    model.save(treebank.word2vec_model)

