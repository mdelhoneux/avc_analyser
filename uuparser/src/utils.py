from collections import defaultdict, Counter, OrderedDict
import re
import os
from time import time
from itertools import chain
from operator import itemgetter
import random
import codecs, json, pickle

# a global variable so we don't have to keep loading from file repeatedly
iso_dict = {}
reverse_iso_dict = {}

class ConllEntry:
    def __init__(self, id, form, lemma, pos, cpos, feats=None, parent_id=None, relation=None,
        deps=None, misc=None, treebank_id=None, proxy_tbank=None, language=None, char_rep=None):

        self.id = id
        self.form = form
        self.char_rep = char_rep if char_rep else form
        self.norm = normalize(self.char_rep)
        self.cpos = cpos
        self.pos = pos
        self.parent_id = parent_id
        self.relation = relation

        self.lemma = lemma
        self.feats = self.feats_to_dict(feats)
        self.feats_string = feats
        self.deps = deps
        self.misc = misc

        self.pred_parent_id = None
        self.pred_relation = None
        self.treebank_id = treebank_id
        self.proxy_tbank = proxy_tbank
        self.language = language

        self.pred_pos = None
        self.pred_cpos = None

        #self.composed_rep = False


    def feats_to_dict(self, features):
        feats = features.split("|")
        if len(feats)>1:
            feats = [feat.split("=") for feat in feats]
            return {feat[0]:feat[1] for feat in feats}
        else:
            return {}


    def __str__(self):
        values = [str(self.id), self.form, self.lemma, \
                  self.pred_cpos if self.pred_cpos else self.cpos,\
                  self.pred_pos if self.pred_pos else self.pos,\
                  self.feats_string, str(self.pred_parent_id) if self.pred_parent_id \
                  is not None else str(self.parent_id), self.pred_relation if\
                  self.pred_relation is not None else self.relation, \
                  self.deps, self.misc]
        return '\t'.join(['_' if v is None else v for v in values])

class Treebank(object):
    def __init__(self,trainfile,devfile,testfile):
        self.name = 'noname'
        self.trainfile = trainfile
        self.devfile = devfile
        self.dev_gold = devfile
        self.test_gold = testfile
        self.testfile = testfile
        self.outfilename = None
        self.proxy_tbank = None

class UDtreebank(Treebank):
    def __init__(self, treebank_info, options):
        """
        Read treebank info to a treebank object
        """
        self.name, self.iso_id = treebank_info
        #hack
        if not hasattr(options, 'proxy_tbank'):
            self.proxy_tbank = None
        if not options.shared_task:

            if options.datadir:
                data_path = os.path.join(options.datadir,self.name)
            else:
                data_path = os.path.join(options.testdir,self.name)

            if options.testdir:
                test_path = os.path.join(options.testdir,self.name)
            else:
                test_path = data_path

            if options.golddir:
                gold_path = os.path.join(options.golddir,self.name)
            else:
                gold_path = test_path

            self.trainfile = os.path.join(data_path, self.iso_id + "-ud-train.conllu")
            self.devfile = os.path.join(test_path, self.iso_id + "-ud-dev.conllu")
            self.testfile = os.path.join(test_path, self.iso_id + "-ud-dev.conllu")
            self.test_gold = os.path.join(gold_path, self.iso_id + "-ud-dev.conllu")
            self.dev_gold = os.path.join(gold_path, self.iso_id + "-ud-dev.conllu")
        else:
            if not options.predict:
                self.trainfile = os.path.join(options.datadir, self.iso_id + ".conllu")
            #self.testfile = os.path.join(options.testdir, self.iso_id + "-udpipe.conllu")
            #self.testfile = os.path.join(options.testdir, self.iso_id + ".conllu")
            self.testfile = os.path.join(options.testdir, self.iso_id + ".txt")
            if options.golddir:
                self.test_gold = os.path.join(options.golddir, self.iso_id + ".conllu")
            else:
                self.test_gold = self.testfile
            self.devfile = self.testfile
            self.dev_gold = self.test_gold
        self.outfilename = self.iso_id + '.conllu'

class ParseForest:
    def __init__(self, sentence):
        self.roots = list(sentence)

        for root in self.roots:
            root.children = []
            root.scores = None
            root.parent = None
            root.pred_parent_id = None
            root.pred_relation = None
            root.vecs = None
            root.lstms = None

    def __len__(self):
        return len(self.roots)


    def Attach(self, parent_index, child_index):
        parent = self.roots[parent_index]
        child = self.roots[child_index]

        child.pred_parent_id = parent.id
        del self.roots[child_index]


def isProj(sentence):
    forest = ParseForest(sentence)
    unassigned = {entry.id: sum([1 for pentry in sentence if pentry.parent_id == entry.id]) for entry in sentence}

    for _ in xrange(len(sentence)):
        for i in xrange(len(forest.roots) - 1):
            if forest.roots[i].parent_id == forest.roots[i+1].id and unassigned[forest.roots[i].id] == 0:
                unassigned[forest.roots[i+1].id]-=1
                forest.Attach(i+1, i)
                break
            if forest.roots[i+1].parent_id == forest.roots[i].id and unassigned[forest.roots[i+1].id] == 0:
                unassigned[forest.roots[i].id]-=1
                forest.Attach(i, i+1)
                break

    return len(forest.roots) == 1

def get_vocab(treebanks,datasplit,char_map={}):
    """
    Collect frequencies of words, cpos, pos and deprels + languages.
    """

    te = time()

    data = read_conll_dir(treebanks,datasplit,char_map=char_map)

    # could use sets directly rather than counters for most of these,
    # but having the counts might be useful in the future or possibly for debugging etc
    wordsCount = Counter()
    charsCount = Counter()
    posCount = Counter()
    cposCount = Counter()
    relCount = Counter()
    tbankCount = Counter() # note that one language can have several treebanks
    langCount = Counter()

    for sentence in data:
        for node in sentence:
            if isinstance(node, ConllEntry):
                wordsCount.update([node.norm])
                if node.char_rep != u"*root*":
                    charsCount.update(node.char_rep)
                posCount.update([node.pos])
                cposCount.update([node.cpos])
                relCount.update([node.relation])
                treebank_id = node.treebank_id
                tbankCount.update([treebank_id])
                lang = get_lang_from_tbank_id(treebank_id)
                langCount.update([lang])

    print "Finished collecting vocab in %.2fs"%(time()-te)

    # the redundancy with wordsCount is deliberate and crucial to ensure the word lookup
    # loads the same when predicting with a saved model later on
    # this is also another reason not to use sets for everything here as they are unordered
    # which creates problems when loading from file at predict time
    return (wordsCount, wordsCount.keys(), charsCount.keys(), posCount.keys(),
       cposCount.keys(), relCount.keys(), tbankCount.keys(), langCount.keys())

def load_iso_dict(json_file='uuparser/src/utils/ud_iso.json'):
    print "Loading ISO dict from %s"%json_file
    global iso_dict
    ud_iso_file = codecs.open(json_file,encoding='utf-8')
    json_str = ud_iso_file.read()
    iso_dict = json.loads(json_str)

def load_reverse_iso_dict(json_file='uuparser/src/utils/ud_iso.json'):
    global reverse_iso_dict
    if not iso_dict:
        load_iso_dict(json_file=json_file)
    reverse_iso_dict = {v: k for k, v in iso_dict.iteritems()}

def load_lang_iso_dict(json_file='uuparser/src/utils/ud_iso.json'):

    if not iso_dict:
        load_iso_dict(json_file)
    lang_iso_dict = {}

    for tb_name, tb_iso in iso_dict.items():
        lang_name = get_lang_from_tbank_name(tb_name)
        lang_iso = get_lang_iso(tb_iso)

        lang_iso_dict[lang_name] = lang_iso

    return lang_iso_dict

# convert treebank to language by removing everything after underscore
def get_lang_from_tbank_name(tbank_name):

    # weird exceptions of separate langs with dashes
    m = re.match('UD_(Norwegian-(Bokmaal|Nynorsk))',tbank_name)
    if m:
        lang = m.group(1)
    else:
        m = re.search('^UD_(.*?)(-|$)',tbank_name)
        lang = m.group(1) if m else tbank_name

    return lang

def get_lang_from_tbank_id(tbank_id):
    if not tbank_id:
        return None

    if not reverse_iso_dict:
        load_reverse_iso_dict()
    return get_lang_from_tbank_name(reverse_iso_dict[tbank_id])

# gets everything before the underscore in treebank iso e.g. "sv_talbanken" -> "sv"
# with an exception for the two Norwegian variants where it's useful to consider them
# as separate languages
def get_lang_iso(treebank_iso):

    # before UD 2.2 not all treebank isos contained underscores
    if not re.search(r'_',treebank_iso):
        return treebank_iso
    else:
        m = re.match(r'(.*_(nynorsk|bokmaal)?)',treebank_iso)
        return m.group(1).rstrip('_')

# from a list of treebanks, return those that match a particular language
def get_treebanks_from_lang(treebank_ids,lang):
   return [treebank_id for treebank_id in treebank_ids if get_lang_from_tbank_id(treebank_id) == lang]

def get_all_treebanks(options):

    if not iso_dict:
        load_iso_dict(options.json_isos)
    treebank_metadata = iso_dict.items()

    json_treebanks = [UDtreebank(ele, options) for ele in treebank_metadata]

    return json_treebanks

def read_conll_dir(treebanks,filetype,maxSize=-1,char_map={}):
    #print "Max size for each corpus: ", maxSize
    if filetype == "train":
        return chain(*(read_conll(treebank.trainfile, treebank.iso_id, treebank.proxy_tbank, maxSize, train=True, char_map=char_map) for treebank in treebanks))
    elif filetype == "dev":
        return chain(*(read_conll(treebank.devfile, treebank.iso_id, treebank.proxy_tbank, train=False, char_map=char_map) for treebank in treebanks))
    elif filetype == "test":
        return chain(*(read_conll(treebank.testfile, treebank.iso_id, treebank.proxy_tbank, train=False, char_map=char_map) for treebank in treebanks))

def read_conll(filename, treebank_id=None, proxy_tbank=None, maxSize=-1, hard_lim=False, vocab_prep=False, drop_nproj=False, train=True, char_map={}):
    # hard lim means capping the corpus size across the whole training procedure
    # soft lim means using a sample of the whole corpus at each epoch
    fh = codecs.open(filename,'r',encoding='utf-8')
    print "Reading " + filename
    if vocab_prep and not hard_lim:
        maxSize = -1 # when preparing the vocab with a soft limit we need to use the whole corpus
    dropped = 0
    sents_read = 0
    if treebank_id:
        language = get_lang_from_tbank_id(treebank_id)
    else:
        language = None
    root = ConllEntry(0, '*root*', '*root*', 'ROOT-POS', 'ROOT-CPOS', '_', -1, 'rroot',
        '_', '_',treebank_id=treebank_id, proxy_tbank=proxy_tbank,language=language)
    tokens = [root]
    yield_count = 0
    if maxSize > 0 and not hard_lim:
        sents = []
    for line in fh:
        tok = line.strip().split('\t')
        if not tok or line.strip() == '': # empty line, add sentence to list or yield
            if len(tokens)>1:
                sents_read += 1
                conll_tokens = [t for t in tokens if isinstance(t,ConllEntry)]
                if not drop_nproj or isProj(conll_tokens): # keep going if it's projective or we're not dropping non-projective sents
                    if train:
                        inorder_tokens = inorder(conll_tokens)
                        for i,t in enumerate(inorder_tokens):
                            t.projective_order = i
                        for tok in conll_tokens:
                            tok.rdeps = [i.id for i in conll_tokens if i.parent_id == tok.id]
                            if tok.id != 0:
                                tok.parent_entry = [i for i in conll_tokens if i.id == tok.parent_id][0]
                    if maxSize > 0:
                        if not hard_lim:
                            sents.append(tokens)
                        else:
                            yield tokens
                            yield_count += 1
                            if yield_count == maxSize:
                                print "Capping size of corpus at " + str(yield_count) + " sentences"
                                break;
                    else:
                        yield tokens
                else:
                    #print 'Non-projective sentence dropped'
                    dropped += 1
            tokens = [root]
        else:
            if line[0] == '#' or '-' in tok[0] or '.' in tok[0]: # a comment line, add to tokens as is
                tokens.append(line.strip())
            else: # an actual ConllEntry, add to tokens
                char_rep = tok[1] # representation to use in character model
                if language and language in char_map:
                    for char in char_map[language]:
                        char_rep = re.sub(char,char_map[language][char],char_rep)
                if tok[2] == "_":
                    tok[2] = tok[1].lower()
                token = ConllEntry(int(tok[0]), tok[1], tok[2], tok[4], tok[3], tok[5], int(tok[6]) if tok[6] != '_' else -1, tok[7], tok[8], tok[9],treebank_id=treebank_id,proxy_tbank=proxy_tbank,language=language,char_rep=char_rep)

                tokens.append(token)

    if hard_lim and yield_count < maxSize:
        print 'Warning: unable to yield ' + str(maxSize) + ' sentences, only ' + str(yield_count) + ' found'

# TODO: deal with case where there are still unyielded tokens
# e.g. when there is no newline at end of file
#    if len(tokens) > 1:
#        yield tokens

    print '%u sentences read'%(sents_read)

    if maxSize > 0 and not hard_lim:
        if len(sents) > maxSize:
            sents = random.sample(sents,maxSize)
            print "Yielding " + str(len(sents)) + " random sentences"
        for toks in sents:
            yield toks

def write_conll(fn, conll_gen):
    print "Writing to " + fn
    sents = 0
    with codecs.open(fn, 'w', encoding='utf-8') as fh:
        for sentence in conll_gen:
            sents += 1
            for entry in sentence[1:]:
                fh.write(unicode(entry) + '\n')
                #print str(entry)
            fh.write('\n')
        print "Wrote " + str(sents) + " sentences"

def write_conll_multiling(conll_gen, treebanks):
    tbank_dict = {treebank.iso_id:treebank for treebank in treebanks}
    cur_tbank = conll_gen[0][0].treebank_id
    outfile = tbank_dict[cur_tbank].outfilename
    fh = codecs.open(outfile,'w',encoding='utf-8')
    print "Writing to " + outfile
    for sentence in conll_gen:
        if cur_tbank != sentence[0].treebank_id:
            fh.close()
            cur_tbank = sentence[0].treebank_id
            outfile = tbank_dict[cur_tbank].outfilename
            fh = codecs.open(outfile,'w',encoding='utf-8')
            print "Writing to " + outfile
        for entry in sentence[1:]:
            fh.write(unicode(entry) + '\n')
        fh.write('\n')


def parse_list_arg(l):
    """Return a list of line values if it's a file or a list of values if it
    is a string"""
    if os.path.isfile(l):
        f = codecs.open(l, 'r', encoding='utf-8')
        return [line.strip("\n").split()[0] for line in f]
    else:
        return [el for el in l.split(" ")]

numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+");
def normalize(word):
    return 'NUM' if numberRegex.match(word) else word.lower()

def evaluate(gold,test,conllu):
    scoresfile = test + '.txt'
    print "Writing to " + scoresfile
    if not conllu:
        #os.system('perl src/utils/eval.pl -g ' + gold + ' -s ' + test  + ' > ' + scoresfile + ' &')
        os.system('perl uuparser/src/utils/eval.pl -g ' + gold + ' -s ' + test  + ' > ' + scoresfile )
    else:
        os.system('python\
                  uuparser/src/utils/evaluation_script/conll17_ud_eval.py -v -w\
                  uuparser/src/utils/evaluation_script/weights.clas ' + gold +'\
                  ' + test + ' > ' + scoresfile)
    score = get_LAS_score(scoresfile,conllu)
    return score

def inorder(sentence):
    queue = [sentence[0]]
    def inorder_helper(sentence,i):
        results = []
        left_children = [entry for entry in sentence[:i] if entry.parent_id == i]
        for child in left_children:
            results += inorder_helper(sentence,child.id)
        results.append(sentence[i])

        right_children = [entry for entry in sentence[i:] if entry.parent_id == i ]
        for child in right_children:
            results += inorder_helper(sentence,child.id)
        return results
    return inorder_helper(sentence,queue[0].id)

def set_seeds(options):
    python_seed = 1
    if not options.predict and options.dynet_seed: # seeds shouldn't make any difference when predicting
        print "Using default Python seed"
        random.seed(python_seed)

def generate_seed():
    return random.randint(0,10**9) # this range seems to work for Dynet and Python's random function

def get_LAS_score(filename, conllu=True):
    score = None
    with codecs.open(filename,'r',encoding='utf-8') as fh:
        if conllu:
            for line in fh:
                if re.match(r'^LAS',line):
                    elements = line.split()
                    score = float(elements[6]) # should extract the F1 score
        else:
            las_line = [line for line in fh][0]
            score = float(las_line.split('=')[1].split()[0])

    return score

def extract_embeddings_from_file(filename, words=None, max_emb=-1, filtered_filename=None):
    # words should be a set used to filter the embeddings
    print "Extracting embeddings from", filename
    ts = time()
    line_count = 0
    error_count = 0 # e.g. invalid utf-8 in embeddings file

    with open(filename,'r') as fh: # byte string

        fh.readline() # ignore first line with embedding stats
        embeddings = OrderedDict()

        for line in fh:
            if max_emb < 0 or line_count < max_emb:
                try:
                    # only split on normal space, not e.g. non-break space
                    eles = line.decode('utf-8').strip().split(" ")
                    word = re.sub(u"\xa0"," ",eles[0]) # replace non-break space with regular space
                    if not words or word in words:
                        embeddings[word] = [float(f) for f in eles[1:]]
                except UnicodeDecodeError:
#                    print "Unable to read word at line %i: %s"%(line_count, word)
                    error_count += 1
                line_count += 1
                if line_count % 100000 == 0:
                    print "Reading line: " + str(line_count)
            else:
                break

    print "Read %i embeddings in %.2gs"%(line_count,time()-ts)
#    print "%i utf-8 errors"%error_count
    if words:
        print "%i entries found from vocabulary (out of %i)"%(len(embeddings),len(words))

    if filtered_filename and embeddings:
        print "Writing filtered embeddings to " + filtered_filename
        with open(filtered_filename,'w') as fh_filter:
            no_embeddings = len(embeddings)
            embedding_size = len(embeddings.itervalues().next())
            fh_filter.write("%i %i\n"%(no_embeddings,embedding_size))
            for word in embeddings:
                line = re.sub(" ",u"\xa0",word).encode('utf-8') + " " + \
                    " ".join(["%.6f"%value for value in embeddings[word]]) + "\n"
                fh_filter.write(line)

    return embeddings

def get_external_embeddings(options,lang=None,words=None,chars=False):

    external_embedding = {}

    if options.ext_emb_file:

        external_embedding = extract_embeddings_from_file(options.ext_emb_file,
            words, options.max_ext_emb)

    elif options.ext_emb_dir:

        if not lang:
            raise Exception("No language specified for external embeddings")
        else:

            lang_iso_dict = load_lang_iso_dict(options.json_isos)
            emb_dir = os.path.join(options.ext_emb_dir,lang)

            if chars:
                emb_file = os.path.join(emb_dir,lang_iso_dict[lang] + '.vectors.chars.txt')
            else:
                if options.shared_task or options.unfiltered_vecs:
                    emb_file = os.path.join(emb_dir,lang_iso_dict[lang] + '.vectors.txt')
                else:
                    emb_file = os.path.join(emb_dir,lang_iso_dict[lang] + '.vectors.filtered.txt')

            if os.path.exists(emb_file):
                external_embedding.update(extract_embeddings_from_file(emb_file, words, options.max_ext_emb))
            else:
                print "Warning: %s does not exist, proceeding without"%(emb_file)

    return external_embedding

# for the most part, we want to send stored options to the parser when in
# --predict mode, however we want to allow some of these to be updated
# based on the command line options specified by the user at predict time
def fix_stored_options(stored_opt,options):

    stored_opt.predict = True
    opts = ['forced_tbank_emb', 'ext_emb_dir', 'ext_emb_file', 'max_ext_emb',
            'shared_task', 'char_map_file', 'lang_emb_size']
    for opt in opts:
        if hasattr(options, opt):
            option_value = getattr(options, opt)
            setattr(stored_opt, opt, option_value)

def parser_process_data(treebank, model_dir, traindata, testdata, predict=False):
    from arc_hybrid import ArcHybridLSTM
    params = os.path.join(model_dir,treebank.iso_id, 'params.pickle')
    model_file = os.path.join(model_dir,treebank.iso_id, 'barchybrid.model')
    with open(params, 'r') as paramsfp:
        stored_vocab, stored_opt = pickle.load(paramsfp)

    parser = ArcHybridLSTM(stored_vocab, stored_opt)
    parser.Load(model_file)

    if predict:
        traindata = list(parser.Predict([treebank],"train",stored_opt))
        testdata = list(parser.Predict([treebank],"dev",stored_opt))
    else:
        for sentence in traindata:
            conll_sentence = [entry for entry in sentence if isinstance(entry, ConllEntry)]
            conll_sentence = conll_sentence[1:] + [conll_sentence[0]]
            parser.feature_extractor.getWordEmbeddings(conll_sentence,
                                                       False, stored_opt)
        for sentence in testdata:
            conll_sentence = [entry for entry in sentence if isinstance(entry, ConllEntry)]
            conll_sentence = conll_sentence[1:] + [conll_sentence[0]]
            parser.feature_extractor.getWordEmbeddings(conll_sentence,
                                                       False, stored_opt)
    return traindata, testdata

