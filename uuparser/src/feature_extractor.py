from bilstm import BiLSTM, LSTM
import utils
import dynet as dy
import random
from collections import defaultdict
import codecs, re, os

class FeatureExtractor(object):
    def __init__(self,model,options,vocab,nnvecs):

        self.word_counts, words, chars, pos, cpos, self.irels, treebanks, langs = vocab

        self.model = model
        self.nnvecs = nnvecs

        extra_words = 2 # MLP padding vector and OOV vector
        self.words = {word: ind for ind, word in enumerate(words,extra_words)}
        #why not just len(self.words)?
        self.word_lookup = self.model.add_lookup_parameters((len(self.words)+extra_words, options.word_emb_size))

        extra_pos = 2 # MLP padding vector and OOV vector
        self.pos = {pos: ind for ind, pos in enumerate(cpos,extra_pos)}
        self.pos_lookup = self.model.add_lookup_parameters((len(cpos)+extra_pos, options.pos_emb_size))

        extra_chars = 1 # OOV vector
        self.chars = {char: ind for ind, char in enumerate(chars,extra_chars)}
        self.char_lookup = self.model.add_lookup_parameters((len(chars)+extra_chars, options.char_emb_size))

        extra_treebanks = 1 # Padding vector
        self.treebanks = {treebank: ind for ind, treebank in enumerate(treebanks,extra_treebanks)}
        self.treebank_lookup = self.model.add_lookup_parameters((len(treebanks)+extra_treebanks, options.tbank_emb_size))

        # initialise word vectors with external embeddings where they exist
        if (options.ext_emb_dir or options.ext_emb_file) and not options.predict:
            self.external_embedding = defaultdict(lambda:{})
            for lang in langs:
                if options.word_emb_size > 0:
                    self.external_embedding["words"].update(utils.get_external_embeddings(options,lang,self.words.viewkeys()))
                if options.char_emb_size > 0:
                    self.external_embedding["chars"].update(utils.get_external_embeddings(options,lang,self.chars,chars=True))
            self.init_lookups(options)

        self.lstm_input_size = options.word_emb_size + options.pos_emb_size + options.tbank_emb_size +\
            2* (options.char_lstm_output_size if options.char_emb_size > 0 else 0)

        print "Word-level LSTM input size: " + str(self.lstm_input_size)

        self.bilstms = []
        if options.no_bilstms > 0:
            if options.unidir_lstm is not None:
                #replace the BiLSTMs with unidirectional ones
                #it's ugly to still call it bilstm but easier
                if options.unidir_lstm:
                    self.bilstms.append(LSTM(self.lstm_input_size, 2*options.lstm_output_size, self.model,
                                               dropout_rate=0.33,
                                               direction=options.unidir_lstm,
                                               layers= options.no_bilstms))
            else:
                self.bilstms.append(BiLSTM(self.lstm_input_size, options.lstm_output_size, self.model,
                                      dropout_rate=0.33))
                for i in range(1,options.no_bilstms):
                    self.bilstms.append(BiLSTM(2*options.lstm_output_size,
                                      options.lstm_output_size, self.model,
                                      dropout_rate=0.33))
            #used in the PaddingVec
            self.word2lstm = self.model.add_parameters((options.lstm_output_size*2, self.lstm_input_size))
            self.word2lstmbias = self.model.add_parameters((options.lstm_output_size*2))
        else:
            self.word2lstm = self.model.add_parameters((self.lstm_input_size, self.lstm_input_size))
            self.word2lstmbias = self.model.add_parameters((self.lstm_input_size))

        self.char_bilstm = BiLSTM(options.char_emb_size,
                                  options.char_lstm_output_size, self.model,
                                  dropout_rate=0.33)

        self.charPadding = self.model.add_parameters((options.char_lstm_output_size*2))

        #recursive composition things
        if options.use_recursive_composition:
            deprel_dir = [(rel,direction) for rel in self.irels for direction in [0,1]]
            extra_deprel = 1 # padding rel vec
            #this does not work
            self.ideprel_dir = {val:ind for ind,val in enumerate(deprel_dir, extra_deprel)}
            self.deprel_lookup = self.model.add_lookup_parameters((len(self.ideprel_dir)+extra_deprel, options.deprel_size))
            lstm_out_dim = options.lstm_output_size*2
            if options.use_recursive_composition == 'RecNN':
                self.hCompos = self.model.add_parameters((lstm_out_dim,lstm_out_dim))
                self.dCompos = self.model.add_parameters((lstm_out_dim,lstm_out_dim))
                self.rCompos = self.model.add_parameters((lstm_out_dim,options.deprel_size))
                self.biasCompos = self.model.add_parameters((lstm_out_dim))
            else:
                compos_in_dim = lstm_out_dim*2 + options.deprel_size
                self.composLSTM = dy.VanillaLSTMBuilder(1, compos_in_dim, lstm_out_dim, self.model)

    def Init(self,options):
        paddingWordVec = self.word_lookup[1] if options.word_emb_size > 0 else None
        paddingPosVec = self.pos_lookup[1] if options.pos_emb_size > 0 else None
        paddingCharVec = self.charPadding.expr() if options.char_emb_size > 0 else None
        paddingTbankVec = self.treebank_lookup[0] if options.tbank_emb_size > 0 else None

        self.paddingVec = dy.tanh(self.word2lstm.expr() *\
            dy.concatenate(filter(None,[paddingWordVec,
                                        paddingPosVec,
                                        paddingCharVec,
                                        paddingTbankVec])) + self.word2lstmbias.expr())

        self.empty = self.paddingVec if self.nnvecs == 1 else\
            dy.concatenate([self.paddingVec for _ in xrange(self.nnvecs)])

    def getWordEmbeddings(self, sentence, train, options, test_embeddings=defaultdict(lambda:{})):
        for root in sentence:
            root.vecs = defaultdict(lambda: None) # all vecs are None by default (possibly a little risky?)
            if options.word_emb_size > 0:
                if train:
                    word_count = float(self.word_counts.get(root.norm, 0))
                    dropFlag = random.random() > word_count/(0.25+word_count)
                    root.vecs["word"] = self.word_lookup[self.words.get(root.norm, 0) if not dropFlag else 0]
                else: # need to check in test_embeddings at prediction time
                    if root.norm in self.words:
                        root.vecs["word"] = self.word_lookup[self.words[root.norm]]
                    elif root.norm in test_embeddings["words"]:
                        root.vecs["word"] = dy.inputVector(test_embeddings["words"][root.norm])
                    else:
                        root.vecs["word"] = self.word_lookup[0]
            if options.pos_emb_size > 0:
                root.vecs["pos"] = self.pos_lookup[self.pos.get(root.cpos,0)]
            if options.char_emb_size > 0:
                root.vecs["char"] = self.get_char_vector(root,train,test_embeddings["chars"])
            if options.tbank_emb_size > 0:
                if options.forced_tbank_emb:
                    treebank_id = options.forced_tbank_emb
                elif root.proxy_tbank:
                    treebank_id = root.proxy_tbank
                else:
                    treebank_id = root.treebank_id
                # this is a bit of a hack for models trained on an old version of the code
                # that used treebank name rather than id as the lookup
                if not treebank_id in self.treebanks and treebank_id in utils.reverse_iso_dict and \
                    utils.reverse_iso_dict[treebank_id] in self.treebanks:
                    treebank_id = utils.reverse_iso_dict[treebank_id]
                root.vecs["treebank"] = self.treebank_lookup[self.treebanks[treebank_id]]

            root.vec = dy.concatenate(filter(None, [root.vecs["word"],
                                                        root.vecs["pos"],
                                                        root.vecs["char"],
                                                        root.vecs["treebank"]]))

        for bilstm in self.bilstms:
            bilstm.set_token_vecs(sentence,train)

        #save the bilstm vector - if I call .vec.value() outside of here it has a weird ass
        #behaviour and can somehow get all vec values to 0
        for root in sentence:
            root.contextual_vector = root.vec.value()


    def get_char_vector(self,root,train,test_embeddings_chars={}):

        if root.char_rep == "*root*": # no point running a character analysis over this placeholder token
            return self.charPadding.expr() # use the padding vector if it's the root token
        else:
            char_vecs = []
            for char in root.char_rep:
                if char in self.chars:
                    char_vecs.append(self.char_lookup[self.chars[char]])
                elif char in test_embeddings_chars:
                    char_vecs.append(dy.inputVector(test_embeddings_chars[char]))
                else:
                    char_vecs.append(self.char_lookup[0])
            return self.char_bilstm.get_sequence_vector(char_vecs,train)

    def init_lookups(self,options):

        if self.external_embedding["words"]:
            print 'Initialising %i word vectors with external embeddings'%len(self.external_embedding["words"])
            for word in self.external_embedding["words"]:
                if len(self.external_embedding["words"][word]) != options.word_emb_size:
                    raise Exception("Size of external embedding does not match specified word embedding size of %s"%(options.word_emb_size))
                self.word_lookup.init_row(self.words[word],self.external_embedding["words"][word])
        elif options.word_emb_size > 0:
            print 'No word external embeddings found: all vectors initialised randomly'

        if self.external_embedding["chars"]:
            print 'Initialising %i char vectors with external embeddings'%len(self.external_embedding["chars"])
            for char in self.external_embedding["chars"]:
                if len(self.external_embedding["chars"][char]) != options.char_emb_size:
                    raise Exception("Size of external embedding does not match specified char embedding size of %s"%(options.char_emb_size))
                self.char_lookup.init_row(self.chars[char],self.external_embedding["chars"][char])
        elif options.char_emb_size > 0:
            print 'No character external embeddings found: all vectors initialised randomly'
