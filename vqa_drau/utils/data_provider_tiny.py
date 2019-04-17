# -*- coding: utf-8 -*-
import numpy as np
import re, json, random
import config_small as config
import torch.utils.data as data
import spacy
import h5py
 
QID_KEY_SEPARATOR = '/'
ZERO_PAD = '_PAD'
GLOVE_EMBEDDING_SIZE = 300
class VQADataProvider:

    def __init__(self, opt, folder='./result', batchsize=64, max_length=15, mode='train'):
        self.opt = opt
        self.batchsize = batchsize
        self.d_vocabulary = None
        self.batch_index = None
        self.batch_len = None
        self.rev_adict = None
        self.max_length = max_length
        self.mode = mode
        self.qdic, self.adic = VQADataProvider.load_data(mode)

        with open('./%s/vdict.json'%folder,'r') as f:
            self.vdict = json.load(f)
        with open('./%s/adict.json'%folder,'r') as f:
            self.adict = json.load(f)

        self.n_ans_vocabulary = len(self.adict)
        self.nlp = spacy.load('en_vectors_web_lg')
        self.glove_dict = {} # word -> glove vector

    @staticmethod
    def load_vqa_json(data_split):
        """
        Parses the question and answer json files for the given data split. 
        Returns the question dictionary and the answer dictionary.
        """
        qdic, adic = {}, {}

        with open(config.DATA_PATHS[data_split]['ques_file'], 'r') as f:
            qdata = json.load(f)['questions']
            for q in qdata:
                qdic[data_split + QID_KEY_SEPARATOR + str(q['question_id'])] = \
                    {'qstr': q['question'], 'iid': q['image_id']}

        if 'test' not in data_split:
            with open(config.DATA_PATHS[data_split]['ans_file'], 'r') as f:
                adata = json.load(f)['annotations']
                for a in adata:
                    adic[data_split + QID_KEY_SEPARATOR + str(a['question_id'])] = \
                        a['answers']

        print ('parsed', len(qdic), 'questions for', data_split)
        return qdic, adic

    @staticmethod
    def load_genome_json():
        """
        Parses the genome json file. Returns the question dictionary and the
        answer dictionary.
        """
        qdic, adic = {}, {}

        with open(config.DATA_PATHS['genome']['genome_file'], 'r') as f:
            qdata = json.load(f)
            for q in qdata:
                key = 'genome' + QID_KEY_SEPARATOR + str(q['id'])
                qdic[key] = {'qstr': q['question'], 'iid': q['image']}
                adic[key] = [{'answer': q['answer']}]

        print ('parsed', len(qdic), 'questions for genome')
        return qdic, adic

    @staticmethod
    def load_data(data_split_str):
        all_qdic, all_adic = {}, {}
        for data_split in data_split_str.split('+'):
            assert data_split in config.DATA_PATHS.keys(), 'unknown data split'
            if data_split == 'genome':
                qdic, adic = VQADataProvider.load_genome_json()
                all_qdic.update(qdic)
                all_adic.update(adic)
            else:
                qdic, adic = VQADataProvider.load_vqa_json(data_split)
                all_qdic.update(qdic)
                all_adic.update(adic)
        return all_qdic, all_adic

    def getQuesIds(self):
        return self.qdic.keys()

    def getStrippedQuesId(self, qid):
        return qid.split(QID_KEY_SEPARATOR)[1]

    def getImgId(self,qid):
        return self.qdic[qid]['iid']

    def getQuesStr(self,qid):
        return self.qdic[qid]['qstr']

    def getAnsObj(self,qid):
        if self.mode == 'test-dev' or self.mode == 'test':
            return -1
        return self.adic[qid]

    @staticmethod
    def seq_to_list(s):
        t_str = s.lower()
        for i in [r'\?',r'\!',r'\'',r'\"',r'\$',r'\:',r'\@',r'\(',r'\)',r'\,',r'\.',r'\;']:
            t_str = re.sub( i, '', t_str)
        for i in [r'\-',r'\/']:
            t_str = re.sub( i, ' ', t_str)
        q_list = re.sub(r'\?','',t_str.lower()).split(' ')
        q_list = filter(lambda x: len(x) > 0, q_list)
        # py2.7 to py3 compatibility
        q_list = list(q_list)
        return q_list

    def extract_answer(self,answer_obj):
        """ Return the most popular answer in string."""
        if self.mode == 'test-dev' or self.mode == 'test':
            return -1
        answer_list = [ answer_obj[i]['answer'] for i in range(10)]
        dic = {}
        for ans in answer_list:
            if ans in dic:
                dic[ans] +=1
            else:
                dic[ans] = 1
        max_key = max((v,k) for (k,v) in dic.items())[1]
        return max_key

    def extract_answer_prob(self,answer_obj):
        """ Return the most popular answer in string."""
        if self.mode == 'test-dev' or self.mode == 'test':
            return -1

        answer_list = [ ans['answer'] for ans in answer_obj]
        prob_answer_list = []
        for ans in answer_list:
            if ans in self.adict:
                prob_answer_list.append(ans)
    def extract_answer_list(self,answer_obj):
        answer_list = [ ans['answer'] for ans in answer_obj]
        prob_answer_vec = np.zeros(self.opt.NUM_OUTPUT_UNITS)
        for ans in answer_list:
            if ans in self.adict:
                index = self.adict[ans]
                prob_answer_vec[index] += 1
        return prob_answer_vec / np.sum(prob_answer_vec)
 
        if len(prob_answer_list) == 0:
            if self.mode == 'val' or self.mode == 'test-dev' or self.mode == 'test':
                return 'hoge'
            else:
                raise Exception("This should not happen.")
        else:
            return random.choice(prob_answer_list)
 
    def qlist_to_vec(self, max_length, q_list):
        """
        Converts a list of words into a format suitable for the embedding layer.

        Arguments:
        max_length -- the maximum length of a question sequence
        q_list -- a list of words which are the tokens in the question

        Returns:
        qvec -- A max_length length vector containing one-hot indices for each word
        cvec -- A max_length length sequence continuation indicator vector
        """
        qvec = np.zeros(max_length)
        cvec = np.zeros(max_length)
        glove_matrix = np.zeros((max_length, GLOVE_EMBEDDING_SIZE))
        """  pad on the left   """
        # for i in xrange(max_length):
        #     if i < max_length - len(q_list):
        #         cvec[i] = 0
        #     else:
        #         w = q_list[i-(max_length-len(q_list))]
        #         # is the word in the vocabulary?
        #         if self.vdict.has_key(w) is False:
        #             w = ''
        #         qvec[i] = self.vdict[w]
        #         cvec[i] = 0 if i == max_length - len(q_list) else 1
        """  pad on the right   """
        for i in range(max_length):
            if i >= len(q_list):
                pass
            else:
                w = q_list[i]
                if w not in self.glove_dict:
                    self.glove_dict[w] = self.nlp(u'%s' % w).vector
                glove_matrix[i] = self.glove_dict[w]
                if not w in self.vdict:
                    w = ''
                qvec[i] = self.vdict[w]
                cvec[i] = 1 
        return qvec, cvec, glove_matrix
 
    def answer_to_vec(self, ans_str):
        """ Return answer id if the answer is included in vocabulary otherwise '' """
        if self.mode =='test-dev' or self.mode == 'test':
            return -1

        if ans_str in self.adict:
            ans = self.adict[ans_str]
        else:
            ans = self.adict['']
        return ans
 
    def vec_to_answer(self, ans_symbol):
        """ Return answer id if the answer is included in vocabulary otherwise '' """
        if self.rev_adict is None:
            rev_adict = {}
            for k,v in self.adict.items():
                rev_adict[v] = k
            self.rev_adict = rev_adict

        return self.rev_adict[ans_symbol]
 
    def create_batch(self,qid_list):

        qvec = (np.zeros(self.batchsize*self.max_length)).reshape(self.batchsize,self.max_length)
        cvec = (np.zeros(self.batchsize*self.max_length)).reshape(self.batchsize,self.max_length)
        ivec = np.zeros((self.batchsize, 2048, self.opt.IMG_FEAT_SIZE))
        if self.mode == 'val' or self.mode == 'test-dev' or self.mode == 'test':
            avec = np.zeros(self.batchsize)
        else:
            avec = np.zeros((self.batchsize, self.opt.NUM_OUTPUT_UNITS))
        glove_matrix = np.zeros((self.batchsize, self.max_length, GLOVE_EMBEDDING_SIZE))

        # Colab can't handle thousands of npz files, so we switched to hdf5.
        # Although opening and closing the file on every batch creation is inefficient,
        # colab forces us to do so since it freaks out with large files too.
        # data_split hack: used to be per element, but since we need to access a specific h5 file now, we need to define beforehand
        data_split = qid_list[0].split(QID_KEY_SEPARATOR)[0]
        with h5py.File(config.DATA_PATHS[data_split]['features'], "r") as feature_file:
         
            for i,qid in enumerate(qid_list):

                # load raw question information
                q_str = self.getQuesStr(qid)
                q_ans = self.getAnsObj(qid)
                q_iid = self.getImgId(qid)

                # convert question to vec
                q_list = VQADataProvider.seq_to_list(q_str)
                t_qvec, t_cvec, t_glove_matrix = self.qlist_to_vec(self.max_length, q_list)

                try:
                    qid_split = qid.split(QID_KEY_SEPARATOR)
                    data_split = qid_split[0]
                    if data_split == 'genome':
                        #t_ivec = feature_file[str(q_iid)][:]
                        t_ivec = np.load(config.DATA_PATHS['genome']['features_prefix'] + str(q_iid) + '.jpg.npz')['x']
                    else:
                        t_ivec = feature_file[str(q_iid)][:]
                        #t_ivec = np.load(config.DATA_PATHS[data_split]['features_prefix'] + str(q_iid) + '.npz')['x'] # my format
                        #t_ivec = np.load(config.DATA_PATHS[data_split]['features_prefix'] + str(q_iid).zfill(12) + '.jpg.npz')['x']
                   
                    #if self.opt.IMG_FEAT_TYPE == 'faster_rcnn_resnet_pool5':
                        # transpose FRCNN features cause of how I saved them.
                    #    t_ivec = np.transpose(t_ivec)
     
                    # reshape t_ivec to D x FEAT_SIZE
                    if len(t_ivec.shape) > 2:
                        t_ivec = t_ivec.reshape((2048, -1))
                    t_ivec = ( t_ivec / np.sqrt((t_ivec**2).sum()) )
                except:
                    t_ivec = 0.
                    print ('data not found for qid : ', q_iid,  self.mode)
                 
                # convert answer to vec
                if self.mode == 'val' or self.mode == 'test-dev' or self.mode == 'test':
                    q_ans_str = self.extract_answer(q_ans)
                    t_avec = self.answer_to_vec(q_ans_str)
                else:
                    t_avec = self.extract_answer_list(q_ans)
                qvec[i,...] = t_qvec
                cvec[i,...] = t_cvec
                ivec[i,:,0:t_ivec.shape[1]] = t_ivec
                avec[i,...] = t_avec
                glove_matrix[i,...] = t_glove_matrix

        return qvec, cvec, ivec, avec, glove_matrix

 
    def get_batch_vec(self):
        if self.batch_len is None:
            self.n_skipped = 0
            qid_list = self.getQuesIds()
            qid_list = list(qid_list) # old python2 code returns list. not the case with py3
            random.shuffle(qid_list)
            self.qid_list = qid_list
            self.batch_len = len(qid_list)
            self.batch_index = 0
            self.epoch_counter = 0

        def has_at_least_one_valid_answer(t_qid):
            answer_obj = self.getAnsObj(t_qid)
            answer_list = [ans['answer'] for ans in answer_obj]
            for ans in answer_list:
                if ans in self.adict:
                    return True

        counter = 0
        t_qid_list = []
        t_iid_list = []
        while counter < self.batchsize:
            t_qid = self.qid_list[self.batch_index]
            t_iid = self.getImgId(t_qid)
            if self.mode == 'val' or self.mode == 'test-dev' or self.mode == 'test':
                t_qid_list.append(t_qid)
                t_iid_list.append(t_iid)
                counter += 1
            elif has_at_least_one_valid_answer(t_qid):
                t_qid_list.append(t_qid)
                t_iid_list.append(t_iid)
                counter += 1
            else:
                self.n_skipped += 1 

            if self.batch_index < self.batch_len-1:
                self.batch_index += 1
            else:
                self.epoch_counter += 1
                qid_list = self.getQuesIds()
                qid_list = list(qid_list) # old python2 code returns list. not the case with py3
                random.shuffle(qid_list)
                self.qid_list = qid_list
                self.batch_index = 0
                print("%d questions were skipped in a single epoch" % self.n_skipped)
                self.n_skipped = 0

        t_batch = self.create_batch(t_qid_list)
        return t_batch + (t_qid_list, t_iid_list, self.epoch_counter)

class VQADataset(data.Dataset):

    def __init__(self, mode, batchsize, folder, opt):
        self.batchsize = batchsize 
        self.mode = mode 
        self.folder = folder
        self.opt = opt 
        if self.mode == 'val' or self.mode == 'test-dev' or self.mode == 'test':
            pass
        else:
            self.dp = VQADataProvider(opt, batchsize=self.batchsize, mode=self.mode, folder=self.folder)

    def __getitem__(self, idx):
        if self.mode == 'val' or self.mode == 'test-dev' or self.mode == 'test':
            pass
        else:
            word, cont, feature, answer, glove_matrix, _, _, epoch = self.dp.get_batch_vec()
        word_length = np.sum(cont, axis=1)
        return word, word_length, feature, answer, glove_matrix, epoch

    def __len__(self):
        # TODO: fix this from iteration len to dataset len
        # need to change get_batch_vec to get_sample_vec
        return self.opt.MAX_ITERATIONS
