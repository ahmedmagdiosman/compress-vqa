import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import sys
sys.path.append("..")
from . import cbp                          # pytorch ver >=0.4

class drau_glove(nn.Module):
    def __init__(self, opt):
        super(drau_glove, self).__init__()
        
        def conv_prelu(inp, out):
            conv_prelu_block = nn.Sequential()

            conv_prelu_block.add_module("conv", nn.Conv2d(in_channels=inp, out_channels=out,
                kernel_size=1, padding=0, bias=True))
            conv_prelu_block.add_module("conv_prelu", nn.PReLU())

            return conv_prelu_block

        def linear_prelu(inp, out):
            linear_prelu_block = nn.Sequential()

            linear_prelu_block.add_module("linear", nn.Linear(in_features=inp, out_features=out))
            linear_prelu_block.add_module("linear_prelu", nn.PReLU())

            return linear_prelu_block

        # common layers and options
        self.opt = opt
        self.JOINT_EMBEDDING_SIZE = opt.JOINT_EMBEDDING_SIZE            # 1024
        self.MAX_QUESTION_SIZE = opt.MAX_WORDS_IN_QUESTION              # T
        self.IMG_FEAT_SIZE = opt.IMG_FEAT_SIZE                          # K
        self.IMG_CHANNEL_SIZE = opt.IMAGE_CHANNEL                       # 2048
        self.QUESTION_VOCAB_SIZE = opt.quest_vob_size                   
        self.LSTM_CHANNEL_NUM = opt.LSTM_UNIT_NUM                       # 1024
        self.NUM_IMG_GLIMPSE = opt.NUM_IMG_GLIMPSE                      # 2
        self.NUM_QUESTION_GLIMPSE = opt.NUM_QUESTION_GLIMPSE            # 2
        self.FUSION_OUT_DIM = opt.FUSION_OUT_DIM                        # 16000    
        self.NUM_OUTPUT_UNITS = opt.NUM_OUTPUT_UNITS                    # 3000

        self.Dropout_LSTM = nn.Dropout(p=opt.LSTM_DROPOUT_RATIO)        # 0.3
        self.Dropout_Fusion = nn.Dropout(p=opt.FUSION_DROPOUT_RATIO)    # 0.1
        self.Att_Softmax = nn.Softmax(dim=2)

        # question features
        self.Embedding = nn.Embedding(self.QUESTION_VOCAB_SIZE, 300)
        self.Tanh = nn.Tanh()

        self.Q_LSTM1 = nn.LSTM(input_size=300*2, hidden_size=self.LSTM_CHANNEL_NUM, batch_first=False)
        # separate layers to avoid pytorch warning about last layer dropout
        self.Q_LSTM2 = nn.LSTM(input_size=self.LSTM_CHANNEL_NUM, hidden_size=self.LSTM_CHANNEL_NUM, batch_first=False)
       
        self.Conv1_i_proj = conv_prelu(self.IMG_CHANNEL_SIZE, self.JOINT_EMBEDDING_SIZE)                # 2048 -> 1024
        self.Conv2_i_proj = conv_prelu(self.IMG_FEAT_SIZE, self.MAX_QUESTION_SIZE)                      # K -> T

        self.Conv1_q_proj = conv_prelu(self.JOINT_EMBEDDING_SIZE*2, self.JOINT_EMBEDDING_SIZE)          # 2048 -> 1024
        self.Conv2_q_proj = conv_prelu(self.MAX_QUESTION_SIZE, self.IMG_FEAT_SIZE)                      # T -> K
        
        # RAU components
        self.QAtt_LSTM = nn.LSTM(input_size=self.JOINT_EMBEDDING_SIZE, hidden_size=self.JOINT_EMBEDDING_SIZE*2, batch_first=False)
        self.IAtt_LSTM = nn.LSTM(input_size=self.JOINT_EMBEDDING_SIZE, hidden_size=self.JOINT_EMBEDDING_SIZE*2, batch_first=False)

        self.Conv1_RTAU = conv_prelu(2,1)
        self.Conv2_RTAU = conv_prelu(self.JOINT_EMBEDDING_SIZE*2, self.NUM_QUESTION_GLIMPSE) 
        self.Conv1_RVAU = conv_prelu(2,1)
        self.Conv2_RVAU = conv_prelu(self.JOINT_EMBEDDING_SIZE*2, self.NUM_IMG_GLIMPSE)

        self.Linear_RTAU = linear_prelu(self.LSTM_CHANNEL_NUM*2*self.NUM_QUESTION_GLIMPSE,self.LSTM_CHANNEL_NUM*2)
        self.Linear_RVAU = linear_prelu(self.IMG_CHANNEL_SIZE* self.NUM_IMG_GLIMPSE, self.IMG_CHANNEL_SIZE)

        # fusion
        self.mcb = cbp.CompactBilinearPooling(self.JOINT_EMBEDDING_SIZE*2, self.JOINT_EMBEDDING_SIZE*2, self.FUSION_OUT_DIM)
        # classifier
        self.Linear_Classifier = nn.Linear(self.FUSION_OUT_DIM, self.NUM_OUTPUT_UNITS)


    def _forward_imgfeat(self, img_feature):
        img_feat = img_feature.unsqueeze(3)                                # N x 2048 x K x 1

        return img_feat

    def _forward_qfeat(self, question, glove):
        q = torch.transpose(question, 1, 0)                              # type Longtensor, T x N
        glove = glove.permute(1, 0, 2)                                   # type float, T x N x 300
        
        q_emb = self.Tanh(self.Embedding(q))                             # T x N x 300
        q_emb = torch.cat((q_emb, glove), 2)                             # T x N x 600
        
        q_lstm1, _ = self.Q_LSTM1(q_emb)                                 # T x N x 1024
        q_lstm1 = self.Dropout_LSTM(q_lstm1)
        q_lstm2, _ = self.Q_LSTM2(q_lstm1)                               # T x N x 1024
        q_lstm2 = self.Dropout_LSTM(q_lstm2)

        q_lstm = torch.cat((q_lstm1,q_lstm2),2)
        q_lstm = q_lstm.permute(1, 2, 0)                                 # N x 2048 x T
       
        q_feat = q_lstm.unsqueeze(3)                                     # N x 2048 x T x 1

        return q_feat

    def _forward_combine(self, q_feat, img_feat):
        
        conv_i_proj = self.Conv1_i_proj(img_feat)                        # N x 1024 x K x 1
        conv_i_q_proj = self.Conv2_i_proj(conv_i_proj.permute(0,2,3,1))  # N x T x 1 x 1024
        conv_i_q_proj = conv_i_q_proj.permute(0,3,1,2)                   # N x 1024 x T x 1    
    
        conv_q_proj = self.Conv1_q_proj(q_feat)                          # N x 1024 x T x 1
        conv_q_i_proj = self.Conv2_q_proj(conv_q_proj.permute(0,2,3,1))  # N x K x 1 x 1024
        conv_q_i_proj = conv_q_i_proj.permute(0,3,1,2)                   # N x 1024 x K x 1
        
        rvau_input = torch.cat((conv_i_proj, conv_q_i_proj), 3)          # N x 1024 x K x 2
        rvau_input = rvau_input.permute(0,3,1,2)                         # N x 2 x 1024 x K

        rtau_input = torch.cat((conv_q_proj, conv_i_q_proj), 3)          # N x 1024 x T x 2
        rtau_input = rtau_input.permute(0,3,1,2)                         # N x 2 x 1024 x T        

        return rvau_input, rtau_input
    
    def _forward_RVAU(self, rvau_input, img_feat):
       
        conv1_rvau = self.Conv1_RVAU(rvau_input)                         # N x 1 x 1024 x K
        conv1_rvau = conv1_rvau.squeeze().permute(2,0,1)                 # K x N x 1024
        
        iatt_lstm, _ = self.IAtt_LSTM(conv1_rvau)                        # K x N x 2048
        iatt_lstm = self.Dropout_LSTM(iatt_lstm).permute(1,2,0)          # N x 2048 x K
        
        conv2_rvau = self.Conv2_RVAU(iatt_lstm.unsqueeze(3)).squeeze()   # N x IG x K
        iatt_maps = self.Att_Softmax(conv2_rvau)                         # N x IG x %K
        
        iatt_maps_list = iatt_maps.unbind(dim=1)                         # IG x [N x %K]
        
        img_feat = img_feat.squeeze()                                    # N x 2048 x K
        iatts = []
        for att_map in iatt_maps_list:
            att_map = att_map.unsqueeze(1).expand_as(img_feat)           # N x 2048(replicated) x %K
            att = att_map * img_feat                                     # N x 2048 x K
            att = att.sum(dim=2)                                         # N x 2048
            iatts.append(att)
        iatt = torch.cat(iatts,dim=1)                                    # N x 4096
        
        rvau_out = self.Linear_RVAU(iatt)                                # N x 2048

        return rvau_out

    def _forward_RTAU(self, rtau_input, q_feat):

        conv1_rtau = self.Conv1_RTAU(rtau_input)                         # N x 1 x 1024 x T
        conv1_rtau = conv1_rtau.squeeze().permute(2,0,1)                 # T x N x 1024
        
        qatt_lstm, _ = self.QAtt_LSTM(conv1_rtau)                        # T x N x 2048
        qatt_lstm = self.Dropout_LSTM(qatt_lstm).permute(1,2,0)          # N x 2048 x T
        
        conv2_rtau = self.Conv2_RTAU(qatt_lstm.unsqueeze(3)).squeeze()   # N x TG x T
        qatt_maps = self.Att_Softmax(conv2_rtau)                         # N x TG x %T
        
        qatt_maps_list = qatt_maps.unbind(dim=1)                         # TG x [N x %T]
        
        q_feat = q_feat.squeeze()                                        # N x 2048 x T
        qatts = []
        for att_map in qatt_maps_list:
            att_map = att_map.unsqueeze(1).expand_as(q_feat)             # N x 2048(replicated) x %T
            att = att_map * q_feat                                       # N x 2048 x T
            att = att.sum(dim=2)                                         # N x 2048
            qatts.append(att)
        qatt = torch.cat(qatts,dim=1)                                    # N x 4096
        
        rtau_out = self.Linear_RTAU(qatt)                                # N x 2048
        
        return rtau_out
           
    
    def _forward_fusion(self, rvau_out, rtau_out):
        eps = 1e-12
        
        # fusion
        mcb_out = self.mcb(rvau_out, rtau_out)                           # N x 16000
        mcb_signedsqrt = torch.mul(torch.sign(mcb_out), torch.sqrt(torch.abs(mcb_out)+eps))
        mcb_l2norm = F.normalize(mcb_signedsqrt)
        mcb_dropout = self.Dropout_Fusion(mcb_l2norm)
        
        return mcb_dropout
     
    def forward(self, data, word_length, img_feature, glove):
       
        img_feat = self._forward_imgfeat(img_feature)
        q_feat = self._forward_qfeat(data,glove)
        
        rvau_input, rtau_input = self._forward_combine(q_feat, img_feat)
        rvau_out = self._forward_RVAU(rvau_input,img_feat)
        rtau_out = self._forward_RTAU(rtau_input,q_feat)

        mcb_out = self._forward_fusion(rvau_out,rtau_out)
        
        prediction = self.Linear_Classifier(mcb_out)
        prediction = F.log_softmax(prediction, dim=1)

        return prediction
