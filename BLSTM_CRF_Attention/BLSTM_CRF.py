import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from data_utils import get_embedding_weight
from torch.autograd import Variable
torch.manual_seed(1)
START_TAG = "<START>"
STOP_TAG = "<STOP>"

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 将重复的代码剥离出来
#
def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


# 创建模型
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        # self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        # weight = get_embedding_weight()
        # self.word_embeds = nn.Embedding.from_pretrained(weight)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)
        # 256 --> 5 {B, I, O, START_TAG, STOP_TAG}
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        # 5 * 5 的矩阵
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size)).to(device)
        # 这两个语句执行了一个约束，即我们从不向开始标记转移，也从不从停止标记转移
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000
        self.hidden = self.init_hidden()

        ## attn
        self.w_omega = Variable(torch.zeros(self.hidden_dim, self.hidden_dim)).to(device)
        self.u_omega = Variable(torch.zeros(self.hidden_dim)).to(device)

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2).to(device),
                torch.randn(2, 1, self.hidden_dim // 2).to(device))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function(求配分函数的前向算法)
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.
        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas.to(device)
        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size).to(device)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1).to(device)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def attention_net(self, lstm_out):
        # (sequence_len * batch_size, hidden*layer_num)  here, the code top is hidden_dim // 2
        out_reshape = torch.Tensor.reshape(lstm_out, [-1, self.hidden_dim])

        # (sequence_len * batch_size, attn_size)   here attn_size = hidden_size
        attn_tanh = torch.tanh(torch.mm(out_reshape, self.w_omega))

        # (sequence_len * batch_size, 1)
        attn_hidden_layer = torch.mm(attn_tanh, torch.Tensor.reshape(self.u_omega, [-1, 1]))

        # (batch_size, sequence_len)
        alphas = F.softmax(torch.Tensor.reshape(attn_hidden_layer, [-1, attn_hidden_layer.size()[0]]))

        # (batch_size, sequence_len, 1)
        alphas_reshape = torch.Tensor.reshape(alphas, [-1, alphas.size()[1], 1])

        # (batch_size, sequence_len, hidden*layer_num)
        state = lstm_out.permute(1, 0, 2)

        # (batch_size, hidden_size*layer_num)
        # attn_out = torch.sum(state * alphas_reshape, 1)

        # [batch, sequence, hidden_dim]
        attn_out = state * alphas_reshape
        return attn_out

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1).to(device)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)

        attn_output = self.attention_net(lstm_out)

        lstm_attn_output = attn_output.view(len(sentence), self.hidden_dim).to(device)
        lstm_feats = self.hidden2tag(lstm_attn_output).to(device)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1).to(device)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long).to(device), tags]).to(device)
        for i, feat in enumerate(feats):
            score = score + \
                    self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)  # tensor([[-10000., -10000., -10000., -10000., -10000.]])
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0 # tensor([[-10000., -10000., -10000.,      0., -10000.]])

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars # tensor([[-10000., -10000., -10000.,      0., -10000.]])
        for feat in feats:   #
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size): # [0, 1, 2, 3, 4]
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag].to('cpu')
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]].to('cpu')
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats).to(device)
        gold_score = self._score_sentence(feats, tags).to(device)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats.to('cpu'))
        return score, tag_seq

