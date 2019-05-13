import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random

from TransformerBlock import TransformerDecoderBlock as DecoderBlock
from TransformerBlock import get_sinusoid_encoding_table
# from bert_embedding import get_bert_embedding
from layers import BERT, LinearProjection, LinearNet_TwoLayer, FFN, PostNet, LinearProjection
from text.symbols import symbols
import hparams as hp
# print(len(symbols))
from module import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device("cpu")

if_parallel = True


class WESS_Encoder(nn.Module):
    """
    Encoder
    (pre-transformer replaced by GRU)
    """

    def __init__(self,
                 #  vocab_max_size=2000,
                 embedding_size=256,
                 GRU_hidden_size=768,
                 GRU_num_layers=1,
                 GRU_batch_first=True,
                 GRU_bidirectional=True,
                 #  bert_prenet_hidden=1024,
                 #  bert_prenet_output=256,
                 CBHG_prenet_hidden=1024,
                 CBHG_prenet_output=256,
                 #  bert_hidden=256,
                 #  bert_n_layers=2,
                 #  bert_attn_heads=4,
                 #  embedding_postnet_hidden=1024,
                 #  embedding_postnet_output=256,
                 embeddingnet_hidden=256,
                 embeddingnet_num_layer=1,
                 embeddingnet_batch_first=True,
                 embeddingnet_bi=True,
                 dropout=0.1):
        """
        :param encoder_hparams
        """

        super(WESS_Encoder, self).__init__()
        # self.vocab_max_size = vocab_max_size
        self.embedding_size = embedding_size
        self.GRU_hidden = GRU_hidden_size
        self.GRU_num_layers = GRU_num_layers
        self.GRU_batch_first = GRU_batch_first
        self.GRU_bidirectional = GRU_bidirectional
        self.cbhg_prenet_hidden = CBHG_prenet_hidden
        self.cbhg_prenet_output = CBHG_prenet_output
        # self.bert_hidden = bert_hidden
        # self.bert_n_layers = bert_n_layers
        # self.bert_attn_heads = bert_attn_heads
        # self.embedding_postnet_hidden = embedding_postnet_hidden
        # self.embedding_postnet_output = embedding_postnet_output
        self.embeddingnet_hidden = embeddingnet_hidden
        self.embeddingnet_num_layer = embeddingnet_num_layer
        self.embeddingnet_batch_first = embeddingnet_batch_first
        self.embeddingnet_bi = embeddingnet_bi
        self.dropout = dropout

        # Embeddings
        self.pre_embedding = nn.Embedding(len(symbols)+1, self.embedding_size)
        # self.position_embedding = nn.Embedding.from_pretrained(
        #     get_sinusoid_encoding_table(self.vocab_max_size, self.embedding_size), freeze=True)

        # self.pre_GRU = nn.GRU(input_size=self.embedding_size,
        #                       hidden_size=self.GRU_hidden,
        #                       num_layers=self.GRU_num_layers,
        #                       batch_first=self.GRU_batch_first,
        #                       dropout=self.dropout,
        #                       bidirectional=self.GRU_bidirectional)

        self.pre_GRU = nn.GRU(input_size=self.embedding_size,
                              hidden_size=self.GRU_hidden,
                              num_layers=self.GRU_num_layers,
                              batch_first=self.GRU_batch_first,
                              bidirectional=self.GRU_bidirectional)

        self.CBHG_prenet = LinearNet_TwoLayer(
            self.GRU_hidden, self.cbhg_prenet_hidden, self.cbhg_prenet_output)

        # self.bi_Transformer = BERT(hidden=self.bert_hidden,
        #                            n_layers=self.bert_n_layers,
        #                            attn_heads=self.bert_attn_heads,
        #                            dropout=self.dropout)

        # self.EmbeddingNet = LinearNet_TwoLayer(
        #     self.embedding_size, self.embedding_postnet_hidden, self.embedding_postnet_output)

        self.CBHG = CBHG()

        self.EmbeddingNet = nn.GRU(input_size=self.cbhg_prenet_output,
                                   hidden_size=self.embeddingnet_hidden,
                                   num_layers=self.embeddingnet_num_layer,
                                   batch_first=self.embeddingnet_batch_first,
                                   bidirectional=self.embeddingnet_bi)

    def init_GRU_hidden(self, batch_size, num_layers, hidden_size):
        if self.GRU_bidirectional:
            return torch.zeros(num_layers*2, batch_size,  hidden_size).to(device)
        else:
            return torch.zeros(num_layers*1, batch_size,  hidden_size).to(device)

    def get_GRU_embedding(self, GRU_output):
        # print(GRU_output.size())
        out_1 = GRU_output[:, 0:1, :]
        out_2 = GRU_output[:, GRU_output.size(1)-1:, :]

        out = out_1 + out_2
        out = out[:, :, 0:out.size(2)//2] + out[:, :, out.size(2)//2:]

        # print(out.size())
        return out

    def get_GRU_output(self, GRU_output):
        out = GRU_output
        out = out[:, :, 0:out.size(2)//2] + out[:, :, out.size(2)//2:]

        # print(out.size())
        return out

    def cal_P_GRU(self, batch, gate_for_words_batch):
        list_input = list()
        list_output = list()

        for ind in range(len(gate_for_words_batch)-1):
            list_input.append(
                batch[gate_for_words_batch[ind]:gate_for_words_batch[ind+1]])

        # print(len(list_input))
        for one_word in list_input:
            one_word = torch.stack([one_word])

            # pos_input = torch.Tensor(
            #     [i for i in range(one_word.size(1))]).long().to(device)
            # position_embedding = self.position_embedding(pos_input)
            # position_embedding = position_embedding.unsqueeze(0)

            # one_word = one_word + position_embedding
            # output_one_word = self.P_transformer_block(one_word)
            # print(output_one_word.size())

            # self.pre_GRU.flatten_parameters()
            output_one_word = self.pre_GRU(one_word)[0]
            output_one_word = self.get_GRU_embedding(output_one_word)
            word = output_one_word.squeeze(0)
            # word = output_one_word[output_one_word.size()[0]-1]
            list_output.append(word)

        output = torch.stack(list_output)
        output = output.squeeze(1)
        # print(output.size())

        return output

    # def pad_by_word(self, words_batch):
    #     len_arr = np.array(list())
    #     for ele in words_batch:
    #         len_arr = np.append(len_arr, ele.size(0))
    #     max_size = int(len_arr.max())
    #     # print(max_size)

    #     def pad(tensor, target_length):
    #         embedding_size = tensor.size(1)
    #         pad_tensor = torch.zeros(1, embedding_size).to(device)

    #         for i in range(target_length-tensor.size(0)):
    #             tensor = torch.cat((tensor, pad_tensor))

    #         return tensor

    #     padded = list()
    #     for one_batch in words_batch:
    #         one_batch = pad(one_batch, max_size)
    #         padded.append(one_batch)
    #     padded = torch.stack(padded)

    #     return padded

    # def pad_all(self, word_batch, embeddings):
    #     # print(word_batch.size())
    #     # print(embeddings.size())
    #     if word_batch.size(1) == embeddings.size(1):
    #         return word_batch, embeddings

    #     if word_batch.size(1) > embeddings.size(1):
    #         pad_len = word_batch.size(1) - embeddings.size(1)
    #         pad_vec = torch.zeros(word_batch.size(
    #             0), pad_len, embeddings.size(2)).float().to(device)
    #         embeddings = torch.cat((embeddings, pad_vec), 1)
    #         return word_batch, embeddings

    #     if word_batch.size(1) < embeddings.size(1):
    #         pad_len = embeddings.size(1) - word_batch.size(1)
    #         pad_vec = torch.zeros(word_batch.size(
    #             0), pad_len, embeddings.size(2)).float().to(device)
    #         word_batch = torch.cat((word_batch, pad_vec), 1)
    #         return word_batch, embeddings

    def pad_bert_embedding_and_GRU_embedding(self, bert_embedding, GRU_embedding):
        """One batch"""

        len_bert = len(bert_embedding)
        len_GRU = GRU_embedding.size(0)
        max_len = max(len_bert, len_GRU)

        # pad_embedding = torch.zeros(1, GRU_embedding.size(1)).to(device)

        # for i in range(max_len - len_GRU):
        #     GRU_embedding = torch.cat((GRU_embedding, pad_embedding), 0)

        # for i in range(max_len - len_bert):
        #     bert_embedding = torch.cat((bert_embedding, pad_embedding), 0)

        # print(GRU_embedding.size())
        # print(bert_embedding.size())

        GRU_embedding = torch.cat((GRU_embedding, torch.zeros(
            max_len - len_GRU, GRU_embedding.size(1)).to(device)), 0)
        bert_embedding = torch.cat((bert_embedding, torch.zeros(
            max_len - len_bert, GRU_embedding.size(1)).to(device)), 0)

        # output = bert_embedding + GRU_embedding

        return bert_embedding, GRU_embedding

    def pad_all(self, bert_transformer_input):
        len_list = list()
        for batch in bert_transformer_input:
            len_list.append(batch.size(0))

        max_len = max(len_list)

        # pad_embedding = torch.zeros(
        #     1, bert_transformer_input[0].size(1)).to(device)

        for index, batch in enumerate(bert_transformer_input):
            bert_transformer_input[index] = torch.cat((bert_transformer_input[index], torch.zeros(
                max_len - bert_transformer_input[index].size(0), bert_transformer_input[index].size(1)).to(device)), 0)

        bert_transformer_input = torch.stack(bert_transformer_input)

        return bert_transformer_input

    def forward(self, x, bert_embeddings, gate_for_words, indexs_list):
        """
        :param: x: (batch, length)
        :param: bert_embeddings: (batch, length, 768)
        :param: gate_for_words: (batch, indexs)
        """

        # Embedding
        x = self.pre_embedding(x)
        # print("x:", x.size())

        # P_GRU
        words_batch = list()
        for index, batch in enumerate(x):
            words_batch.append(self.cal_P_GRU(batch, gate_for_words[index]))

        # words_batch = self.pad_by_word(words_batch)
        # bert_embeddings = self.pad_by_word(bert_embeddings)
        # words_batch, bert_embeddings = self.pad_all(
        #     words_batch, bert_embeddings)
        # # print(words_batch.size())
        # # print(bert_embeddings.size())
        # bert_input = words_batch + bert_embeddings

        # # Add Position Embedding
        # pos_input = torch.stack([torch.Tensor([i for i in range(
        #     bert_input.size(1))]).long() for i in range(bert_input.size(0))]).to(device)
        # pos_embedding = self.position_embedding(pos_input)
        # bert_input = bert_input + pos_embedding

        # encoder_output = self.bert_encoder(bert_input)

        # New

        # If you need data parallel, you will need some extra processing.
        if self.training and if_parallel:
            # print(indexs_list)
            bert_embeddings = [bert_embeddings[i] for i in indexs_list]

        if not (len(words_batch) == len(bert_embeddings)):
            raise ValueError(
                "the length of bert embeddings is not equal to the length of GRU embeddings.")

        bert_transformer_input = list()

        for batch_index in range(len(words_batch)):

            # print("bert_embedding:", bert_embeddings[batch_index])
            # print(words_batch[batch_index])

            bert_embedding, GRU_embedding = self.pad_bert_embedding_and_GRU_embedding(
                bert_embeddings[batch_index], words_batch[batch_index])
            add_bert_GRU = bert_embedding + GRU_embedding
            bert_transformer_input.append(add_bert_GRU)

        bert_transformer_input = self.pad_all(bert_transformer_input)

        # pos_input_one_batch = torch.Tensor(
        #     [i for i in range(bert_transformer_input.size(1))]).long()
        # pos_input = torch.stack([pos_input_one_batch for _ in range(
        #     bert_transformer_input.size(0))]).to(device)
        # position_embedding = self.position_embedding(pos_input)

        #  print("position_embedding:", position_embedding.size())
        # print("position embedding:", position_embedding)

        # bert_transformer_input = bert_transformer_input + position_embedding

        # print("bert_transformer_input:", bert_transformer_input.size())
        bert_transformer_input = self.CBHG_prenet(bert_transformer_input)
        # print("bert_transformer_input:", bert_transformer_input.size())

        # bert_transformer_input = bert_transformer_input + position_embedding

        encoder_output_word = self.CBHG(bert_transformer_input)
        # encoder_output_word = bert_transformer_input

        # encoder_output_alpha = self.EmbeddingNet(x)
        # pos_input_alpha = torch.stack([torch.Tensor([i for i in range(x.size(1))]).to(
        #     device) for _ in range(x.size(0))]).long()
        # position_embedding_alpha = self.position_embedding(pos_input_alpha)
        # encoder_output_alpha = x + position_embedding_alpha

        encoder_output_alpha, _ = self.EmbeddingNet(x)
        encoder_output_alpha = self.get_GRU_output(encoder_output_alpha)
        # encoder_output_alpha = x

        # print(encoder_output_word.size())
        # print(encoder_output_alpha.size())

        return encoder_output_word, encoder_output_alpha


class TacotronDecoder(nn.Module):
    """
    Tacotron Decoder
    """

    def __init__(self, hidden_size=128):
        super(TacotronDecoder, self).__init__()
        self.prenet = Prenet(hp.num_mels, hidden_size * 2, hidden_size)
        self.attn_decoder = AttentionDecoder(hidden_size * 2)
        self.postnet = PostNet()

    def forward(self, decoder_input, memory):
        # Initialize hidden state of GRUcells
        attn_hidden, gru1_hidden, gru2_hidden = self.attn_decoder.inithidden(
            decoder_input.size()[0])
        outputs = list()

        # Training phase
        if self.training:
            # Prenet
            dec_input = self.prenet.forward(decoder_input)
            timesteps = dec_input.size()[2] // hp.outputs_per_step

            # [GO] Frame
            # prev_output = dec_input[:, :, 0]
            # print(prev_output.size())
            GO_frame = torch.zeros(dec_input.size(
                0), dec_input.size(1)).to(device)
            # print(GO_frame.size())

            prev_output, attn_hidden, gru1_hidden, gru2_hidden = self.attn_decoder.forward(
                GO_frame, memory, attn_hidden=attn_hidden, gru1_hidden=gru1_hidden, gru2_hidden=gru2_hidden)
            outputs.append(prev_output)

            for i in range(timesteps):
                # Teacher Forced
                if random.random() < hp.teacher_forcing_ratio:
                    # Get spectrum at rth position
                    prev_output = dec_input[:, :, (i+1)*hp.outputs_per_step-1]
                else:
                    # Get last output
                    prev_output = prev_output[:, :, -1]
                    # prev_output = output[-1]

                prev_output, attn_hidden, gru1_hidden, gru2_hidden = self.attn_decoder.forward(
                    prev_output, memory, attn_hidden=attn_hidden, gru1_hidden=gru1_hidden, gru2_hidden=gru2_hidden)

                outputs.append(prev_output)

                # if random.random() < hp.teacher_forcing_ratio:
                #     # Get spectrum at rth position
                #     prev_output = dec_input[:, :, i * hp.outputs_per_step]
                # else:
                #     # Get last output
                #     prev_output = prev_output[:, :, -1]

            # Concatenate all mel spectrogram

            outputs = torch.cat(outputs, 2)

            # print(outputs.size())
            # print(decoder_input.size())

            outputs = outputs[:, :, 0:decoder_input.size(2)]

            # outputs_postnet = self.postnet(outputs) + outputs

        # Test
        else:
            # [GO] Frame
            prev_output = decoder_input

            for i in range(hp.max_iters):
                prev_output = self.prenet.forward(prev_output)
                prev_output = prev_output[:, :, 0]
                prev_output, attn_hidden, gru1_hidden, gru2_hidden = self.attn_decoder.forward(
                    prev_output, memory, attn_hidden=attn_hidden, gru1_hidden=gru1_hidden, gru2_hidden=gru2_hidden)
                outputs.append(prev_output)
                prev_output = prev_output[:, :, -1].unsqueeze(2)

            outputs = torch.cat(outputs, 2)

            # outputs_postnet = self.postnet(outputs) + outputs

        # return outputs, outputs_postnet
        return outputs


# class PostProcessingNet(nn.Module):
#     """
#     Post-processing Network
#     """

#     def __init__(self, hidden_size=125, num_freq=1025):
#         super(PostProcessingNet, self).__init__()
#         self.postcbhg = CBHG(
#             hidden_size, K=8, projection_size=hp.num_mels, is_post=True)
#         self.linear = SeqLinear(hidden_size * 2, num_freq)

#     def forward(self, decoder_output):
#         out = self.postcbhg.forward(decoder_output)
#         out = self.linear.forward(torch.transpose(out, 1, 2))

#         return out


class WETaSS(nn.Module):
    """
    WESS combine with Tacotron
    """

    def __init__(self):
        super(WETaSS, self).__init__()
        self.encoder = WESS_Encoder()
        self.decoder_word = TacotronDecoder()
        self.decoder_alpha = TacotronDecoder()
        # self.postnet = PostProcessingNet()
        self.postnet = PostNet()

    def forward(self, x, bert_embeddings, gate_for_words, mel_target=None, indexs_list=None):
        if self.training:
            encoder_output_word, encoder_output_alpha = self.encoder(
                x, bert_embeddings, gate_for_words, indexs_list)
            decoder_output_word = self.decoder_word(
                mel_target, encoder_output_word)
            decoder_output_alpha = self.decoder_alpha(
                mel_target, encoder_output_alpha)

            mel_output = decoder_output_word + decoder_output_alpha
            mel_output = mel_output.contiguous().transpose(1, 2)

            mel_output_postnet = self.postnet(mel_output) + mel_output

            mel_target = mel_target.contiguous().transpose(1, 2)

            return mel_output, mel_output_postnet, mel_target
        else:
            decoder_input = torch.zeros(1, hp.num_mels, 1).to(device)

            encoder_output_word, encoder_output_alpha = self.encoder(
                x, bert_embeddings, gate_for_words, indexs_list)
            decoder_output_word = self.decoder_word(
                decoder_input, encoder_output_word)
            decoder_output_alpha = self.decoder_alpha(
                decoder_input, encoder_output_alpha)

            mel_output = decoder_output_word + decoder_output_alpha
            mel_output = mel_output.contiguous().transpose(1, 2)

            mel_output_postnet = self.postnet(mel_output) + mel_output

            return mel_output, mel_output_postnet


# if __name__ == "__main__":
#     # Test
#     test_decoder = TacotronDecoder().to(device)
#     output = test_decoder(torch.randn(2, 80, 121).to(
#         device), torch.randn(2, 56, 256).to(device))
#     print(output.size())

#     # test_postnet = PostProcessingNet().to(device)
#     # out = test_postnet(output)
#     # print(out.size())

#     test_WETaSS = WETaSS().to(device)
#     print(test_WETaSS)
