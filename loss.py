import torch
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class WESSLoss(nn.Module):
    """WESS Loss"""

    def __init__(self):
        super(WESSLoss, self).__init__()

    def forward(self, mel_output, mel_output_postnet,  mel_target):
        # gate_len = gate_predicted.size(1)
        # mel_len = mel_output.size(1)

        # print("gate pre:", gate_target.size())

        # if gate_len < gate_target.size(1):
        #     raise ValueError("gate predicted is smaller than gate target!")

        # if mel_len < mel_target.size(1):
        #     raise ValueError("mel output is smaller than mel target!")

        # pad = torch.ones(gate_target.size(0), gate_predicted.size(
        #     1)-gate_target.size(1)).to(device)
        # gate_target = torch.cat((gate_target, pad), 1)

        # print("gate processed:", gate_target.size())

        mel_target.requires_grad = False
        # gate_target.requires_grad = False

        # mel_loss = nn.MSELoss()(mel_output, mel_target)
        # mel_postnet_loss = nn.MSELoss()(mel_output_postnet, mel_target)
        # gate_loss = nn.BCEWithLogitsLoss()(gate_predicted, gate_target)
        # gate_loss = nn.BCEWithLogitsLoss()(gate_predicted, gate_target)

        mel_loss = torch.abs(mel_output - mel_target)
        mel_loss = torch.mean(mel_loss)

        mel_postnet_loss = torch.abs(mel_output_postnet - mel_target)
        mel_postnet_loss = torch.mean(mel_postnet_loss)

        return mel_loss, mel_postnet_loss
