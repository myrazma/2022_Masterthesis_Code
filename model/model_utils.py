# containing utils like the regression head to share among different models
# e.g. here are methods / classes that should stay the same or can be used among different models

import torch.nn as nn

class RegressionHead(nn.Module):

    def __init__(self, dropout=0.2, D_in=768, D_hidden1=100, D_hidden2=10, D_out=1):
        self.bert_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(D_in, D_hidden1))

        self.regressor = nn.Sequential(
            nn.Linear(D_hidden1, D_hidden2),
	        nn.ReLU(),
            nn.Linear(D_hidden2, D_out))

    def forward(self, bert_outputs):
        bert_output = bert_outputs[1]
        bert_head_output = self.bert_head(bert_output)
        outputs = self.regressor(bert_head_output)
        return outputs
