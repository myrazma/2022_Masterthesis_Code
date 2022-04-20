# containing utils like the regression head to share among different models
# e.g. here are methods / classes that should stay the same or can be used among different models

import torch.nn as nn

class RegressionHead(nn.Module):
    """Regression head for Bert model

    Args:
        nn (nn.Module): Inherit from nn.Module
    """

    def __init__(self, dropout=0.2, D_in=768, D_hidden1=100, D_hidden2=10, D_out=1):
        super(RegressionHead, self).__init__()
        self.bert_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(D_in, D_hidden1),
	        nn.ReLU())

        self.regressor = nn.Sequential(
            nn.Linear(D_hidden1, D_hidden2),
	        nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(D_hidden2, D_out))

    def forward(self, bert_outputs):
        bert_output = bert_outputs[1]
        bert_head_output = self.bert_head(bert_output)
        outputs = self.regressor(bert_head_output)
        return outputs


def count_updated_parameters(model_params):
    """Count the parameters of the model that are updated (requires_grad = True)

    Args:
        model_params (_type_): The model parameters

    Returns:
        int: The number of parameters updated
    """
    model_size = 0
    for p in model_params:
        if p.requires_grad:  # only count if the parameter is updated during training
            model_size += p.flatten().size()[0]
    return model_size