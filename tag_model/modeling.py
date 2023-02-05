import torch
import torch.nn as nn
import torch.nn.functional as F


class TagConfig(object):
    def __init__(self,
                 tag_vocab_size,
                 hidden_size=5,
                 layer_num=1,
                 output_dim=5,
                 dropout_prob=0.1,
                 ):
        self.tag_vocab_size = tag_vocab_size
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.dropout_prob = dropout_prob
        self.output_dim = output_dim

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class TagEmbeddings(nn.Module):
    """Simple tag embeddings, randomly initialized."""
    def __init__(self, config):
        super(TagEmbeddings, self).__init__()
        self.tag_embeddings = nn.Embedding(config.tag_vocab_size, config.hidden_size, padding_idx=0)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(self, input_tag_ids):
        tags_embeddings = self.tag_embeddings(input_tag_ids)
        embeddings = tags_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class TagEmebedding(nn.Module):

    def __init__(self, config):
        super(TagEmebedding, self).__init__()
        # Embedding
        self.hidden_size = config.hidden_size
        self.embed = TagEmbeddings(config)
        # Linear
        self.fc = nn.Linear(config.hidden_size, config.output_dim)
        #  dropout
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(self, flat_input_ids):  # flat_input_ids.size() = (batch_size*num_aspect, seq_len)
        # flat_input_ids = input_tag_ids.view(-1, input_tag_ids.size(-1))
        embed = self.embed(flat_input_ids)
        embed = self.dropout(embed)
        input = embed.view(-1,flat_input_ids.size(1), self.hidden_size)
        # linear
        logit = self.fc(input)
        return logit


class TagPooler(nn.Module):
    def __init__(self, config):
        super(TagPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
