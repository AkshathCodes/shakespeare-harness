"""One-block causal byte transformer and its matmul cost model."""

from torch import nn
from torch.nn import functional as F

from experiment import estimated_step_flops as estimated_step_flops


class ByteTransformer(nn.Module):
    def __init__(self, width, context, vocab_size=256):
        super().__init__()
        self.token = nn.Embedding(vocab_size, width)
        self.position = nn.Embedding(context, width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.attention_out = nn.Linear(width, width, bias=False)
        self.expand = nn.Linear(width, 4 * width, bias=False)
        self.contract = nn.Linear(4 * width, width, bias=False)
        self.logits = nn.Linear(width, vocab_size, bias=False)
        nn.init.uniform_(self.token.weight, -width ** -0.5, width ** -0.5)
        nn.init.uniform_(self.position.weight, -width ** -0.5, width ** -0.5)

    def forward(self, tokens):
        x = self.token(tokens) + self.position.weight[:tokens.size(1)]
        query, key, value = self.qkv(x).unsqueeze(1).chunk(3, dim=-1)
        attended = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        x = x + self.attention_out(attended.squeeze(1))
        x = x + self.contract(self.expand(x).tanh())
        return self.logits(x)
