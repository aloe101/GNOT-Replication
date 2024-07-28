import torch.nn as nn
import torch
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.GELU())
        for i in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

# test:
# num_layers = 3
# input_dim = 10
# hidden_dim = 20
# output_dim = 5

# model = MLP(num_layers, input_dim, hidden_dim, output_dim)
# print(model)

# x = torch.randn(1, input_dim)  
# output = model(x)
# print(output.shape)

class LinearAttention(nn.Module):
    def __init__(self, n_embed, head, n_input_functions=0):
        super().__init__()
        self.head = head
        self.n_embed = n_embed
        self.n_input_functions = n_input_functions
        self.head_dim = n_embed // head

        assert self.head_dim * head == n_embed, "n_embed should be divisible by head"

        self.query = nn.Linear(n_embed, n_embed)
        self.fc_out = nn.Linear(n_embed, n_embed)

        if self.n_input_functions > 0:
            self.key = nn.ModuleList([nn.Linear(n_embed, n_embed) for _ in range(n_input_functions)])
            self.value = nn.ModuleList([nn.Linear(n_embed, n_embed) for _ in range(n_input_functions)])
        else:
            self.key = nn.Linear(n_embed, n_embed)
            self.value = nn.Linear(n_embed, n_embed)
            
    def forward(self, query, input_functions=None):
        batch_size = query.shape[0]
        query_len = query.shape[1]
        q = self.query(query)
        q = q.reshape(batch_size, query_len, self.head, self.head_dim)
        q = q.permute(0, 2, 1, 3)
        q = F.softmax(q, dim=-1)
        
        if input_functions:
            outputs = []
            for i in range(self.n_input_functions):
                key_len = input_functions[i].shape[1]
                value_len = input_functions[i].shape[1]
                
                key = self.key[i](input_functions[i])
                value = self.value[i](input_functions[i])
                # Split the embedding into self.heads different pieces
                key = key.reshape(batch_size, key_len, self.head, self.head_dim)
                key = key.permute(0, 2, 1, 3)
                key = F.softmax(key, dim=-1)

                value = value.reshape(batch_size, value_len, self.head, self.head_dim)
                value = value.permute(0, 2, 1, 3)

                k_sum = key.sum(dim=2, keepdim=True)
                alpha = 1.0 / (q * k_sum).sum(dim=-1, keepdim=True)
                tmp = torch.matmul(key.transpose(-2, -1), value)
                output = alpha * torch.matmul(q, tmp)
                outputs.append(output.reshape(batch_size, query_len, self.n_embed))

            q = q.reshape(batch_size, query_len, self.n_embed)
            stacked_outputs = torch.stack(outputs, dim=0)
            average_output = torch.mean(stacked_outputs, dim=0)
            res = q + average_output

        else:
            key = self.key(query)
            value = self.value(query)
            key = key.reshape(batch_size, query_len, self.head, self.head_dim)
            key = key.permute(0, 2, 1, 3)
            key = F.softmax(key, dim=-1)

            value = value.reshape(batch_size, query_len, self.head, self.head_dim)
            value = value.permute(0, 2, 1, 3)

            k_sum = key.sum(dim=2, keepdim=True)
            alpha = 1.0 / (q * k_sum).sum(dim=-1, keepdim=True)
            tmp = torch.matmul(key.transpose(-2, -1), value)
            output = alpha * torch.matmul(q, tmp)

            q = q.reshape(batch_size, query_len, self.n_embed)
            res = q + output.reshape(batch_size, query_len, self.n_embed)
        
        res = self.fc_out(res)
        return res

# test
# query = torch.randn(4, 10044, 256)
# # input_functions = [torch.randn(4, 805, 256)]

# model = LinearAttention(256, 8)
# output = model(query) 

# print(output.shape)

class HeterogeneousNormalizedAttentionBlock(nn.Module):
    def __init__(self, n_attn_hidden_dim, n_mlp_num_layers, n_mlp_hidden_dim, n_input_hidden_dim, n_expert, n_head, n_input_functions=0):
        super().__init__()
        self.cross_attention = LinearAttention(n_attn_hidden_dim, n_head, n_input_functions)
        self.self_attention = LinearAttention(n_attn_hidden_dim, n_head)
        self.ffn1 = nn.ModuleList(MLP(n_mlp_num_layers, n_input_hidden_dim, n_mlp_hidden_dim, n_mlp_hidden_dim) for _ in range(n_expert))
        self.ffn2 = nn.ModuleList(MLP(n_mlp_num_layers, n_input_hidden_dim, n_mlp_hidden_dim, n_mlp_hidden_dim) for _ in range(n_expert))
    
    def forward(self, scores, query, input_functions=None):
        cross_attention_output = self.cross_attention(query, input_functions)
        cross_attention_output = [self.ffn1[i](cross_attention_output) for i in range(len(self.ffn1))]
        cross_attention_output = torch.stack(cross_attention_output, dim=-1)
        cross_attention_output = (scores * cross_attention_output).sum(dim=-1,keepdim=False)
        query = query + cross_attention_output

        self_attention_output = self.self_attention(query)
        self_attention_output = [self.ffn2[i](self_attention_output) for i in range(len(self.ffn2))]
        self_attention_output = torch.stack(self_attention_output, dim=-1)
        self_attention_output = (scores * self_attention_output).sum(dim=-1,keepdim=False)
        query = query + self_attention_output

        return query


class GNOT(nn.Module):
    def __init__(self, input_dim, theta_dim, input_func_dim, out_dim, n_attn_layers, n_attn_hidden_dim, n_mlp_num_layers, n_mlp_hidden_dim, n_input_hidden_dim, n_expert, n_head, n_input_functions=0):
        super().__init__()
        # input_dim 2
        self.x = MLP(n_mlp_num_layers, input_dim + theta_dim, n_input_hidden_dim, n_input_hidden_dim)

        self.gating = MLP(n_mlp_num_layers, input_dim, n_mlp_hidden_dim, n_expert)
        self.input_func_mlps = nn.ModuleList(MLP(n_mlp_num_layers, input_func_dim, n_mlp_hidden_dim, n_input_hidden_dim) for _ in range(n_input_functions))
        self.blocks = nn.ModuleList(HeterogeneousNormalizedAttentionBlock(n_attn_hidden_dim, n_mlp_num_layers, n_mlp_hidden_dim, n_input_hidden_dim, n_expert, n_head, n_input_functions) for _ in range(n_attn_layers))

        self.out = MLP(n_mlp_num_layers, n_input_hidden_dim, n_mlp_hidden_dim, out_dim)

    def forward(self, x, theta, input_functions=None):
        scores = self.gating(x)
        scores = F.softmax(scores, dim=-1).unsqueeze(2)

        expanded_theta = theta.unsqueeze(1).expand(-1, x.shape[1], -1)
        x = torch.cat([x, expanded_theta], dim=-1)

        query = self.x(x) # torch.Size([4, 10268, 256])
        temp_input_functions = []

        if input_functions is not None:
            for i in range(len(self.input_func_mlps)):
                temp_input_functions.append(self.input_func_mlps[i](input_functions[i]))
        
        for block in self.blocks:
            query = block(scores, query, temp_input_functions)

        query = self.out(query)

        return query