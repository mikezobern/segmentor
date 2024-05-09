import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

# hyperparameters
chat_size = 3 # number of messages
max_iters = 5
eval_interval = 1
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 4 # ME, внутренний эмбеддинг сообщения;
n_head = 4 # число голов
n_layer = 1
dropout = 0.0
# ------------
number_of_semantic_matrixes = 1
num_graphs = 1 # formal reply graph, author graph
heads_for_each_semantic = 1 # сколько голов отдаётся каждой семантической матрице
free_heads = 3 # Число голов без входящих графов

raw_embedding_size = 7  # number of chosen test embedding dimentions
output_size = chat_size
batch_size = 2

# ------------
# ------------
batch_of_raw_message_embeddings = torch.randn((batch_size, chat_size, raw_embedding_size))
batch_of_raw_graphs = torch.randn((batch_size, num_graphs, chat_size, chat_size))
# ------------
# ------------

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    pass
    # предстоит написать!
    return None

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Zero_block(nn.Module):
    """ raw text embeddings to bunch of semantic matrixes """
    def __init__(self, number_of_semantic_matrixes, raw_embedding_size, message_embedding_size):
        super().__init__()
        self.number_of_semantic_matrixes = number_of_semantic_matrixes
        self.head_size = message_embedding_size
        # Единая матрица преобразования для всех семантических проекций пайторч-стиле:
        self.linear = torch.nn.Linear(raw_embedding_size, number_of_semantic_matrixes * message_embedding_size)
        self.linear_2 = torch.nn.Linear(raw_embedding_size, message_embedding_size)

    def forward(self, batch_of_text_embeddings, batch_of_links = None):
        """ Размер пачки эмбеддингов (B T Cr) """
        B, T, C = batch_of_text_embeddings.shape
        # Получив батчи с сырыми текст-эмбеддингами, преобразуем их в эмбеддинги размера голов:
        ln = self.linear(batch_of_text_embeddings)
        btsc = ln.view(B, T, self.number_of_semantic_matrixes, self.head_size)
        # Здесь, возможно, понадобится нормализация
        pass
        # Запоминаем форму тензора:
        B, T, S, C = btsc.shape
        # Превращаем батч с семантическими векторами в батч матриц в три простых шага
        # Шаг #1: копируем содержание нод T раз целиком:
        srep = btsc.repeat(1, T, 1, 1) # B T**T S C
        # Шаг #2: копируем содержание нод T раз, но повторяясь внутри T-измерения:
        intl_rep = btsc.repeat_interleave(T, dim = 1) # B T**T S C
        # Шаг #3: вычисляем евклидову меру
        s = torch.sum((srep - intl_rep)**2, -1)**0.5 # B T**T S C/C
        s_reshaped = - s.transpose(-2, -1).view(B, S, T, T)  # B S T T

        # Теперь получаем эмбеддинги нод из сырых эмбеддингов
        message_embeddings = self.linear_2(batch_of_text_embeddings)
        # Здесь, возможно, понадобится нормализация
        pass

        adj_matrices = None
        if batch_of_links!=None:
            # Объединяем пачки матриц смежности графов с пачками семантического внимания (B G T T)
            # В мульхедаттеншене число голов фиксировано и каждой голове нужно дать свою матрицу смежности
            # Поэтому размерность S должна быть равна числу голов в мультихеде
            # Логика такая: вводим граф авторов, граф формальных реплаев и добиваем единичными матрицами ("свободными")
            # Первый этап: посчитать, сколько у нас в сумме семантических и графовых матриц
            S = self.number_of_semantic_matrixes
            B,G,T,T = batch_of_links.shape
            free_heads_number = n_head - S - G
            # print(S,'S', G, 'G' , n_head, 'n head')
            print('free_heads_number',free_heads_number)
            assert free_heads_number >= 0
            ones_for_matrices = torch.ones(B,free_heads_number,T,T)

            adj_matrices = torch.cat([s_reshaped, batch_of_links, ones_for_matrices], dim=1)
            B,S,T,T = adj_matrices.shape; assert S == n_head
            # adj_matrices = F.softmax(adj_matrices, -1)

        return adj_matrices, message_embeddings

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # self.register_buffer('tril', torch.tril(torch.ones(chat_size, chat_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adjacent=None):
        print('x in Head shape', x.shape)
        B, T, C = x.shape
        if adjacent!=None:
            B,S,T,T = adjacent.shape

        k = self.key(x)  # (B,T,C)
        q = self.query(x)  # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C ** -0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)
        # wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)


        if adjacent!=None:
            print('wei shape in the head', wei.shape)
            adjacent = adjacent.view(B, T, T)
            print('adjacent shape in the head', adjacent.shape)
            wei = wei*adjacent # заменить на маскирование (графы авторов и формальных реплаев чисто нули и единицы)
        else:
            pass
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList([Head(head_size) for _ in range(self.num_heads)])

        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adjacent = None):

        if adjacent!=None:
            print('multihead adjacent shape', adjacent.shape)
            print('multihead adjacent shape[:, 1, :, :]', adjacent[:, 0:1, :, :].shape)
            out = torch.cat([self.heads[hi](x, adjacent[:, hi:hi+1, :, :]) for hi in range(self.num_heads)], dim=-1)
        else:
            out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
class Block(nn.Module):
    """ Transformer block: communication followed by computation
        Возвращает в любом случа пока только лишь эмбеддинги нод, adja не отдаёт"""

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, adjacent = None):
        print('x shape at the start of the block 1', x.shape)
        if adjacent==None:
            print('adjacent None',adjacent)
            x = x + self.sa(self.ln1(x))
            x = x + self.ffwd(self.ln2(x))
        else:
            x = x + self.sa(self.ln1(x), adjacent)
            print('x chape at the block after sa', x.shape)
            x = x + self.ffwd(self.ln2(x))

        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.position_embedding_table = nn.Embedding(chat_size, n_embd)
        self.zero_block = Zero_block(number_of_semantic_matrixes, raw_embedding_size, n_embd)
        # self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.block_1 = Block(n_embd,n_head)
        self.block_2 = Block(n_embd,n_head)
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, output_size)

    def forward(self, raw_mes_embs, graph_adjacenties, targets=None):
        """ New forward function takes batch of embngs, batch of adjacenties and true attention vector of last message as a target"""
        B, G, T, T = graph_adjacenties.shape
        B, T, C = raw_mes_embs.shape
        # матрицы смежности и семантические матрицы из нулевого слоя:
        adjs, mes_embs = self.zero_block(raw_mes_embs, graph_adjacenties)

        # Подмешивание позиции:
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = mes_embs + pos_emb  # (B,T,C)

        # засовываем в первый слой получившуюся пачку векторов:
        x = self.block_1(x, adjs)  # (B,T,C)

        # во второй слой входим без матриц смежности:
        x = self.block_2(x)


        # x = self.ln_f(x)  # (B,T,C)
        # logits = self.lm_head(x)  # (B,T,output_size)

        # if targets is None:
        #     loss = None
        # else:
        #     B, T, C = logits.shape
        #     logits = logits.view(B * T, C)
        #     targets = targets.view(B * T)
        #     loss = F.cross_entropy(logits, targets)
        #
        # return logits, loss
        return x


model = BigramLanguageModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

r = m(batch_of_raw_message_embeddings,batch_of_raw_graphs)
print('output of the block \n', r)
print(r.shape)
'''
model = BigramLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))
'''