import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1338)

# hyperparameters
chat_size = 50 # number of messages
max_iters = 7501
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 1
n_embd = 100 # ME, внутренний эмбеддинг сообщения;
n_head = 10 # число голов
dropout = 0.0
# ------------
number_of_semantic_matrixes = 2 # сколько делаем матриц "искусственного винмания" на основе эмбеддингов
semantic_embedding_size = 100
num_graphs = 1 # formal reply graph, author graph
# ------------
raw_embedding_size = 500  # number of chosen test embedding dimentions
batch_size = 2
# ------------
# ------------

# data loading
def get_batch(split):
    torch.manual_seed(1337)
    # generate a small batch of data of inputs x and targets y
    batch_of_raw_message_embeddings = torch.randn((batch_size, chat_size, raw_embedding_size))
    # batch_of_raw_message_embeddings = F.softmax(batch_of_raw_message_embeddings,-1)
    batch_of_raw_graphs = torch.randn((batch_size, num_graphs, chat_size, chat_size))
    # batch_of_raw_graphs = F.softmax(batch_of_raw_graphs,-1)
    target = torch.randn(batch_size,chat_size)
    # предстоит написать!
    target = F.softmax(target,-1)
    return batch_of_raw_message_embeddings, batch_of_raw_graphs, target

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train']: # 'val'
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            brm, brg, targ = get_batch(split)
            pred, loss = model(brm, brg, targ)
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
        # Единая матрица преобразования для всех number_of_semantic_matrixes семантических проекций пайторч-стиле:
        self.linear = torch.nn.Linear(raw_embedding_size, number_of_semantic_matrixes * semantic_embedding_size)
        self.linear_2 = torch.nn.Linear(raw_embedding_size, message_embedding_size)
    def forward(self, batch_of_text_embeddings, batch_of_links = None):
        """ Размер пачки эмбеддингов (B T Cr) """
        B, T, C = batch_of_text_embeddings.shape

        # Получив батчи с сырыми текст-эмбеддингами, преобразуем их в эмбеддинги размера semantic_embedding_size:
        ln = self.linear(batch_of_text_embeddings)
        btsc = ln.view(B, T, self.number_of_semantic_matrixes, semantic_embedding_size)
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
        s = torch.sum((srep - intl_rep)**2 + 1e-3, -1)**0.5 # B T**T S C/C # 1e-3->1.6743; 1e-4->2.6743
        s_reshaped = - s.transpose(-2, -1).view(B, S, T, T)  # B S T T
        s_reshaped = F.softmax(s_reshaped,-1)

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
            # print('free_heads_number',free_heads_number)
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
        # print('x in Head shape', x.shape)
        B, T, C = x.shape
        if adjacent!=None:
            B,S,T,T = adjacent.shape

        k = self.key(x)  # (B,T,C)
        q = self.query(x)  # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C ** -0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)
        # wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)


        if adjacent!=None:
            # print('wei shape in the head', wei.shape)
            adjacent = adjacent.view(B, T, T)
            # print('adjacent shape in the head', adjacent.shape)
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

            # print('multihead adjacent shape', adjacent.shape)
            # print('multihead adjacent shape[:, 1, :, :]', adjacent[:, 0:1, :, :].shape)
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

class FinalFeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, chat_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(chat_size**2, chat_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(chat_size, chat_size),
            nn.ReLU(),
            nn.Dropout(dropout)
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
        # print('x shape at the start of the block 1', x.shape)
        if adjacent==None:
            # print('adjacent None',adjacent)
            x = x + self.sa(self.ln1(x))
            x = x + self.ffwd(self.ln2(x))
        else:
            x = x + self.sa(self.ln1(x), adjacent)
            # print('x chape at the block after sa', x.shape)
            x = x + self.ffwd(self.ln2(x))

        return x

class Final_block(nn.Module):
    """ Принимает эмбеддинги сообщений, возвращает вектор длиной chat_size """
    def __init__(self, n_embd):
        super().__init__()
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.fff = FinalFeedFoward(chat_size)
        self.layer_norm_1 = nn.LayerNorm((chat_size,chat_size)) # дописать нормализацию
        self.layer_norm_2 = nn.LayerNorm(chat_size)
    def forward(self, x):
        # print('Final layer!')
        B, T, C = x.shape
        k = self.key(x)  # (B,T,C)
        q = self.query(x)  # (B,T,C)
        # print(q)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C ** -0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.layer_norm_1(wei)
        wei = self.dropout(wei)

        out = self.fff(wei.view(B, T*T))
        out = self.layer_norm_2(out)
        out = F.softmax(out, dim = -1)

        return out

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.position_embedding_table = nn.Embedding(chat_size, n_embd)
        self.zero_block = Zero_block(number_of_semantic_matrixes, raw_embedding_size, n_embd)
        self.block_1 = Block(n_embd,n_head)
        self.block_2 = Block(n_embd,n_head)
        self.final_block = Final_block(n_embd)

    def forward(self, raw_mes_embs, graph_adjacenties, targets=None):
        """ New forward function takes batch of embngs, batch of adjacenties
        and true attention vector of last message as a target"""
        B, G, T, T = graph_adjacenties.shape
        B, T, C = raw_mes_embs.shape
        # матрицы смежности и семантические матрицы из нулевого слоя:
        adjs, mes_embs = self.zero_block(raw_mes_embs, graph_adjacenties)
        # Подмешивание позиции:
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = mes_embs + pos_emb  # (B,T,C)

        # засовываем в первый слой получившуюся пачку векторов:

        x = self.block_1(x, adjs)  # (B,T,C)
        # print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n\n\n', x, '\n\n\n\n\n')
        # во второй слой входим без матриц смежности:
        # x = self.block_2(x)
        # финальный блок:
        x = self.final_block(x)

        if targets is None:
             loss = None
             print('===========================================')
        else:
            loss = F.mse_loss(x, targets) #+ F.cross_entropy(x, targets)/10
            # loss = F.cross_entropy(x, targets)
        return x, loss


model = BigramLanguageModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
'''
r = m(batch_of_raw_message_embeddings,batch_of_raw_graphs)
print('output of the block \n', r)
print(r[0].shape)
'''

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


for iter in range(max_iters):
    # print('iteration',iter)
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        # print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        print(f"step {iter}: train loss {losses['train']:.4f}")

    raw_embs, graph_adjs, target = get_batch('train')
    predictions, loss = model(raw_embs, graph_adjs, target)
    # print('loss', loss.item(), 'pred', predictions, 'target', target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
