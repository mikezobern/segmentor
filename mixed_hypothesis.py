import sqlite3; import random; #random.seed(1337)
from get_embedding import get_embedding
# Функция отдаём эмбеддинг случайного дерева.
# Этап 1: получить текст случайного N-дерева
# Получить индексы всех деревьев длиной N или больше
class Train_mixed_and_ordered():
    """ Получаем упорядоченные и неупорядоченные отрывки чата """
    def __init__(self, chat_messages):
        self.chat_messages = chat_messages
        self.conn = sqlite3.connect('blue_stool.db')
        self.cursor = self.conn.cursor()
        self.l_w = 'lixiangautorussia_raw'
        self.l_45_t = 'lixiangautorussia_45_trees'
        self.cursor.execute(f"PRAGMA table_info({self.l_45_t});")
        columns = self.cursor.fetchall()
        print([i[1] for i in columns])
        self.cursor.execute(f"PRAGMA table_info({self.l_w});")
        columns = self.cursor.fetchall()
        print([i[1] for i in columns])
        # Получаем ID всех деревьев данной длины:
        r = self.cursor.execute(f'SELECT dead_tree_id FROM {self.l_45_t} GROUP BY dead_tree_id HAVING COUNT(dead_tree_id) >= {chat_messages}')
        r = r.fetchall()
        self.ids_trees = [i[0] for i in r]
    def get_text_for_ordered(self):
        tree_id = random.choice(self.ids_trees)
        r = self.cursor.execute(f'SELECT message_id FROM {self.l_45_t} WHERE dead_tree_id = {tree_id}')
        r = r.fetchall(); r = [i[0] for i in r]; r = r[:self.chat_messages]
        r_str = ','.join(map(str, r))
        txts = self.cursor.execute(f'SELECT text, caption, username FROM {self.l_w} where message_id IN ({r_str})')
        txts = txts.fetchall()
        txt = ''
        for i in txts:
            txt+= str(i[0])*(i[0]!=None) + str(i[1])*(i[1]!=None) +'\n\n'
        return txt
    def get_text_mixed(self):
        r =  random.sample(self.ids_trees,self.chat_messages)
        r_str = ','.join(map(str, r))
        txts = self.cursor.execute(f'SELECT text, caption, username FROM {self.l_w} where message_id IN ({r_str})')
        txts = txts.fetchall()
        txt = ''
        for i in txts:
            txt += str(i[0]) * (i[0] != None) + str(i[1]) * (i[1] != None) + '\n\n'
        return txt
    def get_ordered_emb(self):

        return get_embedding(self.get_text_for_ordered())
    def get_mixed_emb(self):

        return get_embedding(self.get_text_mixed())

chat_size = 5
tmo = Train_mixed_and_ordered(chat_size)
# print(tmo.get_ordered_emb())

# теперь создаём модель
import torch
import torch.nn as nn
from torch.nn import functional as F
# torch.manual_seed(1338)

class Simple_classifier(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.emb_size = emb_size
        self.lin = nn.Linear(self.emb_size,1)
        self.relu = nn.LeakyReLU()


    def forward(self, text_emb, target = None):
        if target:
            target = torch.tensor(data = target, dtype=torch.float)
        text_emb = torch.tensor(data = text_emb)
        res = self.lin(text_emb)
        res = self.relu(res)
        if target != None:
            loss = F.mse_loss(res, target)
            return res, loss
        return res

model = Simple_classifier(1536)
print(model(tmo.get_ordered_emb()))


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train']: # 'val'
        losses = torch.zeros(eval_iters)
        accura = 0
        for k in range(eval_iters):

            brm, brg, targ = get_batch_f(split)

            pred, loss = model(brm, brg, targ)
            losses[k] = loss.item()

            if torch.argmax(targ)==torch.argmax(pred):
                accura+=1
        accura = accura/eval_iters
        out[split] = losses.mean()
    model.train()
    return out, accura
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

max_iters = 10000
eval_interval = 1
for iter in range(max_iters):
    # print('iteration',iter)
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses, accura = estimate_loss()
        # print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        print(f"step {iter}: train loss {losses['train']:.4f}, accuracy: {accura}")

    raw_embs, graph_adjs, target = get_batch_f('train')
    predictions, loss = model(raw_embs, graph_adjs, target)
    # print('loss', loss.item(), 'pred', predictions, 'target', target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()