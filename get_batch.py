import sqlite3
import torch
import json
import random

class Train_data():
    """ Возвращает эмбеддинги (список списков) и класс родителя """
    def __init__(self, chat_messages):
        self.conn = sqlite3.connect('blue_stool.db')
        self.cursor = self.conn.cursor()
        self.l_w = 'lixiangautorussia_raw'
        self.l_45_t = 'lixiangautorussia_45_trees'
        self.cursor.execute(f"PRAGMA table_info({self.l_45_t});")
        self.chat_messages = chat_messages
        columns = self.cursor.fetchall()
        print([i[1] for i in columns])

        # Теперь для каждого класса создаём собственный список айдишников сообщений: всего 6 наборов айдишников
        ids_by_parent_dict = {}
        l_45_t = 'lixiangautorussia_45_trees'
        ids_0 = self.cursor.execute(f'SELECT row_id FROM {l_45_t} WHERE distance_to_parent>={self.chat_messages} ')
        candidates_0 = ids_0.fetchall()
        ids_by_parent_dict[0] = candidates_0
        for i in range(1, chat_messages):
            candidates = self.cursor.execute(f'SELECT row_id FROM {l_45_t} WHERE distance_to_parent={i} ')
            candidates = candidates.fetchall()
            ids_by_parent_dict[i] = candidates
        self.ids_by_parent_dict = ids_by_parent_dict



    def get_bunch_of_embs_by_row_id(self):
        """ Пачка эмбеддингов одного фргмента чата """
        class_par = random.randint(0,self.chat_messages-1)
        class_size = len(self.ids_by_parent_dict[class_par])


        row_mes_num = self.ids_by_parent_dict[class_par][random.randint(0,class_size-1)][0]
        # row_mes_num = 11000
        parent_num = row_mes_num - self.chat_messages
        res = self.cursor.execute(f'SELECT distance_to_parent, jsoned_embedding FROM {self.l_45_t} WHERE row_id<={row_mes_num} AND row_id>{parent_num}')
        res = res.fetchall()
        embedding_list = []
        # Список эмбеддингов (последний — от сообщения)
        for i in res:
            l = json.loads(i[1])
            embedding_list.append(l)


        # Превращаем список эмбеддингов в тензор:
        emb_tens = torch.tensor(data=embedding_list)

        parent_class = int( int(res[-1][0])*(self.chat_messages > int(res[-1][0])) )
        parent_one_hot = torch.nn.functional.one_hot(torch.arange(0, self.chat_messages) % self.chat_messages)[parent_class]

        return embedding_list, parent_one_hot.tolist()

    def get_batch(self, batch_size):
        examples_list = []
        proto_tensor_embs = []
        proto_tensor_targ = []
        for i in range(batch_size):
            e_l, p_o_h = self.get_bunch_of_embs_by_row_id()
            proto_tensor_embs.append(e_l)
            proto_tensor_targ.append(p_o_h)

        t_e = torch.tensor(data=proto_tensor_embs)
        t_t = torch.tensor(data=proto_tensor_targ, dtype=torch.float)

        B, T, C = t_e.shape
        # print(B, T, C)
        if (B,T,C)!=(batch_size,6,1536):
            print('====================================')
            t_e, t_t = self.get_batch(batch_size)


        return t_e, t_t
    def class_sizes_dict_f(self):
        chat_messages = self.chat_messages
        class_sizes_dict = {}

        candidates_0 = self.cursor.execute(f'SELECT count(row_id) FROM {self.l_45_t} WHERE distance_to_parent>={chat_messages} ')
        candidates_0 = candidates_0.fetchall()
        print(0, '\t', candidates_0[0][0])
        class_sizes_dict[0] = candidates_0[0][0]

        for i in range(1, chat_messages):
            candidates = self.cursor.execute(f'SELECT count(row_id) FROM {self.l_45_t} WHERE distance_to_parent={i} ')
            candidates = candidates.fetchall()
            print(i, '\t', candidates[0][0])
            class_sizes_dict[i] = candidates[0][0]

        print(class_sizes_dict)
        print('Среднее значение ', sum([class_sizes_dict[i] for i in class_sizes_dict if i > 0]) / (chat_messages - 1))
        self.class_sizes_dict = class_sizes_dict

# td = Train_data(6)


# t = td.get_batch(3)
# print(t)