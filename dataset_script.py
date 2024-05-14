""" Скрипт для получения датасета """
from get_embedding import get_embedding_batch, get_embedding
import os
import torch
import sqlite3
import json


# Подключаемся к патрикеевской патрикеевской базе данных
conn = sqlite3.connect('blue_stool.db')
cursor = conn.cursor()
l_w = 'lixiangautorussia_raw'
l_45_t = 'lixiangautorussia_45_trees'
cursor.execute(f"PRAGMA table_info({l_45_t});")
columns = cursor.fetchall()
print([i[1] for i in columns])

# Нумеруем все сообщения в "прошивке ПО"
'''
all_rows = cursor.execute(f'SELECT * FROM {l_45_t}')
all_rows = all_rows.fetchall()


c = 0
for row in all_rows:
    row_id = row[0]
    cursor.execute(f"UPDATE {l_45_t} SET row_id = {c} WHERE message_id = {row_id}")
    c+=1

conn.commit()
'''
# Добавляем столбец
'''
column_name = 'jsoned_embedding'
cursor.execute(f'ALTER TABLE {l_45_t} ADD COLUMN {column_name} TEXT')
conn.commit()
conn.close()
'''

# Определяем расстояние до родителя -- и записываем в базу. Если родителя нет, записываем 0:
'''
all_rows = cursor.execute(f'SELECT * FROM {l_45_t}')
all_rows = all_rows.fetchall()
c = 0
for i in all_rows:
    # print(i)
    parent_mes_id = i[2]
    if parent_mes_id!=None:
        parent_row_id = cursor.execute(f'SELECT row_id FROM {l_45_t} WHERE message_id = {parent_mes_id}')
        parent_row_id = parent_row_id.fetchone()
        if parent_row_id != None:
            parent_row_id = parent_row_id[0]
            # print('parent_row_id = ', parent_row_id)
            mes_row_id = i[7]
            # print('message_row_id', mes_row_id)
            d = mes_row_id - parent_row_id
            cursor.execute(f"UPDATE {l_45_t} SET distance_to_parent = {d} WHERE message_id = {i[0]}")
    c+=1

conn.commit()
conn.close()
# all_distances.sort()
# print(all_distances)

r = cursor.execute(f'SELECT * FROM {l_45_t}')
r = r.fetchall()
for i in r:
    print(i)'''

# Таблица деревье с расстояниями до родителей получена -- теперь нужно для каждого сообщения добавить текст-эмбеддинг
# Перебираем все сообщения:

'''
l_w = 'lixiangautorussia_raw'
l_45_t = 'lixiangautorussia_45_trees'
cursor.execute(f"PRAGMA table_info({l_w});")
columns = cursor.fetchall()
print([i[1] for i in columns])
r = cursor.execute(f'SELECT * FROM {l_45_t}')
r = r.fetchall()

temporary_dict = {} # Словарь mes_id : embedding
emb_batch_size = 10

# Пройтись по базе и собрать в список все айдишники, у которых нет эмбеддинга
res = cursor.execute(f'SELECT message_id FROM {l_45_t} WHERE jsoned_embedding IS NULL')
null_embs = []
res = res.fetchall()
for i in res:
    null_embs.append(i[0])
print('len(null_embs)', len(null_embs))
# теперь наполняем буферный словарь

emb_batch_size = 1000
buffer = []

cc = 0
msg_tot = 0
while null_embs and cc<100:
    position = min(emb_batch_size, len(null_embs))
    batch_to_add, null_embs = null_embs[:position], null_embs[position:]
    # print('batch_to_add', batch_to_add)

    # Теперь составляем словарь buffer_dict
    buffer_dict = {}
    for mes_num in batch_to_add:
        res = cursor.execute(f'SELECT * FROM {l_w} WHERE message_id = {mes_num}')
        res = res.fetchone()
        # print('res', res)
        # получаем строку для отправки в эмбеддинг:
        text_to_emb = ''
        author = str(res[10])*(res[10]!=None) + str(res[11])*(res[11]!=None) + str(res[12])*(res[12]!=None)
        if author=='':
            author = 'Аноним'
        text = str(res[4])*(res[4]!=None) + str(res[7])*(res[7]!=None)
        text_to_emb = 'Автор сообщения: '+ author + '\n'+ text
        buffer_dict[mes_num] = text_to_emb
    # Теперь отправляем пачку текстов openai
    text_batch = []
    mes_ids = []
    for mes_id in buffer_dict:
        text_batch.append(buffer_dict[mes_id])
        mes_ids.append(mes_id)
    print(f'Запрос {len(mes_ids)} эмбеддингов батча {cc} у openai')
    embs_batch = get_embedding_batch(text_batch)

    # теперь добавляем эти эмбеддинги в базу
    print(f'Добавление эмбеддингов батча {cc} в базу...')
    for i in range(len(mes_ids)):
        msg_tot+=1
        # print(f'Сообщение #{msg_tot}, id ={mes_ids[i]}')
        jsond = json.dumps(embs_batch[i])
        cursor.execute(f"UPDATE {l_45_t} SET jsoned_embedding = '{jsond}' WHERE message_id = {mes_ids[i]}")
        conn.commit()
    cc+=1


'''
# Теперь проверка: выводим случайное сообщение из базы
'''
r = cursor.execute(f'SELECT * FROM {l_45_t} WHERE row_id = 25000')
r = r.fetchone()
print(r[0],r[9][:50])


res = cursor.execute(f'SELECT * FROM {l_w} WHERE message_id = 203610')
res = res.fetchone()
author = str(res[10])*(res[10]!=None) + str(res[11])*(res[11]!=None) + str(res[12])*(res[12]!=None)
if author == '':
    author = 'Аноним'
text = str(res[4]) * (res[4] != None) + str(res[7]) * (res[7] != None)
text_to_emb = 'Автор сообщения: ' + author + '\n' + text
print(get_embedding(text_to_emb)[:50])

'''

# Пишем функцию получения эмбеддинга -- на выходе сериализованный json


# Сохраняем в csv-файл все сообщения, упорядоченные по времени

# Открываем файл и получаем оттуда готовый батч