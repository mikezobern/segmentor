""" Скрипт для получения датасета """
from get_embedding import get_embedding
import os
import torch
import sqlite3


# Подключаемся к патрикеевской патрикеевской базе данных
conn = sqlite3.connect('blue_stool.db')
cursor = conn.cursor()
l_w = 'lixiangautorussia_raw'
l_45_t = 'lixiangautorussia_45_trees'
cursor.execute(f"PRAGMA table_info({l_45_t});")
columns = cursor.fetchall()
print([i[1] for i in columns])

# Перебираем все сообщения в "прошивке ПО"

all_rows = cursor.execute(f'SELECT * FROM {l_45_t}')
all_rows = all_rows.fetchall()

'''
c = 0
for row in all_rows:
    row_id = row[0]
    cursor.execute(f"UPDATE {l_45_t} SET row_id = {c} WHERE message_id = {row_id}")
    c+=1
'''

all_rows = cursor.execute(f'SELECT * FROM {l_45_t}')
all_rows = all_rows.fetchall()
c = 0
for i in all_rows:
    print(i)
    c+=1
    if c>10:
        break

# Пишем функцию "расстояние до родителя"


# Запрос для добавления нового столбца с порядковыми номерами строк


# Запрос для обновления значений в новом столбце с порядковыми номерами строк



# Пишем функцию получения эмбеддинга -- на выходе сериализованный json


# Сохраняем в csv-файл все сообщения, упорядоченные по времени

# Открываем файл и получаем оттуда готовый батч