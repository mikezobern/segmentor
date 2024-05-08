import torch
torch.manual_seed(1337)

#####
##### Example data parameters
batch_size = 5
num_messages = 3 # number of messages
raw_embedding_size = 3 # number of chosen test embedding dimentions
num_graphs = 3 # formal reply graph, author graph, time graph

# документ с ручным вычислением:
# https://docs.google.com/spreadsheets/d/1emwkiEDFS0Gu8ev3WqWx8lsyQa6fVrNssLAQU_zRdsg/edit#gid=0
# batch_size = 5 num_messages = 3 raw_embedding_size = 3 num_graphs = 3 semantix_matrixes 3 mes_emb_size = 4

#####
# batch of messages
batch_of_raw_message_embeddings = torch.randn((batch_size, num_messages, raw_embedding_size))
print('batch_of_raw_message_embeddings',batch_of_raw_message_embeddings)
# batches of hypergraphs
batch_of_raw_graphs = torch.randn((batch_size, num_graphs, num_messages, num_messages))

print(batch_of_raw_graphs.shape)
print(batch_of_raw_message_embeddings.shape)

from from_colab_gpt import Zero_block
zb = Zero_block(3, raw_embedding_size, 4)
print('weights', list(zb.parameters()))
btsc = zb(batch_of_raw_message_embeddings) # B T S C
print(btsc)