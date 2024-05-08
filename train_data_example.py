import torch

#####
##### Example data parameters
batch_size = 2
num_messages = 6 # number of messages
embedding_dimentionality = 7 # number of chosen test embedding dimentions
num_graphs = 3 # formal reply graph, author graph, time graph
#####

batch_of_raw_message_embeddings = torch.randn((batch_size, num_messages, embedding_dimentionality)) # batch of messages
batch_of_raw_graphs = torch.randn((batch_size, num_graphs, num_messages, num_messages)) # batches of hypergraphs

print(batch_of_raw_graphs.shape)
print(batch_of_raw_message_embeddings.shape)

# from from_colab_gpt import Zero_block
# zb = Zero_block()