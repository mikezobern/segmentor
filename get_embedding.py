from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
client = OpenAI()

def get_embedding(text, model="text-embedding-3-small"):
   return client.embeddings.create(input = text, model=model).data[0].embedding


def get_embedding_batch(text_list, model="text-embedding-3-small"):
   response = client.embeddings.create(input = text_list, model=model).data
   re = [response[i].embedding for i in range(len(response))]
   return re

# g = get_embedding_batch(['fox','wolf'])[0]
# print(g)