from nano_vectordb import NanoVectorDB
import numpy as np

data_len = 5
fake_dim = 1024
fake_embeds = np.random.rand(data_len, fake_dim)    

fakes_data = [{"__vector__": fake_embeds[i]} for i in range(data_len)]

vdb = NanoVectorDB(fake_dim, storage_file="fool.json")


r = vdb.upsert(fakes_data)
print(r["update"], r["insert"])
# will create/overwrite 'fool.json'
vdb.save()