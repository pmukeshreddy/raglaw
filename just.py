import numpy as np

from data_loader import DataLoader

JUDGMENT_DIR = "./dataset/IN-Abs/train-data/judgement"
SUMMARY_DIR = "./dataset/IN-Abs/train-data/summary"

judgment_embeddings, summary_embedding = np.load("./judgment_embeddings.npy",allow_pickle=True) , np.load("./summary_embeddings.npy",allow_pickle=True)

print(len(judgment_embeddings))
print(len(summary_embedding))



data_loader = DataLoader(JUDGMENT_DIR, SUMMARY_DIR)

raw_data = data_loader.prepare_data()
filtered_data = data_loader.filter_data(raw_data)


print(len(filtered_data))