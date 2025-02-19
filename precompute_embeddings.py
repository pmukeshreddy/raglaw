from embeddings import Embedder
from data_loader import DataLoader
import numpy as np


JUDGMENT_DIR = "/Users/mukeshreddypochamreddy/Documents/rag deplowyment final itteration/dataset/IN-Abs/train-data/judgement"
SUMMARY_DIR = "/Users/mukeshreddypochamreddy/Documents/rag deplowyment final itteration/dataset/IN-Abs/train-data/summary"



embedder = Embedder()
data_loader = DataLoader(JUDGMENT_DIR, SUMMARY_DIR)
raw_data = data_loader.prepare_data()
filtered_data = data_loader.filter_data(raw_data)
judgment_embeddings, summary_embedding = embedder.compute_embeddings(filtered_data)

np.save("judgment_embeddings.npy", judgment_embeddings)
np.save("summary_embedding.npy", summary_embedding)

