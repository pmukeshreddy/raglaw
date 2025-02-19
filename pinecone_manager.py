# pinecone_manager.py
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

class PineconeManager:
    def __init__(self, api_key, index_name):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.index = self.pc.Index(index_name)