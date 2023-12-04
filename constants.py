import chromadb
from chromadb.config import Settings
CHROMA_SETTINGS = chromadb.PersistentClient(
    path="db", 
    settings=Settings(anonymized_telemetry=False))