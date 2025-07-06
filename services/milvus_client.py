import os
from pymilvus import connections, MilvusClient, DataType
from utils.logger import get_logger

logger = get_logger(__name__)

# Configuration
MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
VECTOR_DIM = 768 # LaBSE embedding dimension
SERVER_ADDR = f"http://{MILVUS_HOST}:{MILVUS_PORT}"

def connect_to_milvus():
    try:
        connections.connect(
            alias="default",
            uri=SERVER_ADDR,
            token=MILVUS_TOKEN
        )
        logger.info("Connected to Milvus")
        return MilvusClient(uri=SERVER_ADDR, token=MILVUS_TOKEN)
    except Exception as e:
        logger.error(f"Milvus connection failed: {e}")
        raise

def initialize_collection(client):
    try:
        if not client.has_collection(COLLECTION_NAME):
            schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
            schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=100)
            schema.add_field("vector", DataType.FLOAT_VECTOR, dim=VECTOR_DIM)

            index_params = MilvusClient.prepare_index_params()
            index_params.add_index("vector", "IVF_FLAT", "COSINE", {"nlist": 1024})

            client.create_collection(COLLECTION_NAME, schema=schema, index_params=index_params)
            logger.info(f"Collection {COLLECTION_NAME} created.")
        client.load_collection(COLLECTION_NAME)
    except Exception as e:
        logger.error(f"Collection init failed: {e}")
        raise
