from sentence_transformers import SentenceTransformer, CrossEncoder

labse_model = SentenceTransformer('sentence-transformers/LaBSE', device='cpu')
reranker_model = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
