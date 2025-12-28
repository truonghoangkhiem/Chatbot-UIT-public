"""
Test embedding model to check output dimensions
"""
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from app.config.settings import settings

print(f"Loading model: {settings.emb_model}")
embedding_model = HuggingFaceEmbedding(model_name=settings.emb_model)

test_text = "Điều 1. Cập nhật Quy chế Đào tạo"

print(f"\nTest text: {test_text}")
vector = embedding_model.get_text_embedding(test_text)

print(f"\nVector type: {type(vector)}")
print(f"Vector length: {len(vector) if hasattr(vector, '__len__') else 'N/A'}")
print(f"Vector shape: {vector.shape if hasattr(vector, 'shape') else 'N/A'}")
print(f"First 10 values: {vector[:10] if hasattr(vector, '__getitem__') else vector}")

# Try to convert to list
if hasattr(vector, 'tolist'):
    vector_list = vector.tolist()
    print(f"\nAs list length: {len(vector_list)}")
    print(f"First 10 as list: {vector_list[:10]}")
