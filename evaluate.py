from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def calculate_similarity(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0][0]

# Example: Compare two image embeddings
original_embedding = np.random.rand(512)
personalized_embedding = np.random.rand(512)

similarity = calculate_similarity(personalized_embedding, original_embedding)
print(f"Identity Similarity: {similarity:.2f}")
