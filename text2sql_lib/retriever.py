import sys
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction


def retrieve_similar_examples(question: str, n_results: int = 3, offset: int = 0):
    """
    Retrieves the most similar examples from the ChromaDB collection.
    """
    # Initialize ChromaDB persistent client
    client = chromadb.PersistentClient(path="./chroma_db")

    # Use the same embedding function as when indexing
    embedding_fn = OllamaEmbeddingFunction(
        model_name="nomic-embed-text",
        url="http://localhost:11434/api/embeddings",
    )

    # Get existing collection
    collection = client.get_collection(name="synthetic_text_to_sql")

    # Query top n results
    results = collection.query(
        query_texts=[question],
        n_results=n_results + offset,
        include=["documents", "metadatas", "distances"],
    )

    # Apply offset to the results
    for key in ['documents', 'metadatas', 'distances']:
        if results[key] and results[key][0]:
            results[key][0] = results[key][0][offset:]

    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python retrieve.py <question>")
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    results = retrieve_similar_examples(question)

    # Print results
    print(f"Top 3 results for: '{question}'\n")
    for idx, (doc, meta, dist) in enumerate(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ),
        start=1,
    ):
        print(f"Result {idx} (distance: {dist:.4f}):")
        print(f"Question: {doc}")
        print("Metadata:")
        for k, v in meta.items():
            print(f"  {k}: {v}")
        print()


if __name__ == "__main__":
    main()
