# DRAFT SCRIPT
from setup_loader import SetupLoader
from numpy import dot
from numpy.linalg import norm
from langchain.embeddings.openai import OpenAIEmbeddings

app_setup = SetupLoader()
base_llm, logger, global_conf = (
    app_setup.chat_openai,
    app_setup.logger,
    app_setup.global_conf,
)


def cosine_similarity(vec1, vec2) -> float:
    """Compute the cosine similarity between two vectors.
    Args:
        vec1 (np.array): The first vector.
        vec2 (np.array): The second vector.
    Returns:
        float: The cosine similarity score.
    """
    cos_sim = dot(vec1, vec2) / (norm(vec1) * norm(vec2))
    return cos_sim


def reranked_retrieval(question, retriever, top_n=10):
    """Rerank the top N results retrieved from the vector store.\
    Args:
        question (str): The input question.
        retriever (Retriever): The retriever object.
        top_n (int): The number of results to retrieve and rerank.
    Returns:
        list: The reranked results.
    """
    embeddings = OpenAIEmbeddings()

    # Retrieve top N results based on initial similarity
    initial_results = retriever.retrieve(question, top_n)

    # Compute similarity scores for each result
    question_embedding = embeddings.embed(question)
    scored_results = []
    for result in initial_results:
        result_embedding = embeddings.embed(result)
        similarity_score = cosine_similarity(question_embedding, result_embedding)
        scored_results.append((result, similarity_score))

    # Sort results by similarity score
    scored_results.sort(key=lambda x: x[1], reverse=True)

    # Select top results after sorting
    top_results = [result for result, score in scored_results[:top_n]]

    return top_results
