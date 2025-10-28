from .hybrid_search import rrf_search
from .llm_prompt import Llm

RRF_K = 60
RESULT_LIMIT = 5

def format_docs_for_prompt(results):
    doc_strings = []
    for i, res in enumerate(results, 1):
        title = res.get("title", "N/A")
        description_snippet = res.get("document", "").split('\n')[0][:200] + "..."
        doc_strings.append(f"{i}. {title}: {description_snippet}")
    return "\n".join(doc_strings)

def perform_rag(query):
    search_results = []
    rag_response = ""
    error_message = None

    try:
        search_results = rrf_search(query, RRF_K, RESULT_LIMIT, enhance="", rerank="")
        if not search_results:
            error_message = "No search results found."
            return search_results, rag_response, error_message
    except Exception as e:
        error_message = f"Error during search: {e}"
        return search_results, rag_response, error_message

    docs_for_prompt = format_docs_for_prompt(search_results)

    prompt = f"""Answer the question or provide information based ONLY on the provided documents.
This answer should be tailored to Hoopla users. Hoopla is a movie streaming service.
If the documents do not contain relevant information to answer the query, state that clearly. Do not use outside knowledge.

Query: {query}

Documents:
{docs_for_prompt}

Provide a comprehensive answer that addresses the query based on the documents:"""

    try:
        llm = Llm()
        rag_response = llm.generate_answer(prompt)
    except Exception as e:
        error_message = f"Error generating response from LLM: {e}"
        rag_response = "Failed to generate response."

    return search_results, rag_response, error_message
