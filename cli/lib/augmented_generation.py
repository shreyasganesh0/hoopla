from .hybrid_search import rrf_search, HybridSearch 
from .llm_prompt import Llm
import json 

RRF_K = 60
DEFAULT_LIMIT = 5

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
        search_results = rrf_search(query, RRF_K, DEFAULT_LIMIT, enhance="", rerank="")
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

def perform_summary(query, limit=DEFAULT_LIMIT):
    search_results = []
    summary_response = ""
    error_message = None

    try:
        movies = []
        try:
            with open("data/movies.json", "r") as f:
                data = json.load(f)
                movies = data["movies"]
        except FileNotFoundError:
             error_message = "Error: data/movies.json not found."
             return search_results, summary_response, error_message
        except json.JSONDecodeError:
             error_message = "Error: Could not decode data/movies.json."
             return search_results, summary_response, error_message

        search_results = rrf_search(query, RRF_K, limit, enhance="", rerank="")

        if not search_results:
            error_message = "No search results found."
            return search_results, summary_response, error_message

    except Exception as e:
        error_message = f"Error during search: {e}"
        return search_results, summary_response, error_message

    results_for_prompt = format_docs_for_prompt(search_results)

    prompt = f"""Provide information useful to this query by synthesizing information from multiple search results in detail.
The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Search Results:
{results_for_prompt}

Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""

    try:
        llm = Llm()
        summary_response = llm.generate_summary(prompt) 
    except Exception as e:
        error_message = f"Error generating summary from LLM: {e}"
        summary_response = "Failed to generate summary."

    return search_results, summary_response, error_message

def format_docs_for_citation_prompt(results):
    doc_strings = []
    for i, res in enumerate(results, 1):
        title = res.get("title", "N/A")
        description_snippet = res.get("document", "").split('\n')[0][:200] + "..."
        # Add index for citation
        doc_strings.append(f"[{i}] {title}: {description_snippet}")
    return "\n".join(doc_strings)

def perform_rag_with_citations(query, limit=DEFAULT_LIMIT):
    search_results = []
    citation_response = ""
    error_message = None

    try:
        movies = []
        try:
            with open("data/movies.json", "r") as f:
                data = json.load(f)
                movies = data["movies"]
        except FileNotFoundError:
             error_message = "Error: data/movies.json not found."
             return search_results, citation_response, error_message
        except json.JSONDecodeError:
             error_message = "Error: Could not decode data/movies.json."
             return search_results, citation_response, error_message

        search_results = rrf_search(query, RRF_K, limit, enhance="", rerank="")

        if not search_results:
            error_message = "No search results found."
            return search_results, citation_response, error_message

    except Exception as e:
        error_message = f"Error during search: {e}"
        return search_results, citation_response, error_message

    docs_for_prompt = format_docs_for_citation_prompt(search_results)

    prompt = f"""Answer the question or provide information based on the provided documents.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

Query: {query}

Documents:
{docs_for_prompt}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources using [1], [2], etc. format when referencing information from the numbered documents above.
- If sources disagree, mention the different viewpoints
- If the answer isn't in the documents, say "I don't have enough information"
- Be direct and informative

Answer:"""

    try:
        llm = Llm()
        citation_response = llm.generate_answer_with_citations(prompt)
    except Exception as e:
        error_message = f"Error generating citation answer from LLM: {e}"
        citation_response = "Failed to generate answer with citations."

    return search_results, citation_response, error_message
