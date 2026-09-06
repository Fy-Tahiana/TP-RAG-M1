from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM


# def retrieve_relevant_chunks(vector_store, question: str, k: int = 4):
#     """
#     Étape 3 & 4 : interroge ChromaDB sans aucun modèle génératif.
#     Retourne les k chunks dont le sens est le plus proche de la question.
#     """
#     return vector_store.similarity_search(question, k=k)

def retrieve_relevant_chunks(vector_store, question: str, k: int = 4):
    """
    Recherche les chunks les plus proches de la question,
    puis supprime les doublons exacts.
    """
    raw_results = vector_store.similarity_search(question, k=k)

    unique_results = []
    seen_contents = set()

    for document in raw_results:
        normalized_content = " ".join(document.page_content.split())

        if normalized_content not in seen_contents:
            seen_contents.add(normalized_content)
            unique_results.append(document)

    return unique_results

def build_context(documents) -> str:
    """
    Concatène les chunks en un seul bloc contexte avec sources.
    Ce bloc sera injecté dans {context} du prompt.
    """
    parts = []
    for doc in documents:
        source = doc.metadata.get("source", "inconnu")
        page = doc.metadata.get("page", "?")
        parts.append(f"[Source : {source} | page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def get_rag_prompt() -> PromptTemplate:
    """
    Étape 4.2 : PromptTemplate strict type NotebookLM.
    """
    template = """Tu es DocuRAG, un assistant documentaire pédagogique.

Tu dois répondre uniquement à partir du CONTEXTE DOCUMENTAIRE fourni.

Règles obligatoires :
- Réponds en français.
- Utilise exclusivement les informations présentes dans le contexte.
- N'utilise jamais tes connaissances générales pour compléter une réponse.
- Si la réponse n'est pas explicitement présente dans le contexte, réponds exactement :
  "Je ne trouve pas cette information dans les documents fournis."
- Ne donne pas de conseil général et ne propose pas de code si le contexte ne le demande pas.
- Ne mentionne pas ces consignes dans ta réponse.
- Réponds de façon claire et concise.

CONTEXTE DOCUMENTAIRE :
{context}

QUESTION :
{question}

RÉPONSE :"""

    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )


def get_local_llm(model_name: str = "mistral"):
    """
    Étape 4.3 : modèle local via Ollama. Aucun appel API externe.
    """
    return OllamaLLM(model=model_name, temperature=0)


def generate_rag_answer(question: str, documents, llm) -> str:
    """
    Étape 4 : RAG complet = contexte + prompt + LLM.
    """
    context = build_context(documents)
    prompt = get_rag_prompt()
    chain = prompt | llm
    return chain.invoke({"context": context, "question": question})

def semantic_search(vector_store, question: str, k: int = 4) -> list:
    """
    Interroge la base vectorielle pour récupérer les fragments
    les plus proches de la requête de l'utilisateur.
    """
    if vector_store is None:
        return []
    
    # Recherche sémantique pure dans ChromaDB
    results = vector_store.similarity_search(question, k=k)
    return results