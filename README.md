# DocuRAG

DocuRAG est une application locale permettant d'interroger des documents
PDF, TXT et Markdown à l'aide d'une recherche sémantique et d'un modèle
de langage local.

## Fonctionnalités

- Import de documents PDF, TXT et Markdown
- Découpage des documents en segments
- Création d'embeddings
- Recherche vectorielle avec Chroma
- Mode recherche sémantique
- Mode assistant RAG avec Mistral local
- Affichage des extraits utilisés
- Réinitialisation de la session

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt