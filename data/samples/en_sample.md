# Retrieval-Augmented Generation

## What it is

RAG retrieves relevant passages from a document collection and adds them to the
prompt, so the model answers from those passages instead of only from memory.

## Pipeline

1. Load documents
2. Split into chunks
3. Embed chunks
4. Store vectors
5. Retrieve the closest chunks for a question
6. Generate an answer with citations
