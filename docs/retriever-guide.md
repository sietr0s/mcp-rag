# Retriever Guide

Retriever is responsible for finding relevant chunks for the user query.

## Main Idea

1. Convert query to embedding.
2. Search top-k chunks in vector storage.
3. Return chunks with metadata for further generation.

## Notes

- Keep chunks small and focused.
- Use similarity threshold to reduce noise.
