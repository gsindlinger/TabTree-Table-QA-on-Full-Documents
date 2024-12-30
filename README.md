# Exploring Table Representations for RAG on Large Documents

[![Download Thesis WIP](https://img.shields.io/badge/Download--PDF-Thesis--WIP-orange)](https://www.overleaf.com/read/mqphwrjjhytz#5746a7)

### Introduction / Motivation

This repository explores the capabilities of LLMs for Table QA in documents containing both text and tables. While LLMs have revolutionized tasks involving unstructured text, their architecture — rooted in the transformer model — is primarily designed for sequential data, making it less suited to the inherently two-dimensional structure of tabular data. Existing research highlights the limitations of LLMs in handling tabular heterogeneity, data sparsity, and correlations, leaving room for improvement in tasks like Table QA, particularly when tables are embedded in complex documents.    

This project aims to bridge these gaps by introducing a framework for Table QA on large documents (incorporating tables and text), with the following core components:  
- **TabTree**: A new table serialization method that models tables as directed tree structures.
- A **Retrieval-Augmented Generation (RAG)** framework tailored for the specific Table QA task on large documents.  
- A manually labeled evaluation dataset to benchmark the effectiveness of table representations and retrieval strategies.  

### Research Questions  

The repository addresses the following research questions:  
- How well do LLMs perform in Table QA on large documents with integrated text and tables?  
- Which table representation yields the best quality in Table QA?  
  - How does table representation affect retrieval effectiveness in a RAG pipeline?  
  - How does it influence the generative QA process of LLMs?  
- Which components of a RAG pipeline are most critical for generating accurate responses?  


### Deleting All Qdrant Collections

```bash
QDRANT_URL="http://localhost:6333"; curl -s "$QDRANT_URL/collections" | jq -r '.result.collections[].name' | xargs -I {} curl -X DELETE "$QDRANT_URL/collections/{}"
```
