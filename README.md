# Exploring Table Representations for RAG on Large Documents

[![Download Thesis WIP](https://img.shields.io/badge/Download--PDF-Thesis-WIP-orange)](https://www.overleaf.com/read/mqphwrjjhytz#5746a7)


### Deleting All Qdrant Collections

```bash
QDRANT_URL="http://localhost:6333"; curl -s "$QDRANT_URL/collections" | jq -r '.result.collections[].name' | xargs -I {} curl -X DELETE "$QDRANT_URL/collections/{}"
```
