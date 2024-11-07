# rag-with-tabular-data



### Deleting All Qdrant Collections

```bash
QDRANT_URL="http://localhost:6333"; curl -s "$QDRANT_URL/collections" | jq -r '.result.collections[].name' | xargs -I {} curl -X DELETE "$QDRANT_URL/collections/{}"
```