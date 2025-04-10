import json
from sentence_transformers import SentenceTransformer, util

# Load pre-indexed SHL assessment catalog
with open("shl_catalog.json", "r", encoding="utf-8") as f:
    catalog = json.load(f)

model = SentenceTransformer('all-MiniLM-L6-v2')  # Small & Streamlit-friendly

# Precompute catalog embeddings
catalog_embeddings = model.encode([item["title"] + " " + item["description"] for item in catalog], convert_to_tensor=True)

def get_assessment_recommendation(job_desc, top_k=3):
    jd_embedding = model.encode(job_desc, convert_to_tensor=True)
    hits = util.semantic_search(jd_embedding, catalog_embeddings, top_k=top_k)[0]
    return [catalog[hit["corpus_id"]] for hit in hits]
