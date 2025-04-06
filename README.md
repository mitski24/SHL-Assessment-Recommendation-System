# SHL-Assessment-Recommendation-System
Project Description
In the modern hiring landscape, choosing the right assessments for a job role is critical. This AI system helps automate that by recommending SHL assessments based on a job description or natural language query. Built with a Retrieval-Augmented Generation (RAG) architecture, it leverages vector embeddings and optional LLM reranking to provide intelligent, ranked suggestions.

This system can be used by:

Recruiters who want to match assessments to job roles.

Hiring managers looking for test suggestions for new roles.

Candidates curious about tests for certain skills or jobs.


Features
🔍 Semantic Search with embeddings (RAG-style)

💬 Optional LLM reranking for relevance

📃 Web UI built with Streamlit

⚡ Fast API backend with vector search

📊 Supports top-K recommendations (1 to 10)

📈 Future support for metrics like Recall@K and MAP@K

Libraries
Library	Purpose
sentence-transformers	To generate sentence embeddings using pre-trained transformer models (all-MiniLM-L6-v2)
faiss	For high-speed similarity search over dense vectors
fastapi	Lightweight API framework to expose the recommendation service
uvicorn	ASGI server to run the FastAPI backend
streamlit	Rapid prototyping of interactive UI for the recommender system
pydantic	Data validation for FastAPI models
numpy	Numeric computation and array handling
requests	Communicating between the frontend and backend
json	Reading/writing structured assessment data
scikit-learn (optional)	Can be used later for evaluation metrics (Recall@K, MAP@K)

