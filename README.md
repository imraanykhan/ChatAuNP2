# ChatAuNP v1.01  
*GPT-4o–powered, retrieval-augmented for designing gold-nanoparticle (AuNP) syntheses via the Turkevich family of methods

---
zero shot gpt 4o, uses it for procedural generation
RAG with PDFs (allows for paper PDF uploads, that are autochunked and stored in a FAISS vector database, top-K chunks are injected to the prompt.
Format is JSON
Persistent vector store via a Render disk that keeps papers.index + mapping.pkl so the knowledge STAYS in redeploys
---
