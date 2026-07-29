"""
Retrieval-Augmented Generation over the collected news articles:
embeds articles with SentenceTransformers, indexes them with FAISS,
and answers natural-language questions using Cohere's chat API,
grounded only in the retrieved news context.
"""

import numpy as np
import faiss
import cohere

from sentence_transformers import SentenceTransformer

from src.config import COHERE_API_KEY, EMBEDDING_MODEL_NAME, COHERE_CHAT_MODEL


class NewsRAG:
    def __init__(self, news_database: dict, embedding_model_name: str = EMBEDDING_MODEL_NAME):
        if not COHERE_API_KEY:
            raise ValueError(
                "COHERE_API_KEY is not set. Copy .env.example to .env and add your key."
            )

        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.co = cohere.ClientV2(api_key=COHERE_API_KEY)

        self.documents = []
        self.metadata = []
        self._build_documents(news_database)
        self._build_index()

    def _build_documents(self, news_database: dict):
        for ticker, df in news_database.items():
            for _, row in df.iterrows():
                text = " ".join([
                    str(row["Title"]),
                    str(row["Description"]),
                    str(row["Content"]),
                ])
                self.documents.append(text)
                self.metadata.append({
                    "Ticker": ticker,
                    "Company": row["Company"],
                    "Date": row["Date"],
                })

        print("Documents:", len(self.documents))

    def _build_index(self):
        embeddings = self.embedding_model.encode(self.documents, show_progress_bar=True)
        embeddings = np.array(embeddings)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

        print("Indexed documents:", self.index.ntotal)

    def retrieve(self, query: str, k: int = 5):
        query_embedding = self.embedding_model.encode([query])
        _, indices = self.index.search(np.array(query_embedding), k)

        return [self.documents[idx] for idx in indices[0]]

    def answer(self, question: str) -> str:
        retrieved = self.retrieve(question)
        context = "\n\n".join(retrieved)

        response = self.co.chat(
            model=COHERE_CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a financial AI assistant. Answer ONLY using the "
                        "provided news context. If the answer is not present in "
                        "the context, clearly say that the information is not available."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                },
            ],
        )

        return response.message.content[0].text
