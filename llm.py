# backend/main.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
import uvicorn
import datetime
import difflib
from transformers import pipeline  # or use OpenAI API

qa_pipeline = pipeline("question-answering")
app = FastAPI()
model = SentenceTransformer('all-MiniLM-L6-v2')

# Setup SQLite DB
conn = sqlite3.connect('memory.db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT,
    embedding BLOB,
    created_at TEXT
)
''')
conn.commit()

# Helpers
class Memory(BaseModel):
    text: str

class Query(BaseModel):
    question: str

def embed(text):
    return model.encode([text])[0].tolist()

def get_all_memories():
    cursor.execute("SELECT id, text, embedding, created_at FROM memories")
    return cursor.fetchall()

def cosine_search(question_embedding):
    memories = get_all_memories()
    similarities = []
    for m_id, text, emb, created in memories:
        emb_vector = eval(emb)
        score = cosine_similarity([question_embedding], [emb_vector])[0][0]
        similarities.append((score, text, created))
    similarities.sort(reverse=True)
    return similarities[0] if similarities else None

@app.post("/store")
async def store_memory(memory: Memory):
    emb = embed(memory.text)
    now = datetime.datetime.now().isoformat()
    cursor.execute("INSERT INTO memories (text, embedding, created_at) VALUES (?, ?, ?)",
                   (memory.text, str(emb), now))
    conn.commit()
    return {"message": "Memory stored"}

@app.post("/query")
async def query_memory(query: Query):
    question = query.question
    question_emb = embed(question)

    # Get all stored memories
    memories = get_all_memories()
    results = []

    for m_id, text, emb, created in memories:
        emb_vector = eval(emb)
        score = cosine_similarity([question_emb], [emb_vector])[0][0]

        # You can adjust this threshold
        if score > 0.4:
            # Run QA pipeline for each match
            result = qa_pipeline(question=question, context=text)
            results.append({
                "answer": result["answer"],
                "confidence": round(score * 100, 2),
                "matched_memory": text,
                "date_stored": created
            })

    # Sort by confidence
    results.sort(key=lambda x: x["confidence"], reverse=True)

    if results:
        return {"answers": results}

    return {"answers": []}


@app.post("/update")
async def update_memory(request: Request):
    data = await request.json()
    old = data.get("old")
    new = data.get("new")
    cursor.execute("SELECT id FROM memories WHERE text LIKE ?", (f"%{old}%",))
    row = cursor.fetchone()
    if row:
        new_emb = embed(new)
        cursor.execute("UPDATE memories SET text=?, embedding=? WHERE id=?", (new, str(new_emb), row[0]))
        conn.commit()
        return {"message": "Memory updated"}
    return {"message": "Original memory not found"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
