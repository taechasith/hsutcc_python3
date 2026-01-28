import os
import math
import logging
from flask import Flask, request
from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()

app = Flask(__name__)

logging.basicConfig(
    filename="friend_finder.log",
    level=logging.INFO,
    format="%(asctime)s %(message)s"
)

messages_db = []

def get_client():
    key = os.getenv("MISTRAL_API_KEY", "").strip()
    if not key:
        return None, "MISTRAL_API_KEY is missing. Put it in a .env file next to this script."
    return Mistral(api_key=key), None

def cosine_sim(a, b):
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))

def embed_text(client, text):
    res = client.embeddings.create(
        model="mistral-embed",
        inputs=[text]
    )
    return res.data[0].embedding

def llm_filter(client, new_text, top3):
    prompt = f"New message:\n{new_text}\n\nTop-3 candidates:\n"
    for i, r in enumerate(top3, start=1):
        prompt += f"{i}. {r['nickname']}: {r['text']}\n"
    prompt += "\nPick which ones are truly relevant (similar thought/intent). Reply ONLY numbers like: 1,3 or NONE."

    out = client.chat.complete(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return out.choices[0].message.content.strip()

@app.get("/")
def home():
    return """
    <h2>Find a friend for the moment</h2>
    <form action="/submit" method="post">
      <div>Nickname: <input name="nickname" required></div><br>
      <div>Message: <input name="message" required style="width:420px"></div><br>
      <button type="submit">Find</button>
    </form>
    """

@app.post("/submit")
def submit():
    client, err = get_client()
    if err:
        return f"<pre>{err}</pre><p><a href='/'>Back</a></p>", 400

    nickname = request.form.get("nickname", "").strip()
    text = request.form.get("message", "").strip()

    if not nickname or not text:
        return "<pre>Nickname and message are required.</pre><p><a href='/'>Back</a></p>", 400

    try:
        emb = embed_text(client, text)
    except Exception as e:
        return f"<pre>Embedding error: {e}</pre><p><a href='/'>Back</a></p>", 500

    sims = []
    for rec in messages_db:
        score = cosine_sim(emb, rec["embedding"])
        sims.append({"nickname": rec["nickname"], "text": rec["text"], "score": score})

    sims.sort(key=lambda r: r["score"], reverse=True)
    top3 = sims[:3]

    logging.info(f"ADD nickname={nickname} text={text}")
    logging.info(f"TOP3 {[(r['nickname'], round(r['score'], 4)) for r in top3]}")

    recommended = []
    if top3:
        try:
            llm_out = llm_filter(client, text, top3)
        except Exception as e:
            llm_out = f"LLM_ERROR: {e}"

        chosen = []
        if llm_out.strip().upper() != "NONE":
            for part in llm_out.split(","):
                part = part.strip()
                if part.isdigit():
                    idx = int(part) - 1
                    if 0 <= idx < len(top3):
                        chosen.append(top3[idx])

        recommended = chosen
        logging.info(f"LLM_OUT {llm_out}")
        logging.info(f"RECOMMENDED {[r['nickname'] for r in recommended]}")
    else:
        logging.info("TOP3 []")
        logging.info("RECOMMENDED []")

    messages_db.append({"nickname": nickname, "text": text, "embedding": emb})

    html = "<h2>Top-3 by cosine similarity</h2>"
    if top3:
        html += "<ul>"
        for r in top3:
            html += f"<li><b>{r['nickname']}</b>: {r['text']} (score {r['score']:.3f})</li>"
        html += "</ul>"
    else:
        html += "<p>No previous messages yet.</p>"

    html += "<h2>Friend recommendation</h2>"
    if recommended:
        html += "<ul>"
        for r in recommended:
            html += f"<li><b>{r['nickname']}</b>: {r['text']}</li>"
        html += "</ul>"
    else:
        html += "<p>No strong match this time.</p>"

    html += "<p><a href='/'>Back</a></p>"
    return html

if __name__ == "__main__":
    app.run(port=5000, debug=True)
