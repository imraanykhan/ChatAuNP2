from flask import Flask, render_template, request
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load API key from .env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ChatAuNP system prompt (can move to file later)
system_prompt = """
You are ChatAuNP, an AI advisor on gold nanoparticle synthesis via the Turkevich method...
[Insert your full system_prompt + paper context here]
"""

# Flask app
app = Flask(__name__)

# Ask OpenAI
def ask_chataunp(question):
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        temperature=0.3,
        max_tokens=500
    )
    return response.choices[0].message.content.strip()

# Home route
@app.route("/", methods=["GET", "POST"])
def index():
    answer = ""
    if request.method == "POST":
        question = request.form["question"]
        answer = ask_chataunp(question)

        # Save log
        with open("log.txt", "a") as f:
            f.write(f"Q: {question}\nA: {answer}\n---\n")

    return render_template("index.html", answer=answer)

if __name__ == "__main__":
    app.run(debug=True)
