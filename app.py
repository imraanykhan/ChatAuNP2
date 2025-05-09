from flask import Flask, render_template, request
from openai import OpenAI
from dotenv import load_dotenv
import os, textwrap, datetime

# api key
load_dotenv()                                   # pulls .env locally; on Render use dashboard var
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# system prompt, can trim down
paper_context = textwrap.dedent("""
### Gao & Torrente‑Murciano 2020
- Two‑step reduction Au³⁺→Au⁺→Au⁰; Au⁺ step rate‑limiting.
- Dicarboxyacetone (DC2⁻) strengthens reduction, yields monodispersity.

### Kimling 2006
- Size 9–120 nm tunable via citrate:Au + temperature.
- >85 nm gives elongated shapes; SPR shoulder ≈ 650 nm.

### Dong 2020
- Best PDI < 0.20 at citrate:Au = 2.4–3.2 (15–30 nm).
- <2.0 ratio → bimodal distribution; scale‑up to 1.5 L possible.
""").strip()

system_prompt = f"""
You are ChatAuNP, an AI advisor on gold nanoparticle synthesis via the Turkevich method.
Your task: recommend concrete parameter changes for a requested morphology or size.

Key rules:
- Lower citrate:Au (<1.5) ⇒ larger / anisotropic (stars, rods).
- High temp (~100 °C) ⇒ fast nucleation, smaller NPs.
- AgNO₃, tannic acid, or pH tweaks steer nanostars, nanorods, nanoflowers.
- Citrate is reductant + stabilizer; DC2⁻ (from citrate) is a stronger reductant.
- Seed formation rate and ligand coordination govern final shape.

Literature context:
{paper_context}
"""

# flask app
app = Flask(__name__)

def ask_chataunp(question: str) -> str:
    """Call GPT‑4‑turbo and return answer text."""
    resp = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": question}
        ],
        temperature=0.3,
        max_tokens=500
    )
    return resp.choices[0].message.content.strip()

# routes
@app.route("/", methods=["GET", "POST"])
def index():
    answer = ""
    if request.method == "POST":
        question = request.form.get("question", "").strip()
        if question:
            answer = ask_chataunp(question)
            # Log Q&A with timestamp
            with open("log.txt", "a") as f:
                ts = datetime.datetime.utcnow().isoformat()
                f.write(f"{ts}\nQ: {question}\nA: {answer}\n---\n")
    return render_template("index.html", answer=answer)

# run locally
if __name__ == "__main__":
    app.run(debug=True)        # set debug=False in production
