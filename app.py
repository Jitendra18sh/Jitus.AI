import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are Jitus AI, a professional MCA-level AI tutor.\n"
        "For every answer:\n"
        "- Give definition\n"
        "- Explain step-by-step\n"
        "- Give real-world example\n"
        "- Give technical example if applicable\n"
        "- List advantages & disadvantages\n"
        "- Use simple interview-friendly English\n"
    )
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    message = data.get("message")
    history = data.get("history", [])

    if not message:
        return jsonify({"error": "Message required"}), 400

    messages = [SYSTEM_PROMPT] + history + [
        {"role": "user", "content": message}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.6,
        max_tokens=600
    )

    return jsonify({
        "reply": response.choices[0].message.content
    })

if __name__ == "__main__":
    app.run(debug=True)
