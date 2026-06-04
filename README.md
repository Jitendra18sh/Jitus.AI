# Jitus AI - Autonomous Voice AI Tutor

## Overview

Jitus AI is a production-grade Voice AI Assistant designed to act as a professional MCA-level AI tutor. It combines real-time voice interaction, Large Language Models (LLMs), memory, Retrieval-Augmented Generation (RAG), and tool calling to provide intelligent, context-aware, and interview-friendly explanations.

The primary goal of Jitus AI is to help students learn technical concepts in a structured and easy-to-understand manner.

---

## Features

### Voice Interaction

* Speech-to-Text (STT)
* Text-to-Speech (TTS)
* Real-time voice conversations
* Human-like responses

### AI Tutor Mode

For every response, Jitus AI:

1. Gives a Definition
2. Explains Step-by-Step
3. Provides a Real-World Example
4. Provides a Technical Example (if applicable)
5. Lists Advantages
6. Lists Disadvantages
7. Uses Simple Interview-Friendly English

### Memory System

* Conversation history
* Context retention
* Long-term memory support

### RAG (Retrieval-Augmented Generation)

* PDF Question Answering
* Custom Knowledge Base
* Document Search

### AI Agent Capabilities

* Web Search
* Calculator
* File Analysis
* Resume Review
* Data Analytics
* Coding Assistance

---

## System Architecture

User Voice
↓
Speech-to-Text
↓
LangGraph Agent
↓
Memory Retrieval
↓
Tool Selection
↓
LLM Reasoning
↓
Text-to-Speech
↓
Voice Response

---

## Tech Stack

### Backend

* Python
* FastAPI
* LangGraph
* LangChain

### AI Models

* OpenAI GPT Models

### Voice

* OpenAI Realtime API
* ElevenLabs

### Memory & Storage

* ChromaDB
* PostgreSQL

### Frontend

* React
* Next.js

---

## Project Structure

```text
Jitus-AI/

├── backend/
│
├── agents/
│   ├── planner_agent.py
│   ├── memory_agent.py
│   ├── tutor_agent.py
│   ├── rag_agent.py
│   └── tool_agent.py
│
├── memory/
│   ├── chroma_store.py
│   └── postgres_store.py
│
├── tools/
│   ├── calculator.py
│   ├── web_search.py
│   ├── weather.py
│   └── file_reader.py
│
├── api/
│   └── routes.py
│
├── frontend/
│
├── docs/
│
├── .env
├── requirements.txt
├── main.py
└── README.md
```

---

## Installation

### Clone Repository

```bash
git clone <repository-url>
cd Jitus-AI
```

### Create Virtual Environment

```bash
python -m venv env
```

### Activate Virtual Environment

Windows:

```bash
env\Scripts\activate
```

Linux/Mac:

```bash
source env/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key
```

---

## Running the Project

```bash
python main.py
```

or

```bash
uvicorn main:app --reload
```

---

## Example Response Format

### Definition

A Stack is a linear data structure that follows the Last In First Out (LIFO) principle.

### Step-by-Step Explanation

1. Elements are inserted using Push.
2. Elements are removed using Pop.
3. The last inserted element is removed first.

### Real-World Example

A stack of plates in a kitchen.

### Technical Example

```python
stack = []

stack.append(10)
stack.append(20)

stack.pop()
```

### Advantages

* Easy implementation
* Fast insertion and deletion

### Disadvantages

* Limited access
* Cannot access middle elements directly

### Interview Tip

A stack follows the LIFO principle, meaning the last inserted element is removed first.

---

## Future Enhancements

* Real-time streaming conversations
* Multi-agent architecture
* Personalized learning paths
* Voice cloning
* Multilingual support
* Autonomous research agent
* Advanced memory management

---

## Project Goal

Build a recruiter-ready AI Engineering portfolio project demonstrating:

* Generative AI
* AI Agents
* LangGraph
* LangChain
* RAG
* Vector Databases
* Voice AI
* FastAPI
* Real-Time Systems

---

## Author

Jitendra Kumar

MCA Student | AI Engineer | Machine Learning Enthusiast

Project Name: Jitus AI
