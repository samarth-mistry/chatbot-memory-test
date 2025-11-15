# 🤖 Gemini 2.5 Flash Conversational Chatbot

This project implements a multi-turn, stateful chatbot using **FastAPI** and **LangGraph** with **Gemini 2.5 Flash** for all NLU, Guardrail, and Knowledge Base tasks. It fulfills all the requirements of the Chatbot Development Assignment, including fuzzy time parsing, mid-conversation topic switching, and ethical guardrails.

## ⚙️ Technical Stack

* **Framework:** **FastAPI** (for the single `/chat` endpoint)
* **Conversational Logic:** **LangGraph** (State Machine / Agentic Flow)
* **NLU/AI Model:** **Google Gemini 2.5 Flash**
* **Session State:** In-memory checkpointer (`MemorySaver` from LangGraph)
* **Deployment:** Docker

## 🚀 Setup and Run

### Prerequisites

1.  Python 3.11+
2.  **Docker** (highly recommended)

### 1. Local Setup

1.  **Navigate to project folder.**
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Set API Key:**
    Create a file named `.env` in the project root and add your key:
    ```
    GOOGLE_API_KEY="YOUR_API_KEY_HERE"
    ```
4.  **Run the server:**
    ```bash
    uvicorn app.main:app --reload
    ```

### 2. Docker Setup (Recommended)

1.  **Build the Docker image:**
    ```bash
    docker build -t chatbot-gemini .
    ```
2.  **Run the container:**
    ```bash
    # Pass the API key as an environment variable
    docker run -d -p 8000:8000 --env GOOGLE_API_KEY="YOUR_API_KEY_HERE" chatbot-gemini
    ```
    The bot will be accessible at `http://localhost:8000`.

## 🧪 Testing the Chatbot (Full User Journey)

The bot maintains session state via the `user_id` passed in the JSON payload.

**Endpoint:** `POST http://127.0.0.1:8000/chat`
**Headers:** `Content-Type: application/json`

Use a tool like `curl` to test the full conversational flow:

| Step | User Input (`message` field) | Expected Bot Action |
| :--- | :--- | :--- |
| **1 (Fuzzy Time)** | `I want to book a table this weekend or maybe Monday morning.` | **Clarifies** ambiguous dates. |
| **2 (Clarification)** | `Sunday, please.` | Sets the date; asks for party size. |
| **3 (Party Size)** | `For four people.` | Sets size; asks for final confirmation. |
| **4 (Topic Switch)** | `By the way, what's the capital of Australia?` | Answers the factual question, then **resumes the confirmation question** (e.g., "Now, back to your reservation... Should I confirm that?"). |
| **5 (Guardrail)** | `You're an idiot.` | Replies with: **"Let's keep our conversation respectful, please."** |
| **6 (Final Confirm)** | `Yes-please confirm the reservation.` | Confirms the reservation details. |

***

This completes all phases of the development assignment. The codebase is now structured, functional, and ready for review and deployment.

Do you have any final questions or require any specific explanation of the code structure before concluding?

## 🐳 Run, build and run with Docker Compose

Here are the exact commands you requested for running manually, building the Docker image, and running with Docker Compose.

1) Run locally (module entrypoint, works even if the `uvicorn` console script is not on PATH):

```bash
python -m uvicorn main:app --reload
```

2) Build the Docker image (from the repository root):

```bash
docker build -t chatbot-memory-test .
```

3) Bring up the service using Docker Compose (detached):

```bash
docker compose up -d
```

Notes:
- If you want `docker compose` to rebuild the image before starting, run `docker compose up -d --build`.
- If you prefer the `uvicorn` CLI command (instead of `python -m uvicorn`), ensure the Python scripts location (e.g. `~/.local/bin`) is on your PATH or install into an activated virtualenv.
