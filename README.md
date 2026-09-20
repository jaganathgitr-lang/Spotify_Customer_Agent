# Spotify Customer Support Crew

A 3-agent customer support system built with **CrewAI** (sequential process) and a **Streamlit** UI, built for the Social Eagle Gen AI Architect Program's Weekly Buildathon (Multi-Agent Customer Support).

When a question comes in, three agents run one after another:

```
User question
     |
     v
1. Assistant             -- answers directly, grounded in a RAG knowledge base
     |                       built from data/spotify_web_app_architecture.pdf
     v
2. Web Search Assistant   -- searches the web (Serper) and answers from the results
     |
     v
3. Entry Agent            -- saves the query + both answers to answers.txt,
                              then returns both answers to the user
     |
     v
Streamlit UI shows both answers
```

## The three agents

| Agent | Role |
|---|---|
| **Assistant** | Answers the query directly using a FAISS-backed RAG knowledge base built from `data/spotify_web_app_architecture.pdf` |
| **Web Search Assistant** | Searches the web via `SerperDevTool` and answers from the search results |
| **Entry Agent** | Saves the query and both answers to `answers.txt` (idempotent -- won't write a duplicate if it's accidentally called twice), then returns both answers |

Task 3 (Entry Agent) is given `context=[task1, task2]`, so it has both prior answers available before it writes the file -- this is what makes the "sequential" process actually pass work forward, not just run agents one after another independently.

## Setup

```bash
cd buildathon-support-crew
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in real keys:

```bash
cp .env.example .env
```

```
OPENAI_API_KEY=your-openai-key-here
SERPER_API_KEY=your-serper-key-here
```

- OpenAI key: https://platform.openai.com
- Serper key (free tier): https://serper.dev

Both are read from environment variables at runtime (`load_dotenv()`) -- never hard-coded, and `.env` is gitignored.

## Run it

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Enter a question, click **Submit**, and both answers appear once the crew finishes (a spinner shows while it's running). Every submission appends a new entry to `answers.txt` in the project folder.

## Project files

```
buildathon-support-crew/
├── app.py              # everything: RAG setup, agents, tasks, crew, Streamlit UI
├── data/
│   └── spotify_web_app_architecture.pdf   # knowledge base for the Assistant agent
├── requirements.txt
├── .env.example
├── .env                 # not committed -- your real keys go here
├── .gitignore
└── answers.txt          # generated at runtime, not committed
```

## Example

```
Query: How do I reset my Spotify password?

Answer 1 (Assistant): [grounded in the architecture knowledge base]
Answer 2 (Web Search Assistant): [grounded in live web search results]
```

Both answers, plus the original query, are saved to `answers.txt` and shown in the Streamlit UI.
