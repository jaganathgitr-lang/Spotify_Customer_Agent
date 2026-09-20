"""
Spotify Customer Support Crew -- CrewAI + Streamlit

Three agents run sequentially:
  1. Assistant            -- answers directly from a RAG knowledge base built over
                              data/spotify_web_app_architecture.pdf (FAISS + OpenAI embeddings)
  2. Web Search Assistant -- searches the web (Serper) and answers from the results
  3. Entry Agent          -- saves the query + both answers to answers.txt and
                              returns both answers to the user

All API keys are read from environment variables via .env -- never hard-coded.
"""

import os

import streamlit as st
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from crewai_tools import SerperDevTool
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

PDF_PATH = "data/spotify_web_app_architecture.pdf"
ANSWERS_FILE = "answers.txt"


# ---------------------------------------------------------------------------
# 1. Build the RAG vector store over the Spotify architecture PDF (FAISS)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Spotify knowledge base...")
def build_vector_db():
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_documents(chunks, embeddings)


# ---------------------------------------------------------------------------
# 2. Tools
# ---------------------------------------------------------------------------
def make_spotify_knowledge_tool(vector_db):
    @tool("Spotify Knowledge Base")
    def spotify_knowledge_tool(query: str) -> str:
        """Search Spotify's internal system/product architecture knowledge base
        (frontend, backend, data stores, streaming pipeline, etc.) to answer a
        customer's question about how the Spotify app and its systems work."""
        results = vector_db.similarity_search(query, k=4)
        return "\n\n".join(doc.page_content for doc in results)

    return spotify_knowledge_tool


_last_saved = {"key": None}


@tool("Save Support Ticket")
def save_support_ticket(query: str, answer_1: str, answer_2: str) -> str:
    """Save the customer's query and both agent answers to the answers.txt log
    file. Pass `query`, `answer_1` (the Assistant's answer), and `answer_2`
    (the Web Search Assistant's answer) exactly as given -- do not summarize,
    shorten, or reword either answer. Call this tool only ONCE per query."""
    # Agents occasionally re-call a tool before finalizing their answer
    # (a real, observed behavior, not just a theoretical risk) -- guard
    # against writing the exact same ticket twice in a row rather than
    # relying on prompt instructions alone.
    dedup_key = (query, answer_1, answer_2)
    if _last_saved["key"] == dedup_key:
        return f"Already saved this exact query and both answers to {ANSWERS_FILE} -- skipped duplicate write."

    with open(ANSWERS_FILE, "a", encoding="utf-8") as f:
        f.write(f"Query: {query}\n\n")
        f.write(f"Answer 1 (Assistant):\n{answer_1}\n\n")
        f.write(f"Answer 2 (Web Search Assistant):\n{answer_2}\n\n")
        f.write("-" * 60 + "\n\n")
    _last_saved["key"] = dedup_key
    return f"Saved the query and both answers to {ANSWERS_FILE}."


# ---------------------------------------------------------------------------
# 3. Agents + sequential crew
# ---------------------------------------------------------------------------
def build_crew(query: str, vector_db):
    llm = LLM(model="gpt-4o-mini")

    assistant_agent = Agent(
        role="Assistant",
        goal="Answer the customer's query directly, using the Spotify architecture knowledge base.",
        backstory=(
            "You are a Spotify customer support assistant. You answer questions about "
            "how Spotify's app and systems work using your knowledge base. If the "
            "knowledge base genuinely doesn't cover something, say so honestly rather "
            "than inventing details."
        ),
        tools=[make_spotify_knowledge_tool(vector_db)],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    web_search_agent = Agent(
        role="Web Search Assistant",
        goal="Search the web for the customer's query and produce an answer from the results.",
        backstory=(
            "You are skilled at researching current information online and summarizing "
            "it into a clear, accurate answer for a customer support context."
        ),
        tools=[SerperDevTool()],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    entry_agent = Agent(
        role="Entry Agent",
        goal="Save the customer's query and both prior answers to a file, then return both answers to the user.",
        backstory=(
            "You are meticulous about record-keeping for the support team. You always "
            "save the exact query and both answers to the support log using your tool, "
            "never summarizing or altering them, then present both answers clearly."
        ),
        tools=[save_support_ticket],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    task1 = Task(
        description=(
            f"Customer query: {query}\n\n"
            "Answer this directly using the Spotify Knowledge Base tool. Be clear and concise."
        ),
        expected_output="A direct answer to the customer's query, grounded in the Spotify knowledge base.",
        agent=assistant_agent,
    )

    task2 = Task(
        description=(
            f"Customer query: {query}\n\n"
            "Search the web for this query and produce an answer based on the search results."
        ),
        expected_output="An answer to the customer's query based on current web search results.",
        agent=web_search_agent,
    )

    task3 = Task(
        description=(
            f"The customer's original query was: {query}\n\n"
            "You have two prior answers available in your context: Answer 1 from the "
            "Assistant, and Answer 2 from the Web Search Assistant. Call the Save Support "
            "Ticket tool EXACTLY ONCE, with the exact query, answer_1, and answer_2 -- do "
            "not summarize, shorten, or reword either answer, and do not call the tool "
            "more than once. After saving, return both answers, clearly labeled "
            "'Answer 1' and 'Answer 2'."
        ),
        expected_output="Confirmation the file was saved, followed by both answers clearly labeled.",
        agent=entry_agent,
        context=[task1, task2],
    )

    crew = Crew(
        agents=[assistant_agent, web_search_agent, entry_agent],
        tasks=[task1, task2, task3],
        process=Process.sequential,
        verbose=True,
    )
    return crew, task1, task2


def run_support_crew(query: str, vector_db):
    crew, task1, task2 = build_crew(query, vector_db)
    entry_result = crew.kickoff()
    answer_1 = task1.output.raw if task1.output else ""
    answer_2 = task2.output.raw if task2.output else ""
    return answer_1, answer_2, entry_result.raw


# ---------------------------------------------------------------------------
# 4. Streamlit UI
# ---------------------------------------------------------------------------
def require_api_keys() -> None:
    missing = [k for k in ("OPENAI_API_KEY", "SERPER_API_KEY") if not os.environ.get(k) or "your-" in os.environ.get(k, "")]
    if missing:
        st.error(
            f"Missing/placeholder API key(s): {', '.join(missing)}. "
            "Add real values to your .env file and restart the app."
        )
        st.stop()


def main():
    st.set_page_config(page_title="Spotify Customer Support", page_icon="🎧")
    require_api_keys()

    st.title("🎧 Spotify Customer Support")
    st.caption(
        "Ask a question below. Three agents run in order: an Assistant grounded in "
        "Spotify's architecture knowledge base, a Web Search Assistant, and an Entry "
        "Agent that saves everything to answers.txt and returns both answers."
    )

    vector_db = build_vector_db()

    query = st.text_input("Your question or task", placeholder="e.g. How does Spotify stream audio to the app?")
    submitted = st.button("Submit", type="primary")

    if submitted:
        if not query.strip():
            st.warning("Please enter a question first.")
            st.stop()

        with st.spinner("Running the support crew (Assistant -> Web Search Assistant -> Entry Agent)..."):
            answer_1, answer_2, entry_output = run_support_crew(query.strip(), vector_db)

        st.success(f"Done -- saved to {ANSWERS_FILE}")

        st.subheader("Answer 1 -- Assistant (Spotify Knowledge Base)")
        st.write(answer_1)

        st.subheader("Answer 2 -- Web Search Assistant")
        st.write(answer_2)

        with st.expander("Entry Agent's final response"):
            st.write(entry_output)


if __name__ == "__main__":
    main()
