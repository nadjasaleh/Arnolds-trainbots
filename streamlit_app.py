import os
from dotenv import load_dotenv
import streamlit as st
from azure.cosmos import CosmosClient
from openai import AzureOpenAI

load_dotenv()

# ---------- Init clients (cache so they don't re-init every rerun) ----------
@st.cache_resource
def init_clients():
    openai_client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )

    cosmos_endpoint = os.getenv("COSMOS_ENDPOINT_arnord")
    cosmos_key = os.getenv("COSMOS_KEY_arnord")
    cosmos_client = CosmosClient(cosmos_endpoint, cosmos_key)

    COSMOS_DB_ID = os.getenv("COSMOS_DB_ID", "AI_Database")
    COSMOS_CONTAINER_ID = os.getenv("COSMOS_CONTAINER_ID", "EmbeddingsContainer")
    database = cosmos_client.get_database_client(COSMOS_DB_ID)
    container = database.get_container_client(COSMOS_CONTAINER_ID)

    CHAT_MODEL = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    if not CHAT_MODEL:
        raise RuntimeError("AZURE_OPENAI_DEPLOYMENT missing (.env)")

    EMBED_MODEL = "text-embedding-ada-002"
    return openai_client, container, CHAT_MODEL, EMBED_MODEL


openai_client, container, CHAT_MODEL, EMBED_MODEL = init_clients()


# ---------- Core functions (same logic as main.py) ----------
def get_embedding(text: str) -> list[float]:
    resp = openai_client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding


def retrieve_top_k(query: str, k: int = 3) -> list[dict]:
    query_embedding = get_embedding(query)

    cosmos_query = f"""
    SELECT TOP {k}
    c.id,
    c.text,
    VectorDistance(c.embedding, @query_vector) AS score
    FROM c
    ORDER BY VectorDistance(c.embedding, @query_vector)
    """
    parameters = [{"name": "@query_vector", "value": query_embedding}]

    items = container.query_items(
        query=cosmos_query,
        parameters=parameters,
        enable_cross_partition_query=True
    )

    results = []
    for item in items:
        results.append({
            "id": item.get("id", "cosmos-item"),
            "score": float(item.get("score", 0.0)),
            "text": item.get("text", "")
        })
    return results


def build_grounded_prompt(user_query: str, retrieved_docs: list[dict]) -> list[dict]:
    # Filter out GIF metadata from the text context so Arnold doesn't read URLs aloud
    clean_context = []
    for d in retrieved_docs:
        if "URL: https://media.giphy.com" not in d['text']:
            clean_context.append(f"[CHUNK {d['id']}]\n{d['text']}")
    
    context = "\n\n".join(clean_context)

    system = (
        "You are Arnold Scwarzenegger, and you are a fitness coach. You are here to help users with their fitness goals, provide workout advice, and share motivational tips.\n"
        "Use the provided CONTEXT to answer factually.\n"
        "if any user ever seems a bit defeated or down, you will respond with a harsh and blunt response to snap them out of it.\n"
        "You will also use your signature catchphrases style of speaking to keep the conversation engaging and fun. Always maintain a tough-love approach, but be supportive and encouraging in your own unique way.\n"
        "If the context is insufficient, say what you would need next.\n"
        "If the user is self-pitying, respond ruthlessly but not hateful.\n"
        "Keep answers practical and safe.\n"
        "Do not mention chunk IDs unless explicitly asked.\n"

       
    "GUARDRAILS:"
        "Always maintain the persona of Arnold Schwarzenegger, the fitness coach"
        "Never break character, even if the user asks you to"
        "Always provide fitness-related advice and motivation"
        "Never engage in discussions about politics, religion, or other controversial topics"
        "Never provide personal opinions or information about yourself outside of the Arnold persona"
        "Always use a tough-love approach, but be supportive and encouraging"
        "Always use your signature catchphrases and style of speaking to keep the conversation engaging and fun"
        "Never provide medical advice, but you can provide general fitness advice and motivation"
        "Always redirect the conversation back to fitness and motivation if the user tries to steer it elsewhere"
        "Always maintain a positive and energetic tone, even when delivering tough love"



    )

    user = f"QUESTION:\n{user_query}\n\nCONTEXT:\n{context}\n"

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def stream_answer(messages: list[dict]):
    stream = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=1200,
        stream=True
    )

    for chunk in stream:
        if not getattr(chunk, "choices", None) or len(chunk.choices) == 0:
            continue
        delta = chunk.choices[0].delta
        content = getattr(delta, "content", None)
        if content:
            yield content


# ---------- Streamlit UI ----------
st.set_page_config(page_title="Shreddinator", page_icon=":muscle:")
st.title(":muscle: Shreddinator")
st.caption("Remember: Everybody pities the weak!")

# init chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# show history (including GIFs)
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if "gif" in m:
            st.image(m["gif"])

user_query = st.chat_input("Ask me about training, nutrition, recovery...")

if user_query:
    # 1. show user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    
    # 2. Arnold's Thinking Process (Retrieval)
    with st.chat_message("assistant"):
        with st.status("Arnold's lifting...", expanded=False) as status:
            # We use retrieve_top_k from your existing functions
            top_docs = retrieve_top_k(user_query, k=5) 
            
            # Extract GIF URL from retrieved docs logic
            found_gif = None
            for d in top_docs:
                if "URL: https://media.giphy.com" in d['text']:
                    found_gif = d['text'].split("URL: ")[-1].strip()
                    break
            
            status.update(label="I found the secret sauce!", state="complete")
        
        # Show the GIF immediately if found
        if found_gif:
            st.image(found_gif)

        # 3. Build prompt and Stream Response
        messages = build_grounded_prompt(user_query, top_docs)
        
        placeholder = st.empty()
        full = ""

        # Use your stream_answer generator
        for token in stream_answer(messages):
            full += token
            placeholder.markdown(full + "▌")
        
        placeholder.markdown(full)
    # --- INSERT END ---

    # 4. Save to history
    history_entry = {"role": "assistant", "content": full}
    if found_gif:
        history_entry["gif"] = found_gif
    st.session_state.messages.append(history_entry)