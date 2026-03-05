# 🏋️ Arnold AI – Personal Trainer Chatbot

Arnold AI is a Retrieval-Augmented Generation (RAG) chatbot that acts as a motivational personal trainer inspired by Arnold Schwarzenegger. The chatbot answers fitness questions, suggests workout exercises, and provides motivational coaching while grounding its responses in a fitness knowledge base.

The system demonstrates how modern AI applications can combine **knowledge retrieval, vector search, and large language models** to create an interactive assistant.

---

# 🚀 Features

- 💪 Arnold-inspired motivational coaching style  
- 🧠 Retrieval-Augmented Generation (RAG) architecture  
- 📚 Knowledge base built from fitness documents and quotes  
- 🔎 Vector similarity search using Azure Cosmos DB  
- 🤖 AI responses generated with Azure OpenAI  
- 🖥 Streamlit chat interface  

---

# 🧠 Architecture

The chatbot follows a **RAG pipeline**:

User Question  
↓  
Query Embedding (Azure OpenAI)  
↓  
Vector Search (Cosmos DB)  
↓  
Relevant Knowledge Retrieved  
↓  
Prompt Augmentation  
↓  
GPT-4.1 Response  

This approach improves response accuracy by grounding the model in retrieved context rather than relying only on the model’s internal knowledge.

---

# 🛠 Technologies Used

- **Python**
- **Streamlit** – chatbot interface  
- **Azure OpenAI** – embeddings and GPT-4.1 responses  
- **Azure Cosmos DB** – vector database for knowledge retrieval  
- **LangChain Text Splitter** – document chunking  
- **Azure AI Document Processing** – extracting text from documents  

---

# 📂  How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
