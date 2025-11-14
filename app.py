import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Ollama
from processing import process_uploaded_file, add_to_vector_store, vector_store, retrieve_from_vector_store


@st.cache_resource
def load_rag_chain():
    retriever = retrieve_from_vector_store()
    llm = Ollama(model="llama3:8b")

    prompt_template = """
    Answer the user's question based only on the provided context.
    Cite the source file from the metadata.

    Context:
    {context}

    Question:
    {question}
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    def format_docs(docs):
        return "\n\n".join(f"Source: {doc.metadata['source']}\nContent: {doc.page_content}" for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

rag_chain = load_rag_chain()


# UI
st.title("Multimodal RAG MVP")
st.write("Upload files and chat with your data!")

# Upload Section
uploaded_file = st.file_uploader("Upload image/audio/pdf", type=["png", "jpg", "pdf", "wav"])
if uploaded_file:
    with st.spinner("Processing your file..."):
        data_dict, filename = process_uploaded_file(uploaded_file)
        add_to_vector_store(data_dict, filename)
    st.success(f"Added '{filename}' to FAISS vector store!")

st.divider()

# Chat Section
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_question := st.chat_input("What's up?"):
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke(user_question)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})