import streamlit as st
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.docstore.document import Document
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()

def load_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup.get_text()

def get_vectorstore_from_urls(urls):
    documents = []
    for url in urls:
        content = load_content(url)
        document = Document(page_content=content, metadata={'source': url})
        text_splitter = RecursiveCharacterTextSplitter()
        document_chunks = text_splitter.split_documents([document])
        documents.extend(document_chunks)
    
    if documents:  # Ensure there are documents before creating the vector store
        vector_store = Chroma.from_documents(documents, OpenAIEmbeddings())
        return vector_store
    return None

def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

def get_conversational_rag_chain(retriever_chain): 
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based *only* on the information from the provided website content. Do not use any outside knowledge."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("system", "Use the following context to answer the question: {context}")
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    return response['answer']

def app():
    st.title("Chat with a Website 🌐")

    # Initialize session state
    if "chat_history" not in st.session_state or st.session_state.chat_history is None:
        st.session_state.chat_history = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    urls = st.text_area("Enter the URLs (one per line) to chat with:")
    if st.button("Load Content"):
        url_list = [url.strip() for url in urls.splitlines() if url.strip()]
        st.session_state.vector_store = get_vectorstore_from_urls(url_list)
        if st.session_state.vector_store:
            st.success("Content loaded successfully!")
        else:
            st.warning("No valid content could be loaded from the provided URLs.")

    user_query = st.chat_input("Type your message here...")
    if user_query and st.session_state.vector_store:
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    if st.session_state.chat_history:  # Check if chat_history is initialized
        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.write(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.write(message.content)

if __name__ == "__main__":
    app()
