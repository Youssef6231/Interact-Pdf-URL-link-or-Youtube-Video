import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
import faiss
import tempfile
import os
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.memory import ConversationBufferMemory

# Function to extract text from PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to generate vectorstore for similarity search
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore, text_chunks

# Save the vectorstore locally
def save_vectorstore_locally(vectorstore, file_path):
    faiss.write_index(vectorstore.index, file_path)

# Load vectorstore from a local file
def load_vectorstore_locally(file_path, text_chunks):
    index = faiss.read_index(file_path)
    docstore = InMemoryDocstore({str(i): Document(page_content=text) for i, text in enumerate(text_chunks)})
    embedding_function = OpenAIEmbeddings()  # Initialize the embedding function
    vectorstore = FAISS(embedding_function=embedding_function, index=index, docstore=docstore, index_to_docstore_id={i: str(i) for i in range(len(text_chunks))})
    return vectorstore

# Extract key points (summarization) from the vectorstore
def extract_keypoints_from_vectorstore(vectorstore, max_chunk_size=3000):
    llm = ChatOpenAI()
    retriever = vectorstore.as_retriever()
    documents = retriever.get_relevant_documents("summary")

    summarizer_chain = load_summarize_chain(llm, chain_type="stuff")
    summaries = []
    for i in range(0, len(documents), max_chunk_size):
        chunk_docs = documents[i:i + max_chunk_size]
        summary = summarizer_chain.run(chunk_docs)
        summaries.append(summary)

    if len(summaries) > 1:
        final_docs = [Document(page_content=summary) for summary in summaries]
        final_summary = summarizer_chain.run(final_docs)
    else:
        final_summary = summaries[0]

    # Organize the key points into a bullet-point list, each on a new line
    keypoints = final_summary.split('. ')
    keypoints = [f"• {point.strip()}" for point in keypoints if point.strip()]
    return "\n".join(keypoints)

# Function to create a conversation chain with the vectorstore
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(f"**User**: {message.content}")
        else:
            st.write(f"**Bot**: {message.content}")

def app():
    load_dotenv()

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "pdf_info" not in st.session_state:
        st.session_state.pdf_info = []

    col1, col2, col3 = st.columns([0.3, 0.02, 0.68])  # Unpack three columns

    with col1:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                for pdf in pdf_docs:
                    # Check if PDF is already processed
                    if any(info['name'] == pdf.name for info in st.session_state.pdf_info):
                        continue
                    
                    raw_text = get_pdf_text([pdf])
                    text_chunks = get_text_chunks(raw_text)

                    vectorstore, text_chunks = get_vectorstore(text_chunks)
                    
                    # Save vectorstore to a temporary file
                    temp_file_path = os.path.join(tempfile.gettempdir(), f"{pdf.name}.index")
                    save_vectorstore_locally(vectorstore, temp_file_path)

                    # Load the vectorstore back from the file
                    loaded_vectorstore = load_vectorstore_locally(temp_file_path, text_chunks)
                    
                    # Extract key points
                    keypoints = extract_keypoints_from_vectorstore(loaded_vectorstore)

                    st.session_state.pdf_info.append({
                        "name": pdf.name,
                        "keypoints": keypoints,
                        "vectorstore_path": temp_file_path
                    })

                    # Create conversation chain
                    st.session_state.conversation = get_conversation_chain(loaded_vectorstore)

        for pdf in st.session_state.pdf_info:
            st.write(f"**{pdf['name']}**")
            st.markdown(pdf['keypoints'].replace("\n", "\n\n"))  # Ensure each keypoint is on a new line

    # Vertical black line between columns
    with col2:
        st.markdown("<div style='border-left: 2px solid black; height: 100%;'></div>", unsafe_allow_html=True)

    with col3:  # Adjust content to the third column
        st.header("Chat with PDFs :books:")
        user_question = st.text_input("Ask a question about your documents:")
        if user_question:
            handle_userinput(user_question)

if __name__ == "__main__":
    app()
