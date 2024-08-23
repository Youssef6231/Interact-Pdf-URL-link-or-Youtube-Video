import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
import faiss
import os
import tempfile
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.memory import ConversationBufferMemory

# Function to extract transcript from YouTube
def get_youtube_transcript(video_id):
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    text = " ".join([entry['text'] for entry in transcript])
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
    embedding_function = OpenAIEmbeddings()
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
    if "youtube_info" not in st.session_state:
        st.session_state.youtube_info = []

    col1, col2, col3 = st.columns([0.3, 0.02, 0.68])

    with col1:
        st.subheader("Your YouTube Videos")
        youtube_url = st.text_input("Enter YouTube video URL:")
        if st.button("Process"):
            with st.spinner("Processing"):
                video_id = youtube_url.split("v=")[-1]
                if any(info['id'] == video_id for info in st.session_state.youtube_info):
                    st.warning("This video is already processed.")
                else:
                    transcript = get_youtube_transcript(video_id)
                    text_chunks = get_text_chunks(transcript)

                    vectorstore, text_chunks = get_vectorstore(text_chunks)
                    
                    temp_file_path = os.path.join(tempfile.gettempdir(), f"{video_id}.index")
                    save_vectorstore_locally(vectorstore, temp_file_path)

                    loaded_vectorstore = load_vectorstore_locally(temp_file_path, text_chunks)
                    
                    keypoints = extract_keypoints_from_vectorstore(loaded_vectorstore)

                    st.session_state.youtube_info.append({
                        "id": video_id,
                        "keypoints": keypoints,
                        "vectorstore_path": temp_file_path
                    })

                    st.session_state.conversation = get_conversation_chain(loaded_vectorstore)

        for video in st.session_state.youtube_info:
            st.write(f"**Video ID: {video['id']}**")
            st.markdown(video['keypoints'].replace("\n", "\n\n"))

    with col2:
        st.markdown("<div style='border-left: 2px solid black; height: 100%;'></div>", unsafe_allow_html=True)

    with col3:
        st.header("Chat with YouTube Transcripts :movie_camera:")
        user_question = st.text_input("Ask a question about your video:")
        if user_question:
            handle_userinput(user_question)

if __name__ == "__main__":
    app()
