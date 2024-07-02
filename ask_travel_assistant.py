import os
import openai
import logging
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Set OpenAI API key
openai.api_key = os.environ['OPENAI_API_KEY']

# Model and persistence configuration
LLM_NAME = "gpt-3.5-turbo"
PERSIST_DIRECTORY = 'docs/chroma/'

DATA_PATH = 'data/sorted_chat.txt'

# Initialize the embeddings
embedding_function = OpenAIEmbeddings()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_response(user_input: str, data_path: str=DATA_PATH) -> str:
    """
    Generate a response to a user input using a conversational retrieval chain.
    
    This function initializes the language model, embeddings, vector database, and memory,
    then uses these components to generate a response to the given user input.
    
    Args:
        user_input (str): The input from the user to generate a response for.
        data_path (str): The path to the document to be processed if the vector database does not exist.
    
    Returns:
        str: The generated response.
    """

    # Ensure the persistence directory exists
    if not os.path.exists(PERSIST_DIRECTORY):
        logger.info(f"Persistence directory '{PERSIST_DIRECTORY}' does not exist. Creating vector database.")
        create_vector_db(data_path, PERSIST_DIRECTORY)
    else:
        logger.info(f"Persistence directory '{PERSIST_DIRECTORY}' already exists.")

    # Initialize the language model with specified parameters
    llm = ChatOpenAI(model_name=LLM_NAME, temperature=0)

    logger.info(f"User input: {user_input}")


    # Load the vector database with persistence directory and embedding function
    vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_function)

    # Initialize the memory buffer for conversation history
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # Set the retriever with Maximal Marginal Relevance (MMR) search type
    retriever = vectordb.as_retriever(search_type="mmr")

    # Create a conversational retrieval chain
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory,
    )

    # Invoke the chain with the user input to get the result
    result = qa.invoke({"question": user_input})
    logger.info(f"Generated response: {result['answer']}")

    return result['answer']


def create_vector_db(data_path: str, persist_directory: str = PERSIST_DIRECTORY) -> None:
    """
    Create a vector database from a document, split into chunks, and save to disk.
    
    This function loads the document, splits it into chunks, and stores it in a 
    Chroma vector database, which is then saved to disk.
    
    Args:
        data_path (str): The path to the document to be processed.
        persist_directory (str): The directory to save the vector database.
    """
    

    # Load the document
    loader = TextLoader(data_path)
    documents = loader.load()
    logger.info(f"Loaded document from {data_path}.")

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # Load the chunks into Chroma and save to disk
    db = Chroma.from_documents(docs, embedding_function, persist_directory=persist_directory)
    logger.info(f"Created vector database and saved to {persist_directory}.")

    # Log the number of chunks created
    num_chunks = db._collection.count()
    logger.info(f"Number of chunks: {num_chunks}")


# Setting page title and headers
def set_titles_and_headers():
    st.set_page_config(page_title="Travel bot", page_icon="🏖️")
    st.markdown(
        "<h1 style='text-align: center;'>Travel assistant</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "Welcome to the AI Travel Assistant! 🏖️"
        ,
        unsafe_allow_html=True,
    )

    st.markdown(
        "I can answer your questions about the hotel",
        unsafe_allow_html=True,
    )
    # st.markdown(
    #     "Please describe your situation.",
    #     unsafe_allow_html=True,
    # )
