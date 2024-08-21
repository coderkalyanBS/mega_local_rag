import streamlit as st
from dotenv import load_dotenv
import os
import validators
from pathlib import Path
import time

# Langchain imports
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.agents import create_sql_agent
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

load_dotenv()

# Set up Streamlit app
st.set_page_config(page_title="RAG Mega Project", page_icon="🦜", layout="wide")
st.title("🦜 RAG Mega Project")

# Sidebar for global settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")
os.environ['GROQ_API_KEY'] = api_key

# Main app navigation
app_mode = st.sidebar.selectbox("Choose the app mode", 
    ["Chat with PDF", "Web Search", "SQL Database Query", "URL/YouTube Summarizer"])

# Initialize LLM
@st.cache_resource
def get_llm():
    return ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192")

if api_key:
    llm = get_llm()

    if app_mode == "Chat with PDF":
        st.header("Chat with PDF")

        # PDF upload
        uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
        
        if uploaded_files:
            documents = []
            for uploaded_file in uploaded_files:
                temp_pdf = f"./temp_{uploaded_file.name}"
                with open(temp_pdf, "wb") as file:
                    file.write(uploaded_file.getvalue())
                
                loader = PyPDFLoader(temp_pdf)
                documents.extend(loader.load())
                os.remove(temp_pdf)

            # Process documents
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(documents)
            
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

            # Set up retriever
            retriever = vectorstore.as_retriever()

            # Set up prompt
            template = """Answer the question based only on the following context:
            {context}

            Question: {question}

            Answer: """
            prompt = ChatPromptTemplate.from_template(template)

            # Set up chain
            chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            # Chat input
            user_question = st.text_input("Ask a question about the PDFs:")
            if user_question:
                with st.spinner("Generating answer..."):
                    response = chain.invoke(user_question)
                    st.write(response)

    elif app_mode == "Web Search":
        st.header("Web Search and Chat")

        # Set up tools
        arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
        arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)
        wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
        wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)
        search = DuckDuckGoSearchRun(name="Search")

        tools = [search, arxiv, wiki]
        search_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)

        # Chat interface
        user_input = st.text_input("What would you like to know?")
        if user_input:
            with st.spinner("Searching..."):
                st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                response = search_agent.run(user_input, callbacks=[st_cb])
                st.write(response)

    elif app_mode == "SQL Database Query":
        st.header("SQL Database Query")

        # Database selection
        db_type = st.radio("Select database type:", ["SQLite", "MySQL"])

        if db_type == "SQLite":
            db_path = st.text_input("Enter the path to your SQLite database:")
            if db_path:
                db_uri = f"sqlite:///{db_path}"
        else:
            mysql_host = st.text_input("MySQL Host:")
            mysql_user = st.text_input("MySQL User:")
            mysql_password = st.text_input("MySQL Password:", type="password")
            mysql_db = st.text_input("MySQL Database:")
            if all([mysql_host, mysql_user, mysql_password, mysql_db]):
                db_uri = f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"
            else:
                st.warning("Please provide all MySQL connection details.")
                db_uri = None

        if db_uri:
            # Set up database connection
            db = SQLDatabase.from_uri(db_uri)
            toolkit = SQLDatabaseToolkit(db=db, llm=llm)

            agent = create_sql_agent(
                llm=llm,
                toolkit=toolkit,
                verbose=True,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
            )

            # Query interface
            user_query = st.text_input("Ask a question about your database:")
            if user_query:
                with st.spinner("Generating response..."):
                    response = agent.run(user_query)
                    st.write(response)

    elif app_mode == "URL/YouTube Summarizer":
        st.header("URL/YouTube Summarizer")

        url = st.text_input("Enter a URL (website or YouTube video):")

        if st.button("Summarize"):
            if not url:
                st.error("Please enter a URL.")
            elif not validators.url(url):
                st.error("Please enter a valid URL.")
            else:
                try:
                    with st.spinner("Generating summary..."):
                        if "youtube.com" in url or "youtu.be" in url:
                            loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
                        else:
                            loader = UnstructuredURLLoader(urls=[url], ssl_verify=False)
                        
                        docs = loader.load()
                        
                        prompt_template = """
                        Provide a summary of the following content in about 200 words:
                        Content: {text}
                        Summary:
                        """
                        prompt = ChatPromptTemplate.from_template(prompt_template)
                        
                        chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                        summary = chain.run(docs)
                        
                        st.subheader("Summary:")
                        st.write(summary)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

else:
    st.warning("Please enter your Groq API Key in the sidebar to get started.")