import yaml,os

with open("/Users/nikhil/Desktop/GenAI/Hands-on-practice/individual notebooks/chatgpt_key.yml","r") as file:
    api_creds = yaml.safe_load(file)

os.environ["OPENAI_API_KEY"] = api_creds['openai_key']


from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from operator import itemgetter
import streamlit as st
import tempfile
import os
import pandas as pd

# Customize initial app landing page
st.set_page_config(page_title="File QA Chatbot")
st.title("Welcome to File QA")

@st.cache_resource(ttl="1h")
# Takes uploaded PDFs, creates document chunks, computes embeddings
# Stores document chunks and embeddings in a Vector DB
# Returns a retriever which can look up the Vector DB
# to return documents based on user input
# Stores this in the cache
def configure_retriever(uploaded_files):
  # Read documents
  docs = []
  temp_dir = tempfile.TemporaryDirectory()
  for file in uploaded_files:
    temp_filepath = os.path.join(temp_dir.name, file.name)
    with open(temp_filepath, "wb") as f:
      f.write(file.getvalue())
    loader = PyMuPDFLoader(temp_filepath)
    docs.extend(loader.load())

  # Split into documents chunks
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500,
                                                 chunk_overlap=200)
  doc_chunks = text_splitter.split_documents(docs)

  # Create document embeddings and store in Vector DB
  embeddings_model = OpenAIEmbeddings()
  vectordb = Chroma.from_documents(doc_chunks, embeddings_model)

  # Define retriever object
  retriever = vectordb.as_retriever()
  return retriever

# Manages live updates to a Streamlit app's display by appending new text tokens
# to an existing text stream and rendering the updated text in Markdown
class StreamHandler(BaseCallbackHandler):
  def __init__(self, container, initial_text=""):
    self.container = container
    self.text = initial_text

  def on_llm_new_token(self, token: str, **kwargs) -> None:
    self.text += token
    self.container.markdown(self.text)

# Creates UI element to accept PDF uploads
uploaded_files = st.sidebar.file_uploader(
  label="Upload PDF files", type=["pdf"],
  accept_multiple_files=True
)
if not uploaded_files:
  st.info("Please upload PDF documents to continue.")
  st.stop()

# Create retriever object based on uploaded PDFs
retriever = configure_retriever(uploaded_files)

# Load a connection to ChatGPT LLM
chatgpt = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.1,
                     streaming=True)

# Create a prompt template for QA RAG System
qa_template = """
              Use only the following pieces of context to answer the question at the end.
              If you don't know the answer, just say that you don't know,
              don't try to make up an answer. Keep the answer as concise as possible.

              {context}

              Question: {question}
              """
qa_prompt = ChatPromptTemplate.from_template(qa_template)

# This function formats retrieved documents before sending to LLM
def format_docs(docs):
  return "\n\n".join([d.page_content for d in docs])

# Create a QA RAG System Chain
qa_rag_chain = (
  {
    "context": itemgetter("question") # based on the user question get context docs
      |
    retriever
      |
    format_docs,
    "question": itemgetter("question") # user question
  }
    |
  qa_prompt # prompt with above user question and context
    |
  chatgpt # above prompt is sent to the LLM for response
)


# Store conversation history in Streamlit session state
streamlit_msg_history = StreamlitChatMessageHistory(key="langchain_messages")

# Shows the first message when app starts
if len(streamlit_msg_history.messages) == 0:
  streamlit_msg_history.add_ai_message("Please ask your question?")

# Render current messages from StreamlitChatMessageHistory
for msg in streamlit_msg_history.messages:
  st.chat_message(msg.type).write(msg.content)

# Callback handler which does some post-processing on the LLM response
# Used to post the top 3 document sources used by the LLM in RAG response
class PostMessageHandler(BaseCallbackHandler):
  def __init__(self, msg: st.write):
    BaseCallbackHandler.__init__(self)
    self.msg = msg
    self.sources = []

  def on_retriever_end(self, documents, *, run_id, parent_run_id, **kwargs):
    source_ids = []
    for d in documents: # retrieved documents from retriever based on user query
      metadata = {
        "source": d.metadata["source"],
        "page": d.metadata["page"],
        "content": d.page_content[:200]
      }
      idx = (metadata["source"], metadata["page"])
      if idx not in source_ids: # store unique source documents
        source_ids.append(idx)
        self.sources.append(metadata)

  def on_llm_end(self, response, *, run_id, parent_run_id, **kwargs):
    if len(self.sources):
      st.markdown("__Sources:__ "+"\n")
      st.dataframe(data=pd.DataFrame(self.sources[:3]),
                    width=1000) # Top 3 sources


# If user inputs a new prompt, display it and show the response
if user_prompt := st.chat_input():
  st.chat_message("human").write(user_prompt)
  # This is where response from the LLM is shown
  with st.chat_message("ai"):
    # Initializing an empty data stream
    stream_handler = StreamHandler(st.empty())
    # UI element to write RAG sources after LLM response
    sources_container = st.write("")
    pm_handler = PostMessageHandler(sources_container)
    config = {"callbacks": [stream_handler, pm_handler]}
    # Get LLM response
    response = qa_rag_chain.invoke({"question": user_prompt},config)


#!streamlit run app.py --server.port=8989 &>./logs.txt &