# app.py (top)
import sys
import asyncio

# On Windows, gRPC aio works with the selector event loop policy
if sys.platform.startswith("win"):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Ensure this thread has an event loop
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)


import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import  GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
load_dotenv()

st.set_page_config(page_title="Chat with PDF", page_icon=":books:")
st.title("RAG application built on Gemini model")

loader = PyPDFLoader("INTRODUCTION TO CNN.pdf")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

docs = text_splitter.split_documents(data)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vectorstore = Chroma.from_documents(documents=docs, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001") )

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k" :10} )

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",temperature=0.3, max_tokens=500)

query = st.text_input("Ask a question about the document")


system_prompt =(
    """You are an assistant for question-answering tasks
    use the following pieces of retrieved context to answer
    the question. If you don't know the answer say that you
    don't know. use three sentences maximum and keep the answer
    as concise as possible.
    \n\n
    {context}
    """
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
         ("human", "{input}")
    ]
)

if query:
    question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    response = rag_chain.invoke({"input": query})
    st.write("You asked: ", query)
    st.write("Answer: ", response["answer"])