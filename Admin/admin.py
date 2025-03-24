import boto3
import streamlit as st
import os
import uuid

from dotenv import load_dotenv
# Load environment variables from .env
load_dotenv()  

## s3 client
s3_client = boto3.client("s3")
BUCKET_NAME = os.getenv("BUCKET_NAME")
AWS_REGION = os.getenv("AWS_REGION")

aws_region = os.getenv("AWS_REGION", "us-east-1")  # Default region if missing

session = boto3.Session()
credentials = session.get_credentials()

## Bedrock
from langchain_community.embeddings import BedrockEmbeddings

## Text splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

## Pdf Loader
from langchain_community.document_loaders import PyPDFLoader

## import FAISS
from langchain_community.vectorstores import FAISS

bedrock_client = boto3.client(service_name="bedrock-runtime",region_name=AWS_REGION)

bedrock_embedding = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)


def get_unique_id():
    return str(uuid.uuid4())

## Split the pages / text into chunks
def split_text(pages, chunk_size, chunk_overlap):
    text_spliter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_spliter.split_documents(pages)
    return docs

## create vectore store
def create_vector_store(request_id, documents):
    vectorstore_faiss = FAISS.from_documents(documents, bedrock_embedding)
    file_name = f"{request_id}.bin"
    folder_path = "/tmp/"
    vectorstore_faiss.save_local(index_name=file_name, folder_path=folder_path)

    ## upload to s3
    s3_client.upload_file(Filename=folder_path + "/" + file_name + ".faiss", Bucket=BUCKET_NAME, Key="my_faiss.faiss")
    s3_client.upload_file(Filename=folder_path + "/" + file_name + ".pkl", Bucket=BUCKET_NAME, Key="my_faiss.pkl")

    return True 

## main method
def main(): 
    st.write("This is Admin site for chat with PDF demo")
    uploaded_file = st.file_uploader("Choose a file","pdf")
    if uploaded_file is not None:
        request_uid = get_unique_id()
        st.write(f"Request ID: {request_uid}")
        saved_file_name = f"{request_uid}.pdf"
        with open(saved_file_name, mode="wb") as w:
            w.write(uploaded_file.getvalue())
        
        loader = PyPDFLoader(saved_file_name)
        pages = loader.load_and_split()
        pages_count = len(pages)

        st.write(f"Total pages: {pages_count}")
        st.write("Creating the Vector store")

        ## split text
        splitted_docs = split_text(pages,1000,200)
        st.write(f"Splitted Docs Length: {len(splitted_docs)}")
        st.write(f"============================")
        st.write(splitted_docs[0])
        st.write(f"============================")
        st.write(splitted_docs[1])

        st.write("Creating the Vector store")

        result = create_vector_store(request_uid, splitted_docs)

        if result:
            st.write("Hurray! Document process successfully")
        else:
            st.write("Error!! please check the logs.")

if __name__ == "__main__":
    main()