import boto3
import streamlit as st
import os
import uuid
import time

from dotenv import load_dotenv
# Load environment variables from .env
load_dotenv()  

## s3 client
s3_client = boto3.client("s3")
BUCKET_NAME = os.getenv("BUCKET_NAME")
AWS_REGION = os.getenv("AWS_REGION")


aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION", "us-east-1")  # Default region if missing


session = boto3.Session()
credentials = session.get_credentials()

if credentials:
    print("AWS Access Key ID:", credentials.access_key)
    print("AWS Secret Access Key:", credentials.secret_key)
else:
    print("No credentials found!")

print(os.environ)



## Bedrock
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_aws.llms.bedrock import BedrockLLM
from langchain.llms import Bedrock
from langchain_aws import ChatBedrockConverse

## prompt and chain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, SystemMessage

## Text splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

## Pdf Loader
from langchain_community.document_loaders import PyPDFLoader

## import FAISS
from langchain_community.vectorstores import FAISS

#bedrock_client = boto3.client(service_name="bedrock-runtime",region_name=AWS_REGION)
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=aws_region,
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key
)

bedrock_embedding = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)


def get_unique_id():
    return str(uuid.uuid4())



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

folder_path = "/tmp/"
## load index from s3
def load_index():
    s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.faiss",Filename=f"{folder_path}my_faiss.faiss" )
    s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.pkl",Filename=f"{folder_path}my_faiss.pkl" )


## get llm
MAX_TOKENS = 1000
def get_cloude_llm():
    llm = BedrockLLM(model_id="anthropic.claude-v2:1", client=bedrock_client, model_kwargs={'max_tokens_to_sample':MAX_TOKENS})
    return llm

def get_llama3_llm():
    llm = BedrockLLM(model_id="meta.llama3-70b-instruct-v1:0", client=bedrock_client, model_kwargs={'max_gen_len':MAX_TOKENS}, guardrails={"guardrailIdentifier": "56beqsjxbw5f", "guardrailVersion": "1"})
    return llm

def format_query(question, context):
    """Formats the prompt for Nova Pro chat model."""
    return [
        SystemMessage(content="You are a helpful AI assistant."),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion:\n{question}")
    ]


def get_nova_pro_llm(vectorstore, question, response_placeholder):
    #llm = Bedrock(model_id="amazon.nova-pro-v1:0", client=bedrock_client, model_kwargs={'max_new_tokens':MAX_TOKENS})
    llm = ChatBedrockConverse(model="us.amazon.nova-lite-v1:0")
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
        return_source_documents=True
    )

    docs = vectorstore.similarity_search(question, k=5)
    context = "\n".join([doc.page_content for doc in docs])
    #messages = format_query(question, context)
     # Instead of sending a list, format the input as a string
    formatted_query = f"Context:\n{context}\n\nQuestion:\n{question}"
    
    response = qa.invoke({"chat_history": [], "question": formatted_query})
    response_placeholder.write(response["answer"])


def get_response(llm,vectorstore, question, response_placeholder ):
    ## create prompt / template
    prompt_template = """

    Human: Please use the given context to provide concise answer to the question
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    <context>
    {context}
    </context>

    Question: {question}

    Assistant:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer=qa({"query":question})
    if 'result' in answer:
        result = answer['result']
        if "I do not have enough context" in result or "I apologize" in result or "I don't know the answer to that question" in result:
            response_placeholder.write("The model could not find relevant information to answer your question.")
        else:
            # Simulate streaming by updating the response incrementally
            for i in range(0, len(result), 50):
                response_placeholder.write(result[:i+50])
                time.sleep(0.1)  # Simulate delay
    else:
        response_placeholder.write("No valid response from the model.")

## main method
def main(): 
    st.set_page_config("AI-Powered Q&A Chatbot")
    st.header("AI-Powered Q&A Chatbot")

    load_index()

     # sidebar
    with st.sidebar:
        st.title("Vector store:")
        dir_list = os.listdir(folder_path)
        st.write(f"Files and Directory : {folder_path}")
        st.write(dir_list)

    #dir_list = os.listdir(folder_path)
    #st.write(f"Files and Directory : {folder_path}")
    #st.write(dir_list)

    ## create_index 
    faiss_index = FAISS.load_local(
        index_name="my_faiss", 
        folder_path=folder_path, 
        embeddings=bedrock_embedding,
        allow_dangerous_deserialization=True
    )

    #st.write("Index is ready")
    question = st.text_input("Please ask your question")
    
    response_placeholder = st.empty()

    if st.button("Ask Question"):
        with st.spinner("Fetching Info ....."):
        
            llm = get_cloude_llm()
            # get_response
            get_response(llm, faiss_index, question,response_placeholder)           
            st.success("Done")

    if st.button("Ask Question From LLAMA3"):
        with st.spinner("Fetching Info ....."):
        
            llm = get_llama3_llm()
            # get_response
            get_response(llm, faiss_index, question,response_placeholder)           
            st.success("Done")

    if st.button("Ask Question From Nova Lite"):
        with st.spinner("Fetching Info ....."):
        
            get_nova_pro_llm(faiss_index, question,response_placeholder)
            # get_response
            #get_response(llm, faiss_index, question,response_placeholder)           
            st.success("Done")                  
   
if __name__ == "__main__":
    main()