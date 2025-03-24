
# AWS Bedrock Chat PDF Demo

This project demonstrates an AI-powered chatbot application that uses AWS Bedrock services for embedding, vector storage, and question-answering functionalities. The workspace consists of two main applications: **Admin** and **User**, both containerized using Docker and orchestrated via `docker-compose`.

---

## Project Structure

```
.gitignore
docker-compose.yml
sample_chat_question.txt
Admin/
    .env
    admin.py
    Dockerfile
    README.md
    requirements.txt
    run_command
    test.py
User/
    .env
    app.py
    Dockerfile
    requirements.txt
```

---

## Key Components

### 1. **Admin Application**
- **Purpose**: 
  - Allows uploading PDF files.
  - Splits text into chunks for processing.
  - Creates vector stores using AWS Bedrock embeddings.
  - Stores vector files in an S3 bucket for later retrieval.
- **Features**:
  - PDF file upload and text processing.
  - Integration with AWS Bedrock for embedding generation.
  - Vector store creation using FAISS.
  - S3 bucket integration for storing vector files.

### 2. **User Application**
- **Purpose**:
  - Provides a chatbot interface for querying the vector store.
  - Interacts with AI models like Claude, Llama3, and Nova Lite.
- **Features**:
  - Loads vector stores from S3.
  - Queries vector stores using AI models.
  - Displays responses in a user-friendly Streamlit interface.

---

## Features

### Admin Application
- Upload PDF files for processing.
- Split text into manageable chunks using `RecursiveCharacterTextSplitter`.
- Generate embeddings using AWS Bedrock.
- Create and store FAISS vector indexes in an S3 bucket.

### User Application
- Load vector indexes from S3.
- Query vector indexes using AI models:
  - **Claude**: General-purpose conversational AI.
  - **Llama3**: Instruction-tuned large language model.
  - **Nova Lite**: Lightweight conversational AI.
- Display chatbot responses in a Streamlit interface.

---

## Setup Instructions

### Prerequisites
- Docker and Docker Compose installed.
- AWS credentials with access to S3 and Bedrock services.

### Environment Variables
Both applications use `.env` files to store AWS credentials and bucket configuration. Example:
```env
BUCKET_NAME=accionlab-bedrock-chat-pdf-demo
AWS_ACCESS_KEY_ID=<your-access-key>
AWS_SECRET_ACCESS_KEY=<your-secret-key>
AWS_REGION=us-east-1
```

---

## Build and Run Applications

### Using Docker Compose
1. Build and start both applications:
   ```sh
   docker-compose up --build
   ```
2. Access the Admin application at `http://localhost:8083`.
3. Access the User application at `http://localhost:8084`.

### Running Individual Applications

#### Admin Application
1. Navigate to the `Admin` directory:
   ```sh
   cd Admin
   ```
2. Build the Docker image for the Admin application:
   ```sh
   docker build -t bedrock-pdf-reader-admin .
   ```
3. Run the Admin application container:
   ```sh
   docker run --env-file .env -p 8083:8083 -it bedrock-pdf-reader-admin
   ```
4. Access the Admin application at `http://localhost:8083`.

#### User Application
1. Navigate to the `User` directory:
   ```sh
   cd User
   ```
2. Build the Docker image for the User application:
   ```sh
   docker build -t bedrock-pdf-reader-user .
   ```
3. Run the User application container:
   ```sh
   docker run --env-file .env -p 8084:8084 -it bedrock-pdf-reader-user
   ```
4. Access the User application at `http://localhost:8084`.

---

## Dependencies

### Admin Application
- `streamlit`
- `pypdf`
- `langchain`
- `langchain_community`
- `faiss_cpu`
- `boto3`
- `python-dotenv`

### User Application
- `streamlit`
- `langchain`
- `langchain_community`
- `faiss_cpu`
- `boto3`
- `python-dotenv`
- `langchain_aws`

---

## Notes
- Ensure `.env` files are not exposed in version control as they contain sensitive AWS credentials.
- Use the `sample_chat_question.txt` file to test the chatbot's capabilities.

---

## Future Enhancements
- Add authentication for secure access.
- Improve error handling and logging.
- Extend support for additional AI models.
- Optimize vector store creation and retrieval processes.

---