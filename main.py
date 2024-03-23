from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.prompts import PromptTemplate
from langchain_community.llms.octoai_endpoint import OctoAIEndpoint
from langchain_community.embeddings import OctoAIEmbeddings
from langchain_community.vectorstores import Milvus
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from pymilvus import connections, utility
from datasets import load_dataset
from pydantic import BaseModel

from dotenv import load_dotenv
import os

load_dotenv()
os.environ["OCTOAI_API_TOKEN"] = os.getenv("OCTOAI_API_TOKEN")
milvus_uri = os.getenv("MILVUS_ENDPOINT")
token = os.getenv("MILVUS_KEY")
user = os.getenv("MILVUS_USER")
password = os.getenv("MILVUS_PASSWORD")

agentsMap = {}

origins = [
    "http://localhost:3000",
    "http://localhost:8080",
]

async def initializeVectorStore():
    embeddings = OctoAIEmbeddings(endpoint_url="https://text.octoai.run/v1/embeddings")
    connections.connect("default",
                        uri=milvus_uri,
                        token=token,
                        user=user,
                        password=password)
    print(f"Connecting to DB: {milvus_uri}")
    collection_name = "canadian_legal_data"
    
    # Check if the collection exists
    check_collection = utility.has_collection(collection_name)
    if not check_collection:
        # Load only the first file from the dataset for analysis
        dataset = load_dataset("refugee-law-lab/canadian-legal-data")['train'][0]
        file_text = dataset['unofficial_text']
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=512, chunk_overlap=64,
        )
        texts = text_splitter.split_text(file_text)
        file_texts = [Document(page_content=chunked_text,
                               metadata={"doc_title": dataset["citation"], "chunk_num": i})
                      for i, chunked_text in enumerate(texts)]
        
        # Create the Milvus vector store
        vector_store = Milvus.from_documents(
            file_texts,
            embedding=embeddings,
            connection_args={"uri": milvus_uri, "token": token},
            collection_name=collection_name
        )
    else:
        print("Collection already exists.")
        vector_store = Milvus(
            embeddings,
            connection_args={"uri": milvus_uri, "token": token},
            collection_name=collection_name,
        )

    return vector_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = PromptTemplate.from_template(template)

    vector_store = await initializeVectorStore()

    llm = OctoAIEndpoint(
        endpoint_url="https://text.octoai.run/v1/chat/completions",
        model_kwargs={
            "model": "mixtral-8x7b-instruct-fp16",
            "max_tokens": 128,
            "presence_penalty": 0,
            "temperature": 0.01,
            "top_p": 0.9,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Keep your responses limited to one short paragraph if possible.",
                },
            ],
        },
    )

    retriever = vector_store.as_retriever()
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    agentsMap['chain'] = chain
    yield
    print("Shutdown")
    #Clean up the ML models and release the resource

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RequestBody(BaseModel):
    firstName: str
    lastName: str
    caseDescription: str
    intendedOutcome: str

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/callDosadi")
async def read_item(request: RequestBody):
    print("Starting new request...")
    print(f"{request.firstName}, {request.lastName}, {request.caseDescription}, {request.intendedOutcome}")
    response = agentsMap["chain"].invoke(f"How big is the city of Boston?")
    return response