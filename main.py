from contextlib import asynccontextmanager
from fastapi import FastAPI
from langchain.prompts import PromptTemplate
from langchain_community.llms.octoai_endpoint import OctoAIEndpoint
from langchain_community.embeddings import OctoAIEmbeddings
from langchain_community.vectorstores import Milvus
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from pymilvus import connections, utility

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

async def initializeVectorStore():
    embeddings = OctoAIEmbeddings(endpoint_url="https://text.octoai.run/v1/embeddings")
    connections.connect("default",
                        uri=milvus_uri,
                        token=token,
                        user=user,
                        password=password)
    print(f"Connecting to DB: {milvus_uri}")
    collection_name = "cities"
    check_collection = utility.has_collection(collection_name)
    if check_collection:
        print(f"Collection Available!")
        vector_store = Milvus(
            embeddings,
            connection_args={"uri": milvus_uri, "token": token},
            collection_name=collection_name,
        )
    else:
        files = os.listdir("./data")
        file_texts = []

        for file in files:
            with open(f"./data/{file}") as f:
                file_text = f.read()
            text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=512, chunk_overlap=64, 
            )
            texts = text_splitter.split_text(file_text)
            for i, chunked_text in enumerate(texts):
                file_texts.append(Document(page_content=chunked_text, 
                        metadata={"doc_title": file.split(".")[0], "chunk_num": i}))
        
        vector_store = Milvus.from_documents(
            file_texts,
            embedding=embeddings,
            connection_args={"uri": milvus_uri, "token": token},
            collection_name="cities"
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

class RequestBody(BaseModel):
    bio1: str
    bio2: str

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/")
async def read_item(request: RequestBody):
    print("Starting new request...")
    response = agentsMap["chain"].invoke(f"How big is the city of Boston?")
    print(str(response))
    return response