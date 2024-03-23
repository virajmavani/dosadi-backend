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
from langchain.agents import AgentExecutor, create_react_agent
from langchain.schema import Document
from langchain.tools.retriever import create_retriever_tool
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
        dataset = load_dataset("refugee-law-lab/canadian-legal-data")['train']
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
    template = """Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: Given the case description: "{caseDescription}" and intended case outcome: "{intendedOutcome}, create a legal strategy to use in court."
    Thought:{agent_scratchpad}
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
    tool = create_retriever_tool(
        retriever,
        "search_canadian_legal_data",
        "Searches and returns case law from the Canadian Legal Data dataset.",
    )
    tools = [tool]

    agent = create_react_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tool_names = ["search_canadian_legal_data"],
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True
    )

    agentsMap['agent'] = agent_executor
    
    yield
    print("Shutdown")
    # Clean up the ML models and release the resource


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
    response = agentsMap['agent'].invoke({"caseDescription": request.caseDescription, "intendedOutcome": request.intendedOutcome})
    return response