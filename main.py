from contextlib import asynccontextmanager
from fastapi import FastAPI

agents = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        storage_context = StorageContext.from_defaults(
            persist_dir="./storage/events"
        )
        events_index = load_index_from_storage(storage_context)

        index_loaded = True
    except:
        index_loaded = False

    if not index_loaded:
    # load data
        events_doc = SimpleDirectoryReader(
            input_files=["./data/seattle_events.pdf"]
        ).load_data()

        # build index
        vector_store_events = MilvusVectorStore(dim=1536, collection_name="events", overwrite=True)
        storage_context_events = StorageContext.from_defaults(vector_store=vector_store_events)
        events_index = VectorStoreIndex.from_documents(events_doc, storage_context=storage_context_events)

        # persist index
        events_index.storage_context.persist(persist_dir="./storage/events")
    
    events_engine = events_index.as_query_engine(similarity_top_k=3)

    query_engine_tool = [
        QueryEngineTool(
            query_engine=events_engine,
            metadata=ToolMetadata(
                name="events_10k",
                description=(
                    "Provides information about Events takingg place around Seattle area in the month of March 2024. "
                    "Use a detailed plain text question as input to the tool."
                ),
            ),
        )
    ]
    
    llm = OpenAI(model="gpt-3.5-turbo-0613")

    agent = ReActAgent.from_tools(
        query_engine_tool,
        llm=llm,
        verbose=True,
        # context=context
    )
    print("i am in startup")

    agents["agent"]=agent
    print("i am after agent")
    yield
    print("Shutdown")
    #Clean up the ML models and release the resources
    del data

app = FastAPI(lifespan=lifespan)

class RequestBody(BaseModel):
    bio1: str
    bio2: str

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/plandate")
async def read_item(request: RequestBody):
    print("Starting new request...")
    response = agents["agent"].chat(f"Given the events happening in Seattle in March, plan a date for two people; one of whom like {request.bio1} and other likes {request.bio2}. Provide output in the format of a timeline for the date.")
    print(str(response))
    return response