import os
import openai
import chainlit as cl
from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
)
from oci_storage import OciStorage
from llama_index.llms.openai import OpenAI
from llama_index.llms.groq import Groq
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_parse import LlamaParse
from llama_index.core.callbacks import CallbackManager
from llama_index.core.service_context import ServiceContext
from llama_index.postprocessor.flag_embedding_reranker import (
    FlagEmbeddingReranker,
)
from llama_index.core.memory.chat_memory_buffer import ChatMemoryBuffer
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import GPTVectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_parse import LlamaParse
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.prompts import Prompt
from llama_index.core.callbacks import CallbackManager
from llama_index.core.service_context import ServiceContext
from dotenv import load_dotenv
import nest_asyncio
import chromadb
import logging
import sys
import os
import asyncio
import chainlit as cl

from fastapi.responses import JSONResponse
from chainlit.server import app
import jwt
from datetime import datetime, timedelta

nest_asyncio.apply()
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
load_dotenv()


def create_jwt(identifier: str) -> str:
    to_encode = {
        "identifier": identifier,
        "exp": datetime.now() + timedelta(minutes=60 * 24 * 15),
    }
    encoded_jwt = jwt.encode(
        to_encode, str(os.getenv("CHAINLIT_AUTH_SECRET")), algorithm="HS256"
    )
    return encoded_jwt


@app.get("/custom-auth")
async def custom_auth():
    token = create_jwt(identifier="Test User")
    return JSONResponse({"token": token})


openai.api_key = os.environ.get("OPENAI_API_KEY")

db = chromadb.PersistentClient(path="./chroma_db")


def createAgent(query_engine_tools):
    memory = ChatMemoryBuffer(token_limit=4096)
    # [Optional] Add Context
    context = """\
        You are a Policy Insurance Agent, who knows all about Vehicle Policy related stuff.\
        
        When asked for a single information about the customer like his name or his vehicle number, you should only provide that information. No Other text should be included with the answer.\
        
        You will answer the question by using the given tools. One of the tools contains information about the company's insurance policy \
            
        and another one contains information about one of the customer, which include bank letter, Tax Invoice, Vehicle information, etc..\
            
        You can use these tools to answer the questions as a veteran.\
            
        If You have given a name before, then Whenever the user will refer to himself by using "I" or "Me", you should interpret it that user wanted to know about himself.\
            
        Always try to use all the tools available to you to answer the questions.\
    """
    llm = Groq(model="llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY"), verbose=True)
    agent = ReActAgent.from_tools(
        query_engine_tools, llm=llm, verbose=True, context=context, memory=memory
    )

    return agent


def load_query_engine(index, context="", description=""):
    reranker = load_reranker()
    service_context = ServiceContext.from_defaults(
        callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()])
    )

    query_engine = index.as_query_engine(
        streaming=True,
        similarity_top_k=4,
        node_postprocessors=[reranker],
        service_context=service_context,
        vector_store_query_mode="hybrid",
    )

    # query_engine_tools = [
    #     QueryEngineTool(
    #         query_engine=query_engine,
    #         metadata=ToolMetadata(
    #             name=context,
    #             description=description,
    #         ),
    #     ),
    # ]

    # query_engine = SubQuestionQueryEngine.from_defaults(
    #     query_engine_tools=query_engine_tools,
    #     use_async=True,
    # )
    return query_engine


def check_vector_store(
    collectionName: str,
    parsing_instruction: str,
    data_folder: str = "./data/Disease.pdf",
):
    try:
        chroma_collection = db.get_collection(collectionName)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        # rebuild storage context
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # load index
        index = VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context
        )
        return index
    except Exception as e:
        print(f"Erorr: {e}")
        chroma_collection = db.get_or_create_collection(collectionName)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        parser = LlamaParse(
            api_key=os.getenv("LLAMA_CLOUD_API_KEY"),  # type: ignore
            result_type="markdown",  # type: ignore
            show_progress=True,
            language="en",  # type: ignore
            verbose=True,
            num_workers=2,
            parsing_instruction=parsing_instruction,
        )

        file_extractor = {".pdf": parser}
        documents = SimpleDirectoryReader(
            data_folder,
            file_extractor=file_extractor,  # type: ignore
        ).load_data(show_progress=True)
        node_parser = MarkdownElementNodeParser(
            llm=Groq(
                model="llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY"), verbose=True
            ),
            num_workers=4,
        )
        nodes = node_parser.get_nodes_from_documents(documents=documents)
        base_nodes, objects = node_parser.get_nodes_and_objects(nodes)

        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = GPTVectorStoreIndex(
            nodes=base_nodes + objects,
            storage_context=storage_context,
            verbose=True,
            use_async=True,
        )
        return index


def load_reranker():
    return FlagEmbeddingReranker(
        top_n=5,
        model="BAAI/bge-reranker-large",
    )


oci_storage = OciStorage()
if not os.path.exists(f"data/Official_Docs"):
    oci_storage.download_files(
        folder_name="Original",
        local_directory=f"data/Official_Docs",
    )
else:
    print("The Original Documents are available in the local directory.")

Settings.llm = Groq(
    model="llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY"), verbose=True
)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.context_window = 4096

parent_index = check_vector_store(
    collectionName="insurance",
    parsing_instruction="""
            The provided documents contains information regarding the car insurance policies. There are multple documents refering to "Amendments to Motor Vehicle Insurance Policy against Third Party Liability as per Insurance Authority Circular no 26 of 2020", "MOTOR VEHICLE TAKAFUL POLICY AGAINST LOSS & DAMAGE", "MOTOR VEHICLE TAKAFUL POLICY AGAINST THIRD PARTY LIABILITY" and some conditions and cluases.

            When a benefits/coverage/exclusion is descrived in the document ammend to it and add a text in the folowing benefits format (where coverage could be and exclusion):
            
            All the important data need to be included in the vector knowledge base.
            
            For {nameoftherisk} and in this condition {when and where does the converage Apply} the coverage is {coverageDescription}.
            """,
)


@cl.on_chat_start
async def start():
    cl.user_session.set("oci_storage", oci_storage)
    parent_query_engine = load_query_engine(
        parent_index,
        context="Policy_Tool",
        description="Provides information regarding the vehicle insurance policy. It also provides incentives and different amounts and other aspects related to any kind of accidents When asked anything related to the policies then use this tool Use a detailed plain text question as input to the tool.",
    )
    phoneNumber = await cl.AskUserMessage(
        author="Assistant", content="Hello, Can you please provide your phone number?"
    ).send()

    await cl.Message(
        author="Assistant",
        content="Thank You, Please wait while we fetch your data.",
    ).send()
    cl.user_session.set("phoneNumber", phoneNumber)

    if phoneNumber:
        folder_exist = oci_storage.check_folder_exist(phoneNumber["output"])
        if folder_exist:
            if not os.path.exists(f"data/{phoneNumber['output']}"):
                oci_storage.download_files(
                    folder_name=phoneNumber["output"],
                    local_directory=f"data/{phoneNumber['output']}",
                )
            sub_index = check_vector_store(
                collectionName=phoneNumber["output"],
                data_folder=f"data/{phoneNumber['output']}",
                parsing_instruction="""The provided documents contains information regarding the Person who has applied for insurance. It contains Bank Letter, Schedule of Details of the Insured Vehicle In the Takaful Policy Against Loss and  Damage,Schedule of Details of the Insured Vehicle In the Takaful Policy Against Third Party Liability,TAX INVOICE.
                
                Get all the data from the documents as every data in these documents are important and could be asked by the user later durin QA.
                
                Try to get all the data available in the given documents.""",
            )
            sub_query_engine = load_query_engine(
                sub_index,
                context="User_Tool",
                description="Provides information regarding the customer's vehicle and insurance related information. It contains all the information related to the customer's vehicle and insurance policy. Use a detailed plain text question as input to the tool.",
            )

            query_engine_tools = [
                QueryEngineTool(
                    query_engine=parent_query_engine,
                    metadata=ToolMetadata(
                        name="Policy_Tool",
                        description=(
                            "Provides information regarding the vehicle insurance policy."
                            "It also provides incentives and different amounts and other aspects related to any kind of accidents"
                            "When asked anything related to the policies then use this tool"
                            "Use a detailed plain text question as input to the tool."
                        ),
                    ),
                ),
                QueryEngineTool(
                    query_engine=sub_query_engine,
                    metadata=ToolMetadata(
                        name="User_Tool",
                        description=(
                            "Provides information regarding the customer's vehicle and insurance related information."
                            "It contains all the information related to the customer's vehicle and insurance policy."
                            "When asked for a particular information about the customer, provide only that information."
                            "Use a detailed plain text question as input to the tool."
                        ),
                    ),
                ),
            ]
            agent = createAgent(query_engine_tools)
            cl.user_session.set("agent", agent)
            Name = agent.chat(
                "Provide a welcome message with the name of the customer mentioned in it."
            )
            await cl.Message(
                author="Assistant",
                content=f"{Name}",
            ).send()


@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent")

    res = await cl.make_async(agent.chat)(message.content)

    await cl.Message(content=res, author="Assistant").send()
