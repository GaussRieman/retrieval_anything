from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_deepseek import ChatDeepSeek
from typing import Annotated

import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langgraph.graph.message import add_messages

from langgraph.checkpoint.memory import MemorySaver
from langchain.prompts import PromptTemplate, HumanMessagePromptTemplate


# llm = ChatOllama(
#     base_url = "117.50.186.193:11434",
#     model="qwq:32b-q8_0",
#     temperature=0,
#     max_tokens=256,
# )


llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=1,
    # other params...
)


embeddings = OllamaEmbeddings(
  base_url="117.50.186.193:11434",
  model="rjmalagon/gte-qwen2-7b-instruct:bf16")
# text = "This is a sample text."
# embedded_text = embeddings.embed_documents([text])
# print(embedded_text)

vector_store = InMemoryVectorStore(embeddings)



# Load and chunk contents of the blog
# loader = WebBaseLoader(
#     web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
#     bs_kwargs=dict(
#         parse_only=bs4.SoupStrainer(
#             class_=("post-content", "post-title", "post-header")
#         )
#     ),
# )

# docs = loader.load()

# LOAD pdfs
from langchain_community.document_loaders import PyPDFLoader

file_path = "/datadrive/codes/frank/langchains/rag/baoxiao.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
all_splits = text_splitter.split_documents(docs)
print("all_splits: ", len(all_splits))

# Index chunks
# print("vector_store: ", vector_store.embedding)
# texts = [doc.page_content for doc in all_splits]
_ = vector_store.add_documents(documents=all_splits)

# Define prompt for question-answering
# prompt = hub.pull("rlm/rag-prompt")
# print("prompt: ", prompt)

input_variables = ['context', 'question', 'messages']
input_types = {}
partial_variables = {}
template = """
You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context and conversation history to answer the question. 
If you don't know the answer, just say that you don't know. 
Keep the answer concise.

Conversation History:
{messages}

Context: {context}

Question: {question}

Answer:
"""

prompt=PromptTemplate(input_variables=input_variables, 
                      input_types=input_types, 
                      partial_variables=partial_variables, 
                      template=template)
print("prompt: ", prompt)

# 在 prompt.invoke 调用之前，格式化 messages
def format_messages(messages):
    formatted_messages = ""
    for message in messages:
        formatted_messages += f"{message['role']}: {message['content']}\n"
    return formatted_messages


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    messages: Annotated[list, add_messages]


def format_messages(messages):
    formatted_messages = ""
    for message in messages:
        formatted_messages += f"{message['role']}: {message['content']}\n"
    return formatted_messages


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content, "messages": state["messages"]})
    response = llm.invoke(messages)
    return {"answer": response.content, "messages": [{"role": "assistant", "content": response.content}]}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# draw and save the graph
try:
    # draw the graph
    png_data = graph.get_graph().draw_mermaid_png()
    
    # save the image to a file
    with open("graph.png", "wb") as f:
        f.write(png_data)
    print("saved to graph.png")
except Exception as e:
    print(f"Error when saving: {e}")


config = {"configurable": {"thread_id": "1"}}

def stream_graph_updates(user_input: str):
    events = graph.stream(
        {"question": user_input, "messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values"
    )
    last_event = None  # 初始化 last_event 为 None

    for event in events:
        last_event = event  # 更新 last_event 为当前事件

    if last_event is not None:  # 如果有事件，则处理最后一个事件
        if "answer" in last_event:
            print("Assistant: ", last_event["answer"])
        else:
            print("Last event does not contain 'answer'")
    else:
        print("No events received")


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break