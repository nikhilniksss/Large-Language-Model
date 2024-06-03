import yaml,os

with open("/Users/nikhil/Desktop/GenAI/Hands-on-practice/individual notebooks/chatgpt_key.yml","r") as file:
    api_creds = yaml.safe_load(file)

os.environ["OPENAI_API_KEY"] = api_creds['openai_key']

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
from langchain.schema.runnable.config import RunnableConfig
from langchain.schema import StrOutputParser
from operator import itemgetter
import chainlit as cl

@cl.on_chat_start
# this function is called when the app is starting for the first time.

async def when_chat_starts():
    
    # Load a connection to chat gpt
    chatgpt = ChatOpenAI(model_name = "gpt-3.5-turbo",temperature=0.3,streaming=True)

    SYS_PROMPT = """
                 Act as a helpful assistant and answer question to the best of your ability.
                 Do not make up answers.
                """
    
    # create aprompt tempplate for langchain to use history to give answer
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",SYS_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human","{input}"),
        ]
    )

    # create memory object to store our conversation history
    memory = ConversationBufferMemory(k = 25,return_messages=True)

    #create conversation chain
    conversation_chain = (
        RunnablePassthrough.assign(
            history = RunnableLambda(memory.load_memory_variables)
            |
            itemgetter("history")
        )
        |
        prompt
        |
        chatgpt
        |
        StrOutputParser() # Parse output and show it on UI
    )
    #set session variable to access when user enter prompt
    cl.user_session.set("chain",conversation_chain)
    cl.user_session.set("memory",memory)

@cl.on_message
# This fuction is called when user send prompt message in app

async def on_user_message(message:cl.Message):
    
    # get the chain and memory into the session
    chain = cl.user_session.get("chain")
    memory = cl.user_session.get("memory")

    # this will store the response from chatgpt
    chatgpt_message = cl.Message(content="")

    # Stream the response from chat gpt and show in real time
    async for chunk in chain.astream(
        {"input":message.content},
        config = RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await chatgpt_message.stream_token(chunk)
    # finish displaying full message from chatgpt
    await chatgpt_message.send()

    # store current conversation into memory hisory

    memory.save_context({"input":message.content},
                        {"output":chatgpt_message.content})
    





