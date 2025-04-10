# Create server parameters for stdio connection
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI

openai_api_key = "QWENVLPRICE" #"QWENVLPRICE"
openai_api_base = "http://117.50.186.193:8556/v1"

# llm = ChatOpenAI(
#     model="Qwen/Qwen2.5-VL-7B-Instruct",
#     openai_api_key=openai_api_key,
#     openai_api_base=openai_api_base,
#     max_tokens=1024,
#     temperature=0,
# )

llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=1,
    # other params...
)


server_params = StdioServerParameters(
    command="python",
    # Make sure to update to the full absolute path to your math_server.py file
    args=["/datadrive/codes/frank/langchains/retrieval_anything/mcp/math_server.py"],
)

async def run_agent():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Get tools
            tools = await load_mcp_tools(session)

            # Create and run the agent
            agent = create_react_agent(llm, tools)
            agent_response = await agent.ainvoke({"messages": "what's (10 + 5) x 19?"})
            print(f"Agent Response: {agent_response}")

        
        
if __name__ == "__main__":
    import asyncio
    asyncio.run(run_agent())