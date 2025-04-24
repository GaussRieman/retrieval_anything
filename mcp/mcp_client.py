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
#     max_tokens=None,
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
    args=["/datadrive/codes/frank/langchains/retrieval_anything/mcp/markdown_server.py"],
)

markdown_eg1 = """
### 《洞见通用物体识别率的定义文档for宝洁_20200928》

#### **准确率的定义**

准确率：针对物体A的准确率，指的是对符合识别环境的实物画面进行识别，得到正确识别结果（目标结果为A，识别结果为A；目标结果为非A，识别结果为非A）的比例。

#### **训练数据集**

电动牙刷的正样本数据由全部由易现采集，一共三种牙刷和五张海报，其中牙刷为：黑色底座+黑色刷头，白色底座+白色刷头，粉色底座+白色刷头；海报由宝洁方提供。

负样本包含公开数据集以及易现采集的负样本数据集。

#### **识别物**

识别成果的结果为：

牙刷类目：包括所提供的三种牙刷以及五张海报。

识别物范围以训练数据集的涵盖内容为准。


#### **识别图像要求**

须为正常拍摄条件以及拍摄角度且对焦清晰的图像，识别图像中至少包含待识别物体正面的80％，且识别物部分屏占比不低于20％。

识别环境范围以训练数据集的内容为准。

#### **验收测试环节**

易现方会在项目测试结束后，提交测试报告。测试图片取自易现准备的测试集。测试集部分取自公开数据集中非参与训练的部分，以及一部分实际拍摄照片。易现可以提供该样本，并对选取范围拥有最终解释权。综合以上`训练数据集`、`识别物`、`识别环境`的情况，针对`测试图片`的准确率，确认`牙刷`类目的准确率不低于90%。   
验收方进行验收测试时，需要在考虑以上`训练数据集`、`识别物`、`识别环境`的条件下进行数据测试，并且易现方对测试数据可以有解释权。
"""


async def run_agent():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Get tools
            tools = await load_mcp_tools(session)
            print("tools: ", tools)

            # Create and run the agent
            agent = create_react_agent(llm, tools)
            agent_response = await agent.ainvoke({"messages": f"Take a look at this markdown report and give me a short summary of its structure and main takeaways. 'markdown': {markdown_eg1}"})
            print("Agent Response Length:", len(agent_response["messages"]))
            # Print the agent response
            for message in agent_response["messages"]:
                print(f"Agent: {message}")
                print("---\n")

        
        
if __name__ == "__main__":
    import asyncio
    asyncio.run(run_agent())