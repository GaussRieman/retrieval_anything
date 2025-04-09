from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI
import base64

def img_to_base64(img_path):
    with open(img_path, "rb") as f:
        img = f.read()
    return base64.b64encode(img).decode("utf-8")

url = "/datadrive/codes/frank/langchains/retrieval_anything/data/wendu/5dc5cb10030117245509f4f563f0bc98.jpg"
img_data = img_to_base64(url)
img_data = f"data:image/png;base64,{img_data}"


openai_api_key = "QWENVLPRICE" #"QWENVLPRICE"
openai_api_base = "http://117.50.186.193:8556/v1"

llm = ChatOpenAI(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    openai_api_key=openai_api_key,
    openai_api_base=openai_api_base,
    max_tokens=1024,
    temperature=0,
)

q_tempreture = """
            1.你需要提取图片上的温度数字,位于图片中黑色的显示屏上，数字是红色的，忽略非显示屏区域的文本
            2.先定位到显示屏区域，然后提取显示屏上的阿拉伯数字，一定要数字，而不是其他字符
            3.温度数字可能会有小数点，可能会有遮挡，需要仔细识别
            4.如果存在温度数字，输出{'temperature': 数字}
            5.如果没有温度数字，输出{'temperature': None}
            6.直接输出最终的结果Json，不要输出你的思考过程! 
            """
            

messages = [
    SystemMessage(
        content="Help people with their tasks. You are a helpful assistant."
    ),
    HumanMessage(
        content=[
            {"type": "text", "text": q_tempreture},
            {
                "type": "image_url",
                "image_url": {"url": img_data},
            },
        ],
    ),
]
res = llm.invoke(messages)
print("Raw response:", res)
print("Response:", res.content)
