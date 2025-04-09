from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI
import base64
from functools import partial

def img_to_base64(img_path):
    with open(img_path, "rb") as f:
        img = f.read()
    return base64.b64encode(img).decode("utf-8")


def construct_prompt(system_instruction: str, image_data: str, question: str, context: str = ""):
    messages = [
        SystemMessage(content=system_instruction),
        HumanMessage(content=[
            {"type": "text", "text": f"{question} {context}"}, 
            {"type": "image_url", "image_url": {"url": image_data}}
            ]),
    ]
    return messages


url = "/datadrive/codes/frank/langchains/retrieval_anything/data/wendu/559e0cb750546455854e8c8d57aa1898.jpg"
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
            

# Consturct the prompt directly
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


# Construct the prompt using partial function
partial_prompt = partial(construct_prompt, system_instruction="Help people with their tasks. You are a helpful assistant.")
msg1 = partial_prompt(image_data=img_data, question=q_tempreture)

resp = llm.invoke(msg1)
print("Raw response:", resp)

