from openai import OpenAI
import base64
import cv2
import json


# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "QWENVLPRICE" #"QWENVLPRICE"
openai_api_base = "http://117.50.186.193:8556/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
print("Models:", models)

q_price_v1 = """
            1.提取输入图片的文本内容，有些文本可能会有遮挡，需要仔细识别。
            2.从文本中提取出价格，可能会有一个或多个价格。
            3.比较价格的大小，选择最小的那个价格
            4.输出最小的价格，以数字形式输出
            以json格式输出，格式为{'price': price_number}。
            直接输出最终的结果Json，不要输出你的思考过程! 
            """
q_price_v2 = """
            1.提取输入图片的文本内容，有些文本可能会有遮挡，需要仔细识别。
            2.从文本中提取出价格，可能会有一个或多个价格。
            3.比较价格的大小，选择最小的那个价格
            4.输出最小的价格，以数字形式输出
            5.如果没有价格，输出{'price': None}
            以json格式输出，格式为{'price': price_number}。
            直接输出最终的结果Json，不要输出你的思考过程! 
            """
            
question = """
            1.提取输入价签图片的文本内容，有些文本可能会有遮挡，需要仔细识别。
            2.你会看到一个等级参考列表：['H2', 'H1',None]
            3.判断图片上是否存在等级列表中的文本内容，文本必须完全匹配上，才是存在，否则是不存在
            以json格式输出，格式为{'level': {等级}}。
            直接输出最终的结果Json，不要输出你的思考过程! 
            """
            
q_instance = """
            1.你会看到一张图片，图片上有一些商品
            2.你需要输出图片上商品的数量，不要漏掉任何一个商品
            4.一步一步思考，反复检查你的结果，直到你确定你的结果是最合理的
            5.输出最终的结果Json，格式为{'count': {商品数量}}。
            """
            
q_volume = """
            1.You will see a picture of a retail product
            2.You should try your best to find the volume of the product in the picture
            3.The volume is usually expressed in milliliters (ml) or liters (L) or grams (g) or just a number
            4.Your answer should be in the format of {'volume': {volume_number}}
            """
            
q_store_closed = """
            1.你会看到一张图片，图片上有一个商店的门口
            2.你需要判断商店是否关闭
            3.如果商店关闭，输出{'closed': True}
            4.如果商店没有关闭，输出{'closed': False}
            5.如果图片上没有商店，输出{'closed': None}
            6.直接输出最终的结果Json，不要输出你的思考过程! 
            """

url = "/datadrive/codes/frank/langchains/retrieval_anything/data/186/302951b264a7dee9979681740cf96a9a.jpg"
img = cv2.imread(url)
img = cv2.resize(img, (100, 50))
img_b64 = base64.b64encode(cv2.imencode(".png", img)[1]).decode()
img_b64 = f"data:image/png;base64,{img_b64}"

for i in range(1):
        chat_response = client.chat.completions.create(
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": img_b64
                            },
                        },
                        {"type": "text", "text": q_store_closed},
                    ],
                },
            ],
        )
        
print("Raw response:", chat_response)
resp = chat_response.choices[0].message.content
print("Chat response:", resp, type(resp))