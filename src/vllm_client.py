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
            
q_grouding_task = """
            grounding task: pricetag, in normalized (0,1) coordinates
            output the answer in json format.
            Output the box ONLY.
            """
            
q_box_task = """
            描述: 图中<box>(207,225),(244,381)</box>区域，不要涉及到其他区域    
        """

"""
OCR 图中<box>(356,1290),(653,1442)</box>的文本 5.8
OCR 图中<box>(220,1191),(360,1442)</box>的文本 12.5
"""
q_box_price = """
            输入一个Box区域：<box>(32,650),(331,878)</box>
            1.OCR 得到区域文本内容
            2.根据位置和文本内容，找到对应的价签，如果价签上的文本和商品不一致，则忽略
            3.提取价签上的价格，可能会有一个或多个价格，输出最低价格
            输出{'price': price_number, "product_full_text": product_full_text, "pricetag_full_text": pricetag_full_text}
            如果没有找到价格，输出{'price': -1, "product_full_text": product_full_text, "pricetag_full_text": pricetag_full_text}
            直接输出最终的结果Json，不要输出你的思考过程!
            """


q_price_v3 = f"""
            Box1: <box>(72,1190),(218,1437)</box>
            Box2: <box>(356,1290),(653,1442)</box>
            Box3: <box>(520,56),(996,243)</box>
            Box4: <box>(86,5),(510,254)</box>
            从价签中提取这商品对应的价格，综合考虑位置和文本相关性。
            如果一个商品有多个价格，则选择最小的价格
            如果没有找到价格，则价格输出-1！
            '{{"box_id": box_id, "price": price}}'
            只按顺序输出给定的商品的信息，不要输出其它内容。
            """


unit_url = "/datadrive/codes/frank/langchains/retrieval_anything/data/price/fe6bedaacb3e42c1683260b23669b194_resize.jpg"
unit_img = cv2.imread(unit_url)
# img = cv2.resize(unit_img, (1000, 2000))
# cv2.imwrite(f"{unit_url[:-4]}_resize.jpg", unit_img) 
unit_img_b64 = base64.b64encode(cv2.imencode(".png", unit_img)[1]).decode()
unit_img_b64 = f"data:image/png;base64,{unit_img_b64}"

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
                        "url": unit_img_b64
                    },
                },
                {
                    "type": "text", 
                    "text": q_price_v3
                },
            ],
        },
    ],
)
        
print("Raw response:", chat_response)
resp = chat_response.choices[0].message.content
print("Chat response:", resp, type(resp))
