from openai import OpenAI
import base64
import cv2
import json
from PIL import Image
import numpy as np
import os

import pandas as pd


# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "QWENVLPRICE" #"QWENVLPRICE"
openai_api_base = "http://117.50.186.193:8556/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def img_to_base64(img_path):
    with open(img_path, "rb") as f:
        img = f.read()
    return base64.b64encode(img).decode("utf-8")

models = client.models.list()
print("Models:", models)

url = "/datadrive/codes/frank/langchains/retrieval_anything/data/wendu/7300ACCE-9373-4C7B-883B-01CAA03536E0.png"
img_data = img_to_base64(url)
img_data = f"data:image/png;base64,{img_data}"

q_tempreture = """
            1.你需要提取图片上的温度数字,位于图片中黑色的显示屏上，数字是红色的，忽略非显示屏区域的文本
            2.先定位到显示屏区域，然后提取显示屏上的阿拉伯数字，一定要数字，而不是其他字符
            3.温度数字可能会有小数点，可能会有遮挡，需要仔细识别
            4.如果存在温度数字，输出{'temperature': 数字}
            5.如果没有温度数字，输出{'temperature': None}
            6.直接输出最终的结果Json，不要输出你的思考过程! 
            """


# chat_response = client.chat.completions.create(
#     model="Qwen/Qwen2.5-VL-7B-Instruct",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": img_data,
#                     },
#                 },
#                 {"type": "text", "text": q_tempreture},
#             ],
#         },
#     ],
# )

# print("Raw response:", chat_response)
# print("Response:", chat_response.choices[0].message.content)


def run_img_with_prompt(img_path, prompt):
    img_data = img_to_base64(img_path)
    img_data = f"data:image/png;base64,{img_data}"
    
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
                            "url": img_data,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ],
    )
    
    return chat_response.choices[0].message.content


def parse_response(response):
    try:
        json_res = json.loads(response)
    except json.JSONDecodeError:
        print("Failed to parse JSON response:", response)
        clean_resp = response.strip("```json\n")
        clean_resp = clean_resp.strip("```")
        json_res = json.loads(clean_resp)
    return json_res


def run_folder(img_folder:str):
    results = {}
    res_folder = os.path.join(img_folder, "results")
    os.makedirs(res_folder, exist_ok=True)
    for img_name in os.listdir(img_folder):
        if img_name.endswith(".jpg") or img_name.endswith(".png"):
            img_path = os.path.join(img_folder, img_name)
            response = run_img_with_prompt(img_path, q_tempreture)
            print("Response for {}: {}".format(img_name, response))
            # response = parse_response(response)
            img = cv2.imread(img_path)
            # write the image with the response
            cv2.putText(img, response, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imwrite(os.path.join(res_folder, f"{img_name}_result.jpg"), img)
            
    
    return results


if __name__ == "__main__":
    # Example usage
    img_path = "/datadrive/codes/frank/langchains/retrieval_anything/data/wendu/7300ACCE-9373-4C7B-883B-01CAA03536E0.png"
    prompt = q_tempreture
    response = run_img_with_prompt(img_path, prompt)
    print("Response:", response)
    
    # Run on a folder of images
    img_folder = "/datadrive/codes/frank/langchains/retrieval_anything/data/wendu"
    results = run_folder(img_folder)
    print("Results:", results)
    # Save results to a JSON file
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Results saved to results.json")