# use gradio to build a demo, input a image and a question, and output the answer
import gradio as gr
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI
import base64
import requests
import cv2



openai_api_key = "QWENVLPRICE" #"QWENVLPRICE"
openai_api_base = "http://117.50.186.193:8556/v1"

llm = ChatOpenAI(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    openai_api_key=openai_api_key,
    openai_api_base=openai_api_base,
    max_tokens=1024,
    temperature=0,
)


def img_to_base64(img_path):
    img = cv2.imread(img_path)
    while max(img.shape) > 1024:
        scale = 0.5
        img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
    
    print("img.shape:", img.shape)
    _, img_encoded = cv2.imencode('.png', img)
    img_b64 = base64.b64encode(img_encoded).decode()
    img_b64 = f"data:image/png;base64,{img_b64}"
    return img_b64
    
def construct_prompt(image_url: str, 
                     question: str, 
                     context: str = "",
                     system_instruction: str = "You are a helpful and patient assistant, you help people with anything."):
    messages = [
        SystemMessage(content=system_instruction),
        HumanMessage(content=[
            {"type": "text", "text": f"{question} {context}"}, 
            {"type": "image_url", "image_url": {"url": image_url}}
            ]),
    ]
    return messages


def process_image(image, question):
    img_data = img_to_base64(image)
    
    messages = construct_prompt(img_data, question)
    res = llm.invoke(messages)
    return res.content

css = """
        .output-image img, .input-image img {
            max-height: 80vh; /* Limit to 80% of the viewport height */
            max-width: 80vw;  /* Limit to 80% of the viewport width */
            object-fit: contain; /* Maintain aspect ratio and fit within the bounds */
        }
        .input-column {
            max-height: 10vh; /* Adjust this value as needed */
            overflow-y: auto; /* If content exceeds, allow scrolling within the column */
        }
        """

with gr.Blocks(css=css) as demo:
    gr.Markdown("# Image Question Answering")
    
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="filepath", label="Input Image", 
                                sources=["upload", "webcam"], elem_classes="input-image")
        with gr.Column():
            question = gr.Textbox(label="Question",
                                value="描述图片中的内容",)
            submit = gr.Button("Submit")
            answer = gr.Textbox(label="Answer", interactive=False)
    
    submit.click(process_image, inputs=[image, question], outputs=answer)
    

if __name__ == "__main__":
    # img_path = "/datadrive/codes/frank/langchains/retrieval_anything/data/price/2096_9a71cc8154288e0878efccff4e4ed50_003981_039010_100000_084948.jpg"
    # res = process_image(img_path, "请描述图片中的内容")
    # print("Raw response:", res)
    demo.launch(server_name="0.0.0.0", server_port=8600)