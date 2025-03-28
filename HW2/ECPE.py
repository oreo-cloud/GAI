import gradio as gr # 前端介面
from transformers import pipeline

def load_model():
    # 英文情感分析模型
    model_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    # 中文情感分析模型
    # model_pipeline = pipeline("text-classification", model="IDEA-CCNL/Erlangshen-Roberta-330M-Sentiment")
    return model_pipeline

def classify_text(model, text):
    result = model(text)
    return result

def main():
    model = load_model()
    interface = gr.Interface(fn=lambda text: classify_text(model, text),inputs=gr.Textbox(lines=2, placeholder="Enter Text Here..."),outputs="json",title="使用HuggingFace 的distilbert 進行情感分類",description="使用HuggingFace模型對文字情緒進行分類。請輸入英文句子查看其二元分類.")
    interface.launch()
if __name__ == "__main__":
    main()