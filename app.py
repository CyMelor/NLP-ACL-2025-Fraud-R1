from flask import Flask, render_template, request, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import os
import time
from werkzeug.utils import secure_filename
import tempfile

app = Flask(__name__)

# 加载模型和分词器
MODEL_PATH = "i:\\Study\\Studyitem\\NLP\\end\\models\\bert_fraud_original.pt"
MODEL_NAME = "bert-base-chinese"

# 尝试加载分词器和模型，处理网络超时情况
tokenizer = None
model = None
try_count = 0
max_retries = 3

# 加载模型的函数
def load_model(model_path):
    global model
    if model is None:
        # 重新加载模型框架
        model = BertForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=2,
            ignore_mismatched_sizes=True
        )
    
    # 加载模型权重
    try:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        model.to(device)
        print(f"模型加载成功: {model_path}")
        return True
    except Exception as e:
        print(f"模型加载失败: {e}")
        return False

# 初始化加载分词器和模型框架
while try_count < max_retries and (tokenizer is None or model is None):
    try:
        # 尝试加载分词器
        if tokenizer is None:
            tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
            print("分词器加载成功")
        
        # 尝试加载模型框架
        if model is None:
            model = BertForSequenceClassification.from_pretrained(
                MODEL_NAME,
                num_labels=2,
                ignore_mismatched_sizes=True
            )
            print("模型框架加载成功")
        
    except Exception as e:
        try_count += 1
        print(f"尝试加载模型失败 (第{try_count}/{max_retries}次): {e}")
        if try_count < max_retries:
            print("等待3秒后重试...")
            time.sleep(3)
        else:
            print("达到最大重试次数，模型加载失败")
            raise

# 检查CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载默认模型权重
if os.path.exists(MODEL_PATH):
    load_model(MODEL_PATH)
else:
    raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")

model.to(device)

def preprocess_dialogue(dialogue):
    """预处理对话文本"""
    if pd.isna(dialogue):
        return ""
    return str(dialogue).strip()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_model', methods=['POST'])
def upload_model():
    try:
        if 'model_file' not in request.files:
            return jsonify({'error': '没有选择模型文件'})
        
        file = request.files['model_file']
        
        if file.filename == '':
            return jsonify({'error': '没有选择模型文件'})
        
        if file and file.filename.endswith('.pt'):
            # 使用临时文件保存上传的模型
            with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name
            
            # 加载新模型
            if load_model(temp_file_path):
                # 删除临时文件
                os.unlink(temp_file_path)
                return jsonify({'success': '模型加载成功'})
            else:
                # 删除临时文件
                os.unlink(temp_file_path)
                return jsonify({'error': '模型加载失败，请检查模型文件格式'})
        else:
            return jsonify({'error': '请上传.pt格式的模型文件'})
            
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 获取输入的对话文本
        dialogue = request.form['dialogue']
        
        # 预处理文本
        processed_dialogue = preprocess_dialogue(dialogue)
        
        # 编码文本
        encoding = tokenizer.encode_plus(
            processed_dialogue,
            add_special_tokens=True,
            max_length=128,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        # 将编码后的数据移动到设备上
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # 模型预测
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            confidence = torch.softmax(outputs.logits, dim=1)[0].tolist()
        
        # 准备结果
        result = {
            'prediction': '欺诈' if preds.item() == 1 else '正常',
            'confidence': confidence[preds.item()],
            'dialogue': dialogue
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)})

import webbrowser

if __name__ == '__main__':
    # 定义服务器端口
    port = 8001
    
    # 打开网页
    url = f"http://localhost:{port}"
    print(f"服务器即将启动，访问地址：{url}")
    webbrowser.open(url)
    
    # 启动服务器（在主线程中）
    app.run(debug=False, host='0.0.0.0', port=port)
