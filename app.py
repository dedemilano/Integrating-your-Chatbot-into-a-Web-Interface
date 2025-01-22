from flask import Flask ,request
import json
from flask_cors import CORS

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "facebook/blenderbot-400M-distill"

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name)

conversation_history = []

app = Flask(__name__)
CORS(app)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/chat', methods=['POST'])
def handle_prompt():
    data = request.get_data(as_text=True)
    data = json.loads(data)
    input_text = data['prompt']

    history = '\n'.join(conversation_history)

    inputs = tokenizer.encode_plus(history , input_text , return_tensors='pt')

    outputs = model.generate(**inputs , max_length=60)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    conversation_history.append(input_text)
    conversation_history.append(response)

    return response


if __name__ == '__main__':
    app.run()