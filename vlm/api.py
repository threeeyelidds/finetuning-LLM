from flask import Flask, request, jsonify
from PIL import Image
import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_default_device("cuda")

#Create model
model = AutoModelForCausalLM.from_pretrained(
    "MILVLG/imp-v1-3b", 
    torch_dtype=torch.float16, 
    device_map="auto",
    trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("MILVLG/imp-v1-3b", trust_remote_code=True,cache_dir='home/quanting/data')
PROMPT_FORMAT="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{prompt}? ASSISTANT:"

app = Flask(__name__)


def predict(image_path, text):
    image = Image.open(image_path)
    input_ids = tokenizer(PROMPT_FORMAT.format(prompt=text), return_tensors='pt').input_ids.to("cuda")
    image_tensor = model.image_preprocess(image)

    #Generate the answer
    output_ids = model.generate(
        input_ids,
        max_new_tokens=100,
        images=image_tensor,
        use_cache=True)[0]
    return tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()

@app.route('/', methods=['POST'])
def process_image():
    try:
        # Get the uploaded image from the request
        uploaded_file = request.files['image']
        if not uploaded_file:
            return jsonify({'error': 'No image provided'})

        # Save the image to a temporary file
        temp_image_path = 'temp_image.jpg'
        uploaded_file.save(temp_image_path)

        # Perform reasoning on the image
        text = request.form.get('text', "Describe the objects in the image and summarize")
        response = predict(temp_image_path, text)
        # Clean up the temporary image file
        import os
        os.remove(temp_image_path)

        return jsonify({'response': response})

    except Exception as e:
        return jsonify({'error': str(e)})
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)