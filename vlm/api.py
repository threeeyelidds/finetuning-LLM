from flask import Flask, request, jsonify
from PIL import Image
import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoProcessor, LlavaForConditionalGeneration

torch.set_default_device("cuda")

#Create model
model = AutoModelForCausalLM.from_pretrained(
    "MILVLG/imp-v1-3b", 
    torch_dtype=torch.float16, 
    device_map="auto",
    trust_remote_code=True,
    # load_in_4bit=True
    )
tokenizer = AutoTokenizer.from_pretrained("MILVLG/imp-v1-3b", trust_remote_code=True,cache_dir='home/quanting/data')
PROMPT_FORMAT="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{prompt}? ASSISTANT:"
# model_id = "llava-hf/llava-1.5-7b-hf"
# model = LlavaForConditionalGeneration.from_pretrained(
#     model_id, 
#     torch_dtype=torch.float16, 
#     low_cpu_mem_usage=True, 
#     use_flash_attention_2=True,
#     cache_dir='home/quanting/data'
# ).to(0)
# processor = AutoProcessor.from_pretrained(model_id)
# PROMPT_FORMAT = "USER: <image>\n{prompt}\nASSISTANT:"
tokenizer_llm = AutoTokenizer.from_pretrained("google/gemma-2b-it")
model_llm = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it", device_map="auto")

app = Flask(__name__)

# def predict_with_llava(image_path, text):
#     image = Image.open(image_path)
#     inputs = processor(PROMPT_FORMAT.format(prompt=text), image, return_tensors='pt').to(0, torch.float16)
#     output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
#     return processor.decode(output[0][:], skip_special_tokens=True)
def predict_llm(input_text):
    input_ids = tokenizer_llm(input_text, return_tensors="pt").to("cuda")
    outputs = model_llm.generate(**input_ids,max_new_tokens=1024)
    return tokenizer_llm.decode(outputs[0])

def predict_llm_chat(chat):
    templated_text = tokenizer_llm.apply_chat_template(chat, tokenize=False)
    input_ids = tokenizer_llm(templated_text, return_tensors="pt").to("cuda")
    outputs = model_llm.generate(**input_ids,max_new_tokens=128)
    response = tokenizer_llm.decode(outputs[0])
    return response


@app.route('/llm-chat', methods=['POST'])
def process_text_chat():
    try:
        # Perform reasoning on the image
        data = request.get_json(force=True)  # 'force=True' is optional, ensures JSON parsing regardless of 
        default_response = [{"role": "system", "content": "write the solution of two sum"}]
        # Extract 'chat' list from the JSON data, use default if not found
        chat = data.get('chat', default_response)
        response = predict_llm_chat(chat)

        return jsonify({'response': response})

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/llm', methods=['POST'])
def process_text():
    try:
        # Perform reasoning on the image
        text = request.form.get('text', "write the solution of two sum")
        response = predict_llm(text)

        return jsonify({'response': response})

    except Exception as e:
        return jsonify({'error': str(e)})
    
def predict_vlm(image_path, text):
    image = Image.open(image_path)
    input_ids = tokenizer(PROMPT_FORMAT.format(prompt=text), return_tensors='pt').input_ids.to("cuda")
    image_tensor = model.image_preprocess(image)

    #Generate the answer
    output_ids = model.generate(
        input_ids,
        max_new_tokens=500,
        images=image_tensor,
        use_cache=True)[0]
    return tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()

def predict_vlm_batch(image_paths, texts):
    prompts = [PROMPT_FORMAT.format(prompt=text) for text in texts]
    input_ids = tokenizer(prompts, return_tensors='pt').input_ids.to("cuda")
    images_tensors = torch.stack([model.image_preprocess(Image.open(path)) for path in image_paths]).to('cuda')
    
    #Generate the answer
    output_ids = model.generate(
        input_ids,
        max_new_tokens=500,
        images=images_tensors,
        use_cache=True)[0]
    answers = [tokenizer.decode(output[input_id.shape[1]:], skip_special_tokens=True).strip() for output, input_id in zip(output_ids, input_ids)]
    
    return answers

@app.route('/vlm-batch', methods=['POST'])
def process_image_batch():
    try:
        # Get the uploaded image from the request
        uploaded_file = request.files['image']
        if not uploaded_file:
            return jsonify({'error': 'No image provided'})

        # Save the image to a temporary file
        temp_image_path = 'temp_image.jpg'
        uploaded_file.save(temp_image_path)

        # Perform reasoning on the image
        text = request.form.get('text', "Examine the image closely and identify any paths visible. For each path, evaluate and score from 1 to 5 based on the likelihood of it leading to a kitchen, considering visible cues such as layout, direction, and objects. Provide reasoning for each score.")
        response = predict_vlm_batch(temp_image_path, text)
        # Clean up the temporary image file
        import os
        os.remove(temp_image_path)

        return jsonify({'response': response})

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/vlm', methods=['POST'])
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
        text = request.form.get('text', "Examine the image closely and identify any paths visible. For each path, evaluate and score from 1 to 5 based on the likelihood of it leading to a kitchen, considering visible cues such as layout, direction, and objects. Provide reasoning for each score.")
        response = predict_vlm(temp_image_path, text)
        # Clean up the temporary image file
        import os
        os.remove(temp_image_path)

        return jsonify({'response': response})

    except Exception as e:
        return jsonify({'error': str(e)})
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)