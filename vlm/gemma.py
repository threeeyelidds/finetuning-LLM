from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it")

input_text = "Write me a poem about Machine Learning."
chat = [
   {"role": "user", "content": "Hello, how are you?"},
   {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
   {"role": "user", "content": "I'd like to show off how chat templating works!"},
]

templated_text = tokenizer.apply_chat_template(chat, tokenize=False)
print(templated_text)
input_ids = tokenizer(templated_text, return_tensors="pt")

outputs = model.generate(**input_ids,max_new_tokens=128)
print(tokenizer.decode(outputs[0]))
