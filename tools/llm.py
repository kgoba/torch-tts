# Use a pipeline as a high-level helper
# from transformers import pipeline

# pipe = pipeline("text-generation", model="fblgit/una-cybertron-7b-v2-bf16")

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("fblgit/una-cybertron-7b-v2-bf16")
model = AutoModelForCausalLM.from_pretrained("fblgit/una-cybertron-7b-v2-bf16")



prompt = "Hey, are you conscious? Can you talk to me?"

inputs = tokenizer(prompt, return_tensors="pt")

# Generate

generate_ids = model.generate(inputs.input_ids, max_length=30)

tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]