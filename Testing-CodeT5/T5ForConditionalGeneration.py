# Testing CodeT5



from transformers import RobertaTokenizer, T5ForConditionalGeneration

tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-large')
model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-large')


text = "def add(a, b): " \
       "return (<extra_id_0>"
input_ids = tokenizer(text, return_tensors="pt").input_ids

# simply generate a single sequence
generated_ids = model.generate(input_ids, max_length=10)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
# this prints "user: {user.name}"