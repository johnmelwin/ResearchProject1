import transformers

# Load the CodeT5 model
model = transformers.AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-base")

# Preprocess the input resources
requirements = "The system shall be able to calculate the sum of two numbers."
source_code = "def sum(a, b):\n  return a + b\n"
test_cases = ["assert sum(1, 2) == 3", "assert sum(3, 4) == 7"]

# Identify and establish links between the input resources
links = model.generate(requirements + "\n" + source_code + "\n" + test_cases, max_length=100, task="traceability")

# Postprocess the output resources
links = links[0].decode("utf-8").split("\n")

# Print the links
for link in links:
  print(link)
