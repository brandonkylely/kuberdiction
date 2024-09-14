from transformers import T5Tokenizer, T5ForConditionalGeneration

# load model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("./kuberdiction-t5")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

#  input --> task description, output --> Kubernetes command
def predict_kubectl_command_t5(task_description):
    # prefix to indicate task
    task_description = "translate kubectl task: " + task_description

    # tokenize input
    inputs = tokenizer(task_description, return_tensors="pt", max_length=128, truncation=True)

    # generate prediction
    outputs = model.generate(inputs.input_ids, max_length=128, num_beams=5, early_stopping=True)

    # decode predicted command
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# perform prediction
task = "create a new deployment called 'nginx-deployment'"
predicted_command = predict_kubectl_command_t5(task)
print(f"Suggested command: {predicted_command}")
