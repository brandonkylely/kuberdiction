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
task_array = ["create a new deployment called 'nginx-deployment'",
        "scale the deployment 'nginx-deployment' to 3 replicas",
        "delete the deployment 'nginx-deployment'",
        "create a new pod called 'nginx-pod' using the image 'nginx:latest'",
        "get the logs of the pod 'nginx-pod'",
        "delete the pod 'nginx-pod'",
        "create a new service called 'nginx-service' that listens on port 80 and targets port 80",
        "delete the service 'nginx-service'",
        "create a new namespace called 'my-namespace'",
        "delete the namespace 'my-namespace'"]
# task = task_array[0]
# predicted_command = predict_kubectl_command_t5(task)
# print(f"Suggested command: {predicted_command}")

for task in task_array:
    predicted_command = predict_kubectl_command_t5(task)
    print(f"Task: {task}")
    print(f"Suggested command: {predicted_command}")
