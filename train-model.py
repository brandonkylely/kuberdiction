import pandas as pd
import torch
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

# read csv files
training_df = pd.read_csv('training-data.csv')
validation_df = pd.read_csv('validation-data.csv')

# convert dataframe to dataset (needs to be in this format for the model)
training_dataset = Dataset.from_pandas(training_df)
validation_dataset = Dataset.from_pandas(validation_df)

# tokenizer and padding token
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# tokenize inputs and targets
def preprocess_function(examples):
    inputs = examples['input']
    targets = examples['target']

    # add task-specific prefix to input (helps t5 understand context)
    inputs = ["translate kubectl task: " + input for input in inputs]

    # tokenize inputs and targets with padding and truncation
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding='max_length')

    # tokenize targets and add to 'labels'
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding='max_length')

    # add labels to model inputs
    model_inputs['labels'] = labels['input_ids']

    return model_inputs

# preprocess both training and validation datasets
training_dataset = training_dataset.map(preprocess_function, batched=True)
validation_dataset = validation_dataset.map(preprocess_function, batched=True)

# remove unnecessary columns
training_dataset = training_dataset.remove_columns(['input', 'target'])
validation_dataset = validation_dataset.remove_columns(['input', 'target'])

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_total_limit=2,
    save_steps=500,
    load_best_model_at_end=True
)

# set model to use GPU
model = model.to('cuda')

# initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=training_dataset,
    eval_dataset=validation_dataset,
)

# train the model
trainer.train()

# save the model
trainer.save_model('kuberdiction-t5')
