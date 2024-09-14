import pandas as pd
import torch
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

# read csv files
training_df = pd.read_csv('training-data.csv')
validation_df = pd.read_csv('validation-data.csv')

# convert dataframe to dataset (needs to be in this format for the model)
training_dataset = Dataset.from_pandas(training_df)
validation_dataset = Dataset.from_pandas(validation_df)

# tokenizer and padding token
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# tokenize inputs and targets
def preprocess_function(examples):
    inputs = examples['input']
    targets = examples['target']
    
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

# remove columns
training_dataset = training_dataset.remove_columns(['input', 'target'])
validation_dataset = validation_dataset.remove_columns(['input', 'target'])

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
)

# load the model
model = GPT2LMHeadModel.from_pretrained('gpt2')

# padding token
model.resize_token_embeddings(len(tokenizer))

# set model to use GPU
model = model.to('cuda')

# initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=training_dataset,
    eval_dataset=validation_dataset,
)


# train the model
trainer.train()
