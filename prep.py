import pandas as pd
from sklearn.model_selection import train_test_split

# read excel file
df = pd.read_excel('kubernetes-data.xlsx')

# input: objective --> talks about what the command does
# target: command --> kubectl command
df['input'] = df['objective']
df['target'] = df['command']

# splitting data into training and validation datasets to avoid overfitting
training_df, validation_df = train_test_split(df[['input', 'target']], test_size=0.2, random_state=42)

# create csv files
training_df.to_csv('training-data.csv', index=False)
validation_df.to_csv('validation-data.csv', index=False)
