# Import necessary libraries
import pandas as pd
import re
from sklearn.model_selection import train_test_split

# Load the raw data from a TSV file
df = pd.read_csv('raw_data/raw_data.tsv', delimiter='\t', error_bad_lines=False)

# Drop rows with missing values
df = df.dropna()

# Combine review_headline and review_body to create the review text
df['review_headline'] = df['review_headline'].astype(str) + '.  ' 
df['review_body'] = df['review_headline']+df['review_body']

# Keep only the relevant columns for further processing
df = df[['review_body', 'star_rating']]

# Clean the text
print('cleaning text')

# Remove URLs
df["review_body"] = df["review_body"].str.replace(r'\s*https?://\S+(\s+|$)', ' ').str.strip()

# Remove digits
df['review_body'] = df['review_body'].str.replace('\d+', '')

# Remove special characters
df['review_body'] = df['review_body'].apply(lambda x: re.sub('[^0-9a-zA-Z ]+', '', x))

# Drop rows with missing values after cleaning
df = df.dropna()

# Rename columns for clarity
df = df.rename(columns={'review_body': 'text','star_rating':'label'})

# Adjust labels to start from 0 instead of 1
df['label'] = df['label']-1

# Split the data into training and testing sets
df['label'] = df['label'].astype(int)
print("split data")
train, test = train_test_split(df, test_size=0.2)

# Save the training and testing data to CSV files
train.to_csv('full_data/train.csv',index=False)
test.to_csv('full_data/test.csv',index=False)
