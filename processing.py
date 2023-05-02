import pandas as pd
import re
from sklearn.model_selection import train_test_split

df = pd.read_csv('raw_data/raw_data.tsv', delimiter='\t', error_bad_lines=False)
df = df.dropna()
df['review_headline'] = df['review_headline'].astype(str) + '.  ' 
df['review_body'] = df['review_headline']+df['review_body']
df = df[['review_body', 'star_rating']]

print('cleaning text')
df["review_body"] = df["review_body"].str.replace(r'\s*https?://\S+(\s+|$)', ' ').str.strip()
df['review_body'] = df['review_body'].str.replace('\d+', '')


df['review_body'] = df['review_body'].apply(lambda x: re.sub('[^0-9a-zA-Z ]+', '', x))
df = df.dropna()
df = df.rename(columns={'review_body': 'text','star_rating':'label'})

df['label'] = df['label']-1

df['label'] = df['label'].astype(int)
print("split data")
train, test = train_test_split(df, test_size=0.2)

train.to_csv('full_data/train.csv',index=False)
test.to_csv('full_data/test.csv',index=False)