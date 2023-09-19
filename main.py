#Importing all important libraries for sentiment analysis
import numpy as np
import pandas as pd
import torch
import requests
from bs4 import BeautifulSoup
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification

#choosing the pre-trained NLP BERT model
tokenizer=AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model=AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

#Data scraping using bs4
r=requests.get('https://www.imdb.com/title/tt0816692/reviews')
soup=BeautifulSoup(r.text,'html.parser')
regex=re.compile(',*text show-more__control,*')
results=soup.find_all('div',{'class':regex})
reviews=[result.text for result in results]
max_review_length = 100

# shortened the reviews because BERT model doesn't take values more than 512
shortened_reviews = []
for review in reviews:
    words = review.split()
    if len(words) > max_review_length:
        shortened_review = ' '.join(words[:max_review_length])
    else:
        shortened_review = review
    shortened_reviews.append(shortened_review)

#making of dataframe
df=pd.DataFrame(np.array(shortened_reviews), columns=['review'])

def sentiment_score(review):
    tokens=tokenizer.encode(review, return_tensors='pt')
    result=model(tokens)
    return int(torch.argmax(result.logits))+1

df['sentiment']=df['review'].apply(lambda x: sentiment_score(x[:100]))

#converting to csv file
df.to_csv('result.csv', index=False)