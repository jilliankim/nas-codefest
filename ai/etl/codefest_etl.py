import re
import nltk

# Remove special characters and convert to lowercase
stop_words=set(nltk.corpus.stopwords.words('english'))
def preprocessing(dataframe, text):
    dataframe["clean_text"] = dataframe[text].apply(lambda x: re.sub("[^A-Za-z0-9\/. ]", "", str(x.lower())))

    dataframe['text_without_stopwords'] = dataframe['clean_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    return dataframe
    
    
# Convert labels to integers
def int_conversion(dataframe):
    dataframe['labels'] = dataframe['labels'].astype(np.int)
    return dataframe
    