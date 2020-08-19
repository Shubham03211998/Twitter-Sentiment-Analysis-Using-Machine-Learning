#This module is for preprocessing the data in dataset and tweets fetched
import re
import unicodedata

def pre_processing(tweet):

    # Removing characters above U+FFFF
    tweet = ''.join(c for c in unicodedata.normalize('NFC', tweet) if c <= '\uFFFF')

    # Removing URL from the content
    tweet = re.sub(r'http\S+', '', tweet)

    # Removing smile Emojis from content
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', '', tweet)

    # Removing Laugh Emojis from content
    tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', '', tweet)

    # Removing love Emojis from content
    tweet = re.sub(r'(<3|:\*|💙)', '', tweet)
    tweet = re.sub(r'(🥰|💓|💖|💗|💞|💘|💘|💓|❤|💚|🧡|🧡|💞|💓|💖|💕|💘|💞|öö|🙃|👀|👇|🔵)', '', tweet)

    # Removing wink Emojis from content
    tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', '', tweet)

    # Removing sad Emojis from content
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', '', tweet)

    # Removing cry Emojis from content
    tweet = re.sub(r'(:,\(|:\'\(|:"\()', '', tweet)

    # Removing Usernames from content
    tweet = re.sub('@[^/s]+', '', tweet)

    # Removing Numbers from content
    pattern = '[0-9]'
    tweet = [re.sub(pattern, '', i) for i in tweet]
    x = ""
    tweet = x.join(tweet)

    # Removing repetation of more than 2 letters and converting them to two letters
    # Removing (for eg:-cooooool --> cool) from content
    tweet = re.sub(r'(.)\1+', r'\1\1', tweet)

    # Removing All the special chanacters
    tweet = re.sub(r'(-|\'|\?|\/|\\|\`|\!|\`|\#|\$|\%|\^|\&|\*)', '', tweet)
    tweet = re.sub(r'(\*|\(|\)|\_|\+|\=|\.|\,|\<|\>|\{|\[|\}|\]|\"|\;|\:|\~)', '', tweet)

    # Removing multiple whitespaces
    tweet = " ".join(tweet.split())
    tweet = str(tweet.lower())

    return tweet