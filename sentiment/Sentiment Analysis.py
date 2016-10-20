import pandas
import re
import csv
from anew_module import anew
from nltk.corpus import stopwords as sw

colnames = ['text', 'author', 'cit', 'time', 'vid']

#importing data and removing the first column, which contains observation number
data = pandas.read_csv('C:\Users\Rose Nyameke\OneDrive - North Carolina State University\Classes\Fall 2\Text Mining\Project\Merged Comments.csv', names=colnames, usecols = [1,2,3,4,5], header = 0)

print ("Finished reading data")
text=data.text.tolist()

#make all the text lowercase
lowertext=[i.lower() for i in text]

text_nouser = []
#remove usernames, substitutes any word that begins with a '+' with nothing. will leave names if they are two word names though
for texts in lowertext:
    text_nouser.append(re.sub("\+(\w+)","", texts))

#function to remove non-ascii characters and punctuations
def strip_non_ascii(string):
    stripped = (c for c in string if (ord(c) == 32) or (64 < ord(c) < 127)) #preserving space values
    return ''.join(stripped)

#stopwords using NLTK
stopwords = sw.words('english')
#removing trump's name because it's a valid english word, and exists in the anew module; all other candidate names pose no issues
stopwords.append('trump')
stopwords.append('trumps')

#apply function to remove non-ascii characters
text_noasc = [strip_non_ascii(item) for item in text_nouser]

valence = []
arousal = []
scored_terms = []
video_ids = []

#iterating over term vectors, obtaining sentiment valence and arousal if more than two terms exist in the dictionary
print ("Cleaning and scoring")

#keeping track of the vid column
vid = -1
for item in text_noasc:
    termlist=item.split()
    clean_terms = []
    vid += 1

    #remove the stopwords
    for words in termlist:
        if words not in stopwords:
            clean_terms.append(words)

    if sum(anew.exist(clean_terms)) >=2:
        valence.append ((anew.sentiment(clean_terms))['valence'])
        arousal.append ((anew.sentiment(clean_terms))['arousal'])
        scored_terms.append ([term for term in clean_terms if anew.exist(term)])
        video_ids.append((data['vid'][vid]).split()[:2])

#cleaning up the candidate names
video_ids = [' '.join(vid) for vid in video_ids]

#cleaning up the terms
scored_terms = [','.join(s_term) for s_term in scored_terms]

#combining the list of terms with their scores
values = zip(scored_terms, valence, arousal, video_ids)

#writing file
print ('Writing file')
headers = ['terms', 'valence', 'arousal', 'candidate']
with open('sentiment_scores.csv', 'wb') as out:
    writer = csv.writer( out )
    #write headers
    writer.writerow(headers)
    #write the csv file
    for value in values:
        writer.writerow(value)
