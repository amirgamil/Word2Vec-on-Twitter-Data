# Word2Vec-on-Twitter-Data
In order to determine the language online associated with diseaes like meningitis, this program creates a word2vec model on tens of thousands of scraped twitter data contanining the keyword meningitis. I also create some visualizations which show the relationship between meningitis and other words which seem to be frequently used alongside it.


To ensure this runs successfully, if you don't have the following packages installed, make sure you do so (easiest way is through pip):
1. Install nltk
2. Install gensim
3. Install numpy
4. Install matplotlib
5. Install sklearn

I wasn't able to attach the JSON file containing all of the twitter data I scraped (has around 30/40,000 tweets - I forget the exact number) because it exceeded GitHub's file size limit. So instead, I attached the inital model I load. Feel free to rerun the code and generate your own model! This can be easily adapted if you choose to collect your own twitter/social media data on any topic and is as simple as just passing in the JSON file in the read_twitter_data() function.

One potential room for improvement is to do some more filtering before I pass the text into the model e.g. removing all punctuation, prepositions etc. I do this but not as thoroughly as I could be doing it.

![alt text](https://raw.githubusercontent.com/amirgamil/Word2Vec-on-Twitter-Data/master/2D%Representation%Most%Similar%Words.png)
