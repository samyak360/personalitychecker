from django.shortcuts import render, redirect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from string import punctuation
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Create your views here.

def index(request):
    return render(request,'twitterhandle/homePage.html')

def details(request):
    return render(request,'twitterhandle/detail.html')

def handle(request):
	d={}
	if request.method == "POST":
		handle = request.POST.get("handle")
		handle = "@"+handle
		d['answered']="False"
		data_download = pd.read_csv('mbti_1.csv')
		data = data_download.copy()
		pd.options.display.max_colwidth = 100
		data.head()
		def remove_url(x):
			x = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}     /)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', 'https', x)
			return x
		data.posts = data.posts.apply(remove_url)
		punctuation_symbols = []
		for each in list(punctuation):
			punctuation_symbols.append((each, ' '))

		def remove_puncuation(x):
			for each in punctuation_symbols:
				x = x.replace(*each)
				return x
				data.posts = data.posts.apply(remove_puncuation)
		def remove_digits(x):
			x = ''.join([i for i in x if not i.isdigit()])
			return x

		data.posts = data.posts.apply(remove_digits)

		#4.lowercase and stop word removal
		stop = stopwords.words('english')

		def remove_stop_words(x):
			x = ' '.join([i for i in x.lower().split(' ') if i not in stop])
			return x

		data.posts = data.posts.apply(remove_stop_words)

		#5.remove excess white space
		def remove_extra_white_space(x):
			x = ' '.join(x.split())
			return x

		data.posts = data.posts.apply(remove_extra_white_space)

		#6.remove the types
		def remove_types(x):
			x = re.sub(r'(i|e)(n|s)(t|f)(j|p)\w*|\b(ne|sx|ni|ti|fe|nt|se|si|nf)\b', '', x)
			return x

		data.posts = data.posts.apply(remove_types)

		#using TF-IDF algo
		tf = TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df = 0.02, stop_words = 'english', norm='l2')

		# fit & transform
		tfidf_matrix = tf.fit_transform(data.posts)

		#print('Number of documents:', tfidf_matrix.shape[0], ', number of features:', tfidf_matrix.shape[1])

		tfidf_feature_matrix = pd.DataFrame(tfidf_matrix.toarray(), columns=tf.get_feature_names())
		tfidf_feature_matrix.head()

		#random forest

		X = tfidf_feature_matrix
		Y = data['type']

		X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

		rfc = ensemble.RandomForestClassifier()

		rfc.fit(X_train, Y_train)

		#logistic regression

		lr = LogisticRegression()
		lr.fit(X_train, Y_train)


		personality_types = sorted(data.type.unique())

		Y_pred = lr.predict(X_test)
		confusion_array = confusion_matrix(Y_test, Y_pred)

		df_cm = pd.DataFrame(confusion_array, index = [i for i in personality_types],
			columns = [i for i in personality_types])

		tfidf_feature_matrix2 = tfidf_feature_matrix.copy()

		tfidf_feature_matrix2['type_IE'] = [x[0] for x in data.type]
		tfidf_feature_matrix2['type_NS'] = [x[1] for x in data.type]
		tfidf_feature_matrix2['type_TF'] = [x[2] for x in data.type]
		tfidf_feature_matrix2['type_JP'] = [x[3] for x in data.type]

		tfidf_feature_matrix2.head()

		N = 4
		but = (tfidf_feature_matrix2.type_IE.value_counts()[0], tfidf_feature_matrix2.type_NS.value_counts()[0], 
			tfidf_feature_matrix2.type_TF.value_counts()[0], tfidf_feature_matrix2.type_JP.value_counts()[0])
		top = (tfidf_feature_matrix2.type_IE.value_counts()[1], tfidf_feature_matrix2.type_NS.value_counts()[1], 
			tfidf_feature_matrix2.type_TF.value_counts()[1], tfidf_feature_matrix2.type_JP.value_counts()[1])

		ind = np.arange(N)    # the x locations for the groups
		width = 0.7      # the width of the bars: can also be len(x) sequence

		#Intuitive/Sensing Model:
		data_N = tfidf_feature_matrix2.loc[tfidf_feature_matrix2['type_NS'] == 'N'].sample(n=1197)
		data_S = tfidf_feature_matrix2.loc[tfidf_feature_matrix2['type_NS'] == 'S']
		data_NS = pd.concat([data_N, data_S])

		from sklearn.model_selection import cross_val_score
		#from sklearn.metrics import classification_report

		drop_list = ['type_IE','type_NS',"type_TF",'type_JP']

		Y = data_NS['type_NS']
		X = data_NS.drop(drop_list, axis=1)

		X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

		lr_NS = LogisticRegression()
		lr_NS.fit(X_train, Y_train)

		# print('\nLogistic Regression:')
		# print('Training set score:', lr_NS.score(X_train, Y_train))
		# lr_scores_NS = cross_val_score(lr_NS, X, Y, cv=5)
		# print('Cross Validation Scores: ' , lr_scores_NS)
		# print('Average score: ', np.mean(lr_scores_NS))

		personality_types_NS = sorted(data_NS.type_NS.unique() )

		Y_pred = lr_NS.predict(X_test)
		confusion_array = confusion_matrix(Y_test, Y_pred)

		df_cm = pd.DataFrame(confusion_array, index = [i for i in personality_types_NS],
			columns = [i for i in personality_types_NS])

		#Introverted/Extroverted Model:
		data_I = tfidf_feature_matrix2.loc[tfidf_feature_matrix2['type_IE'] == 'I'].sample(n=1999)

		data_E = tfidf_feature_matrix2.loc[tfidf_feature_matrix2['type_IE'] == 'E']

		data_IE = pd.concat([data_I, data_E])


		Y = data_IE['type_IE']
		X = data_IE.drop(drop_list, axis=1)

		X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

		lr_IE = LogisticRegression()
		lr_IE.fit(X_train, Y_train)

		# print('\nLogistic Regression:')
		# print('Training set score:', lr_IE.score(X_train, Y_train))
		# lr_score_IE = cross_val_score(lr_IE, X, Y, cv=5)
		# print('Cross Validation Scores: ' , lr_score_IE)
		# print('Average Score: ', np.mean(lr_score_IE))

		personality_types_IE = sorted(data_IE.type_IE.unique() )

		Y_pred = lr_IE.predict(X_test)
		confusion_array = confusion_matrix(Y_test, Y_pred)

		df_cm = pd.DataFrame(confusion_array, index = [i for i in personality_types_IE],
			columns = [i for i in personality_types_IE])

		#Judging/Prospecting Model:
		data_P = tfidf_feature_matrix2.loc[tfidf_feature_matrix2['type_JP'] == 'P'].sample(n=3434)
		data_J = tfidf_feature_matrix2.loc[tfidf_feature_matrix2['type_JP'] == 'J']
		data_JP = pd.concat([data_J, data_P])


		Y = data_JP['type_JP']
		X = data_JP.drop(drop_list, axis=1)

		X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

		lr_JP = LogisticRegression()
		lr_JP.fit(X_train, Y_train)

		# print('\nLogistic Regression:')
		# print('Training set score:', lr_JP.score(X_train, Y_train))
		# lr_scores_JP = cross_val_score(lr_JP, X, Y, cv=5)
		# print('\nCross Validation Scores:\n' , lr_scores_JP)
		# print('Average Scores: ', np.mean(lr_scores_JP))

		personality_types_JP = sorted(tfidf_feature_matrix2.type_JP.unique())

		Y_pred = lr_JP.predict(X_test)
		confusion_array = confusion_matrix(Y_test, Y_pred)

		df_cm = pd.DataFrame(confusion_array, index = [i for i in personality_types_JP],
			columns = [i for i in personality_types_JP])

		#Thinking/Feeling Model:
		Y = tfidf_feature_matrix2['type_TF']
		X = tfidf_feature_matrix2.drop(drop_list, axis=1)

		X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

		lr_TF = LogisticRegression()
		lr_TF.fit(X_train, Y_train)

		# print('\nLogistic Regression:')
		# print('Training set score:', lr_TF.score(X_train, Y_train))
		# lr_scores_TF = cross_val_score(lr_TF, X, Y, cv=5)
		# print('\nCross Validation Scores:\n' , lr_scores_TF)
		# print('Average Score: ', np.mean(lr_scores_TF))

		personality_types_TF = sorted(tfidf_feature_matrix2.type_TF.unique())

		Y_pred = lr_TF.predict(X_test)
		confusion_array = confusion_matrix(Y_test, Y_pred)

		df_cm = pd.DataFrame(confusion_array, index = [i for i in personality_types_TF],
			columns = [i for i in personality_types_TF])

		#Twitter Application
		import tweepy

		ckey = "zLceR2GtRk7LiUMXRzLAYQ1ML"
		csecret = "gUQWOfGKI7voYlfLD5rpl5KiQrXQTpzIQXLmPjsaYQO3cGhefo"
		atoken = "819671108-WsXdRIP2tk9oBbh9ISPXAnKfVv567xPX1OsBDfSt"
		asecret = "FSGAGxj2atE2HKqGlLYUS3KC9NGtzc8akeqlNBDrfgc1C"

		auth = tweepy.OAuthHandler(ckey, csecret)

		def get_twitter_type(twitter_handle):
			auth.set_access_token(atoken, asecret)
			print(twitter_handle)
			api = tweepy.API(auth)

			stuff = api.user_timeline(screen_name = twitter_handle, count = 300, include_rts = False)

			twitter_text = ''
			for status in stuff:
				twitter_text += status.text
				twitter_text += ' '

				twitter_text = remove_url(twitter_text)
				twitter_text = remove_puncuation(twitter_text)
				twitter_text = remove_digits(twitter_text)
				twitter_text = remove_stop_words(twitter_text)
				twitter_text = remove_extra_white_space(twitter_text)
				twitter_text = remove_types(twitter_text)

				my_tfidf_matrix2 = tf.transform([twitter_text])

				my_tfidf_feature_matrix2 = pd.DataFrame(my_tfidf_matrix2.toarray(), columns=tf.get_feature_names())
				my_tfidf_feature_matrix2.head()

			d['ans']=lr_IE.predict(my_tfidf_feature_matrix2)[0] 
			d['ans']=d['ans']+ lr_NS.predict(my_tfidf_feature_matrix2)[0] 
			d['ans']=d['ans']+ lr_TF.predict(my_tfidf_feature_matrix2)[0] 
			d['ans']=d['ans']+ lr_JP.predict(my_tfidf_feature_matrix2)[0]
			print(d['ans'])


		from langdetect import detect

		MAX_TWEETS = 500

		auth = tweepy.OAuthHandler(ckey, csecret)
		api = tweepy.API(auth)

		for tweet in tweepy.Cursor(api.search, q='#INFP', rpp=100).items(MAX_TWEETS):
			if detect(tweet._json['text']) == 'en':
				# print(tweet.user.name)
				pass


		get_twitter_type(handle)
		d['answered']="True"

		return render(request,'twitterhandle/handle.html',context=d)
	else:
		return render(request,'twitterhandle/handle.html',context=d)

def myers(request):
    return render(request,'twitterhandle/myers-briggs.html')
