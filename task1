news_data= pd.read_csv("news.csv")
news_data.head(10)
news_data.info()
news_data.shape
news_data["label"].value_counts()
labels= news_data.label
labels.head(10)
#First, we split the dataset into train & test samples:
x_train, x_test, y_train, y_test= train_test_split(news_data["text"], labels, test_size= 0.4, random_state= 7)
#Then we’ll initialize TfidfVectorizer with English stop words
vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train=vectorizer.fit_transform(x_train) 
tfidf_test=vectorizer.transform(x_test)
#Create a PassiveAggressiveClassifier
passive=PassiveAggressiveClassifier(max_iter=50)
passive.fit(tfidf_train,y_train)

y_pred=passive.predict(tfidf_test)
#Create a confusion matrix
matrix= confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])
matrix
#Visualize the confusion matrix
sns.heatmap(matrix, annot=True)
plt.show()
#Calculate the model's accuracy
Accuracy=accuracy_score(y_test,y_pred)
Accuracy*100
Report= classification_report(y_test, y_pred)
print(Report)
