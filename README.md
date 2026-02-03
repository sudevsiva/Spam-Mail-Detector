# Spam-Mail-Detector
# Build a classifier that distinguishes between spam and non-spam (ham) emails using textual data.

#import libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Import file
df=pd.read_csv("D:\\ML project\\spam.csv",encoding="latin1")

#give coloumn name
df=df[['v1','v2']]
df.columns=['label','message']

#label split in to numerical
df['label']=df['label'].map({'spam':1,'ham':0})

#split X and y
X=df['message']
y=df['label']

#split train and test
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.3,random_state=42)

#Vectorize the X

Vector=TfidfVectorizer(stop_words="english")
X_train_tdf=Vector.fit_transform(X_train)
X_test_tdf=Vector.transform(X_test)

#create model
model=LogisticRegression(max_iter=1000)
model.fit(X_train_tdf,y_train)

#Predict the model
y_pred=model.predict(X_test_tdf)

#Print accuracy and classification report
print("accuarcy:",accuracy_score(y_test, y_pred))
print("Classification_report:",classification_report(y_test, y_pred))
    
#Run with sample  
sample=["You won the draw, calim your prize"]
sample_tfdf=Vector.transform(sample)
pred=model.predict(sample_tfdf)
print("spam" if pred[0]==1 else "ham")
