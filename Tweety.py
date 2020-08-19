#This is just a Research Project on Twitter Sentimental Anlysis Using Machine Learning Algorithm Developed By Shubham Dubey
#In this application we have used several machine learning classification algorithms which accept sparse matrix
#And used only those algorithms which work with at least two categories 
# And defined tweets in two categories as Positive and Negative
# It will import all the modules stored in MainImport Module
from MainImport import *

#Imported module contaning GUI functions
from tkinter import *
from PIL import Image, ImageTk
import time
import tkinter.scrolledtext as st 

#Twitter API Authentication Process
class TwitterClient():
    def __init__(self, twitter_user=None):
        self.auth = TwitterAuthenticator().authenticate_twitter_app()
        self.twitter_client = API(self.auth)
        self.twitter_user = twitter_user

    def get_twitter_client_api(self):
        return self.twitter_client


#Twitter API credentials Invocation and Authentication
class TwitterAuthenticator():

    def authenticate_twitter_app(self):
        auth = OAuthHandler(credentials.api['API_KEY'], credentials.api['API_SECRET_KEY'])
        auth.set_access_token(credentials.api['ACCESS_TOKEN'], credentials.api['ACCESS_TOKEN_SECRET'])
        return auth



# Root Frame Design
def raise_frame(frame):
    frame.tkraise()

#Created root of the window
root = Tk()

#Created four frames from root window
f1 = Frame(root)
f2 = Frame(root)
f3 = Frame(root)
f4 = Frame(root)

for frame in (f1, f2, f3, f4):
    frame.grid(row=0, column=0, sticky='news')

# Root title declaration
root.title("TWEETY")
img1 = ImageTk.PhotoImage(Image.open("Icon.png").resize((300, 300),Image.ANTIALIAS) )

#First Frame Starts Here
#Introduction page

f1.configure(bg='red')
Label(f1, text="TWEETY - A SENTIMENTAL ANALYSIS TOOL", bg="white", fg="black",relief="solid").pack(side=TOP, fill=X, padx=2, pady=2)
Label(f1, image=img1, bg="red", fg="black").pack(side=TOP, fill=X)
Label(f1, text="TWEETY",fg="red", bg="red").pack(side=TOP,pady=4)
Label(f1, text="TWEETY", bg="White", fg="black", font=("Times new roman", "18","bold"),relief="solid").pack(side=TOP, fill=X, padx=2, pady=2)
Label(f1, text="This is a tool based on\n Machine Learning algorithms for\n sentiment analysis of tweets\n from Twitter user accounts",font=("Times new roman", "12"), bg="white", fg="black",relief="solid").pack(side=TOP,padx=2, pady=2, fill=X)
Button(f1, text='Get started',relief="solid", command=lambda:raise_frame(f2)).pack(side=BOTTOM,fill=X, padx=2, pady=2)
#End of First Frame


#Second Frame Starts Here
#Terms And Conditions Page
f2.configure(bg='red')
Label(f2, text="TWEETY - Terms And Conditions", bg="white", fg="black",relief="solid").pack(side=TOP, fill=X, padx=2, pady=2)
text_area = st.ScrolledText(f2, width = 27,relief="solid",  height = 18,  font = ("Times New Roman", 15)) 
text_area.pack(pady=10) 
text_area.insert(INSERT, """This Application(Tweety) is 
Developed by Shubham Dubey. 
These terms and conditions 
govern your use to our app.
If you don't Accept this term 
and condition then you may stop
using the application.We reserve
the right to change these Terms 
and Conditions, at any time. It's 
your resposibity to regularly 
check those changes.If you are not
agree to any changes made in our 
application and our terms and 
conditions then you should stop 
using our application instantly.

This application is for research
purpose on Twitter Sentimental 
analysis.In no event I shall be 
liable for any claims, penalties,
loss, damage or expenses,direct 
or indirect loss, result accuracy,
consequential loss or damage, 
loss of profit or goodwill, loss of
data, loss arising from use or 
inability to use the application,
negligence, delict,if you are using
data and stats from the application
then its none of my responsibility.
Nothing in these Terms and 
Conditions shall exclude or limit 
our liability for death or personal 
injury caused by negligence or for 
any liability which cannot be 
excluded or limited under 
applicable law.
""")
text_area.configure(state ='disabled') 
#Exit button to redirect to thank you page
Button(f2, text='Exit',relief="solid", command=lambda:raise_frame(f4)).pack(side=BOTTOM,fill=X, padx=2, pady=2)
#Created a Agree and continue button to if clicked open the analysing window
Button(f2, text='Agree And Continue',relief="solid", command=lambda:raise_frame(f3)).pack(side=BOTTOM,fill=X, padx=2, pady=2)
#End of Second Frame


#Third Frame starts Here
#Analysing window
# Pre-Processing the Dataset tweets and Return the processed tweets
def process(data):
    temp = []
    for text in data['sampletweets']:
        text = pp.pre_processing(text)
        temp.append(text)
    data['sampletweets'] = temp
    return data['sampletweets']


# Main process of comparison of tweets and classification of tweets into positive or negative with diffrent machine learning algorithms
def execute():
    try:
        UserName = username.get()
        TweetNum = NumTweet.get()
        twitter_client = TwitterClient()
        api = twitter_client.get_twitter_client_api()
        tweets = api.user_timeline(screen_name=UserName, count=TweetNum)
        tweets_text = []
        for tweet in tweets:
            tweets_text.append(pp.pre_processing(tweet.text))
        datafile = pd.read_csv('SampleTrainingDataSmall.csv', sep=',', encoding="utf-8")
        x = process(datafile)
        y = datafile['label']

        #Split datafile into random train and test subsets using sklearn.model_selection.train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
        #calling CounVectorizer from sklearn.feature_extraction.text to convert both data files and tweets to a Sparse matrix of token counts
        vector = CountVectorizer()
        #Using Fit function to Learn a vocabulary dictionary of all tokens in the raw documents.
        vector.fit(x_train)
        #Using Transform function to Transform documents to document-term matrix.
        x_train_vec = vector.transform(x_train)
        x_test_vec = vector.transform(x_test)


        # Loop for the no. Of tweets to be analysed
        count = 1
        for tweet in tweets_text:
            #Created window for showing result of analysis
            resultwindow = Tk()
            resultwindow.geometry("440x275")
            resultwindow.configure(bg='red')
            Label(resultwindow, text="Tweet no."+str(count)+":- "+tweet, bg="red", fg="black").place(x = 0)
            count += 1
            #Converting tweet to an array
            tweet = [tweet]
            tweetvec = vector.transform(tweet)
            resultwindow.title("TWEETY-Analysis Result")
            Label(resultwindow, text="Algorithms", bg="red", font='bold', fg="black",width=39,relief="solid").place(x = 10,y = 20)
            Label(resultwindow, text="Result", bg="red", font='bold', fg="black",width=8,relief="solid").place(x = 349,y = 20)  
            
            #Created array to store results of algorithms
            temp = [0] * 10
        # Naive Bayes Algorithms

            # 1:-Calling Multinomial Naive Bayes Algorithm
            Label(resultwindow, text="Multinomial Naive Bayes:", bg="red", fg="black",width=50,relief="groove").place(x = 10,y = 45)
            temp[0] = MultinomialNBAlgo(x_train_vec, y_train, x_test_vec, y_test, tweetvec)
            Label(resultwindow, text=temp[0], bg="red", fg="black",width=10,relief="groove").place(x = 350,y = 45)  

            # 2:-Calling Bernoulli Naive Bayes Algorithm
            Label(resultwindow, text="Bernoulli Naive Bayes", bg="red", fg="black",width=50,relief="groove").place(x = 10,y = 65)  
            temp[1] = BernoulliNaiveBayesAlgo(x_train_vec, y_train, x_test_vec, y_test, tweetvec)
            Label(resultwindow, text=temp[1], bg="red", fg="black",width=10,relief="groove").place(x = 350,y = 65)

        # Linear Model Algorithms

            # 3:-Calling stochastic gradient descent Classifier Algorithm
            Label(resultwindow, text="stochastic gradient descent Classifier", bg="red", fg="black",width=50,relief="groove").place(x = 10,y = 85)  
            temp[2] = SGDClassifierAlgo(x_train_vec, y_train, x_test_vec, y_test, tweetvec)
            Label(resultwindow, text=temp[2], bg="red", fg="black",width=10,relief="groove").place(x = 350,y = 85)  

            # 4:-Calling Logistic regression Algorithm
            Label(resultwindow, text="Logistic regression", bg="red", fg="black",width=50,relief="groove").place(x = 10,y = 105)  
            temp[3] = LogisticRegressionAlgo(x_train_vec, y_train, x_test_vec, y_test, tweetvec)
            Label(resultwindow, text=temp[3], bg="red", fg="black",width=10,relief="groove").place(x = 350,y = 105)

        # Support Vector Machine Classification Algotithms      

            # 5:-Calling Support Vector Classifier Algorithm
            Label(resultwindow, text="Support Vector Classifier", bg="red", fg="black",width=50,relief="groove").place(x = 10,y = 125)  
            temp[4] = SupportVectorClassifierAlgo(x_train_vec, y_train, x_test_vec, y_test, tweetvec)
            Label(resultwindow, text=temp[4], bg="red", fg="black",width=10,relief="groove").place(x = 350,y = 125)  

            # 6:-Calling LinearSupportVectorClassifier Algorithm
            Label(resultwindow, text="LinearSupportVectorClassifier", bg="red", fg="black",width=50,relief="groove").place(x = 10,y = 145)  
            temp[5] = LinearSupportVectorClassifierAlgo(x_train_vec, y_train, x_test_vec, y_test, tweetvec)
            Label(resultwindow, text=temp[5], bg="red", fg="black",width=10,relief="groove").place(x = 350,y = 145) 
            
        # Decision Tree Algorithm

            # 7:-Calling Decision Tree Classifier Algorithm
            Label(resultwindow, text="Decision Tree Classifier", bg="red", fg="black",width=50,relief="groove").place(x = 10,y = 165)  
            temp[6] = DecisionTreeClassifierAlgo(x_train_vec, y_train, x_test_vec, y_test, tweetvec)
            Label(resultwindow, text=temp[6], bg="red", fg="black",width=10,relief="groove").place(x = 350,y = 165) 

        # Nearest Neighbors Classification Algorithms   

            # 8:- Calling kNearest Neighbors Classifier Algorithm
            Label(resultwindow, text="kNearest Neighbors Classifier Algorithm", bg="red", fg="black",width=50,relief="groove").place(x = 10,y = 185)  
            temp[7] = kNearestNeighborsClassifierAlgo(x_train_vec, y_train, x_test_vec, y_test, tweetvec)
            Label(resultwindow, text=temp[7], bg="red", fg="black",width=10,relief="groove").place(x = 350,y = 185)

            # 9:- Calling Nearest Centroid Classifier Algorithm
            Label(resultwindow, text="Nearest Centroid Classifier Algorithm", bg="red", fg="black",width=50,relief="groove").place(x = 10,y = 205)  
            temp[8] = NearestCentroidAlgo(x_train_vec, y_train, x_test_vec, y_test, tweetvec)
            Label(resultwindow, text=temp[8], bg="red", fg="black",width=10,relief="groove").place(x = 350,y = 205) 

        # Ensemble classification Algorithm

            # 10:-Calling Random Forest classifier Algorithm
            Label(resultwindow, text="Random Forest classifier ALgorithm", bg="red", fg="black",width=50,relief="groove").place(x = 10,y = 225)  
            temp[9] = RandomForestClassifierAlgo(x_train_vec, y_train, x_test_vec, y_test, tweetvec)
            Label(resultwindow, text=temp[9], bg="red", fg="black",width=10,relief="groove").place(x = 350,y = 225)  

            #counting the no. of all positives and negative From analysis of diffrent result
            CountNeg=0
            CountPos=0
            for x in temp:
                if x=="Positive":
                    CountPos +=1
                elif x == "Negative":
                    CountNeg +=1

            # Printing the final analysis result
            Label(resultwindow, text="Final Analysis", bg="red", font='bold', fg="black",width=39,relief="solid").place(x = 10,y = 247)

            if CountPos>CountNeg:
                Label(resultwindow, text="Postive", bg="red", font='bold', fg="black",width=8,relief="solid").place(x = 349,y = 247)
            elif CountPos<CountNeg:
                Label(resultwindow, text="Negative", bg="red", font='bold', fg="black",width=8,relief="solid").place(x = 349,y = 247)
            elif CountPos==CountNeg:
                Label(resultwindow, text="Neutral", bg="red", font='bold', fg="black",width=8,relief="solid").place(x = 349,y = 247)



    except Exception as exp:
        # Prints the error
        print(exp)

        # When rate limit reaches
        def ratelimit(self, track):
            # Print limited rate error
            print("Continuing, Limited rate")
            # Continue mining tweets
            return True
        # When timed out

        def timeout(self):
            # Print timeout message
            print(sys.stderr, 'Timeout')
            # Wait 10 seconds
            time.sleep(10)
            # Return nothing
            return


f3.configure(bg='red')
Label(f3, text="TWEETY - A SENTIMENTAL ANALYSIS TOOL", bg="white", fg="black",relief="solid").pack(side=TOP, fill=X, padx=2, pady=2)
Label(f3, image=img1, bg="red", fg="black").pack(side=TOP, fill=X)
Label(f3, text="TWITTER USERNAME", bg="white", fg="black",relief="solid").pack(side=TOP, fill=X, padx=2, pady=5)
username = Entry(f3,relief="solid")
username.pack(side=TOP, padx=2, pady=5)
Label(f3, text="NUMBER OF TWEETS(Only integer values)",relief="solid", bg="white", fg="black").pack(side=TOP, fill=X, padx=2, pady=5)
NumTweet = Entry(f3, relief="solid")
NumTweet.pack(side=TOP, padx=2, pady=5)
#Run button when clicked runs execute function
But1 = Button(f3, text="RUN",relief="solid", command=execute)
But1.pack(side=TOP, fill=X, padx=2, pady=2)
Button(f3, text='Exit',relief="solid", command=lambda:raise_frame(f4)).pack(side=TOP, fill=X, padx=2, pady=2)

#End of Third Frame


#Forth Frame Starts Here
f4.configure(bg='red')
Label(f4, text="TWEETY - A SENTIMENTAL ANALYSIS TOOL", bg="white", fg="black",relief="solid").pack(side=TOP, fill=X, padx=2, pady=2)
Label(f4, image=img1, bg="red", fg="black").pack(side=TOP, fill=X)
Img2 = ImageTk.PhotoImage(Image.open("ThankYou.png").resize((280, 145),Image.ANTIALIAS) )
Label(f4, image=Img2, bg="red", fg="black").pack(side=TOP, fill=X)
#Button creation to quit and close the window
Button(f4, text = 'Quit',relief="solid", command=root.destroy).pack(side=BOTTOM, fill=X, padx=2, pady=2)

#calling the main Frame and running main loop
raise_frame(f1)
root.mainloop()