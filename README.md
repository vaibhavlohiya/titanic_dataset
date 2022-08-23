# Titanic Dataset
Building a machine learning model for the famous titanic dataset using the several different classifiers. However, at the end we found that the Kernel SVM has the best accuracy which is about 83.05 % and a standard deviation of 3.31 %.

Now, this whole process can be devided into 4 parts, which are :-

1. Importing.
2. Data preprocessing
3. Model selection and training 
4. Accuracy measurement and saving final results.

## 1. Importing.

This is the most easiest and least time consuming part, here we just have to import the basic libraries to handle the data and to build our machine learning models. After, doing that we have to import our training and testing dataset. 

We can do this buy reading the '.csv' files using pandas.

    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

## 2. Data preprocessing

This is one of the most important and crucial part in the process of building a machine learning model. As the name suggest, Data pre-processing is something which is done on the raw form of data that we have and filter it so that the data would make sense when we process it. Pre-processing can be divided into 4 main stages which are as follows.

1. Taking care of missing data.
2. Encoding the data.
3. Spliting the data.
4. Feature scaling.

### 1. Taking care of missing data

Data aquisition is tough and most of the time you might not get every single feature of every entry. In other words, you always have to deal with missing data and there are several numpy and pandas functions that can help you. For example- you observe that the 'Age' feature only contains 1046 non-null elements but the total number of entries are 1309. So, to fill the rest of the missing entries you can something like this

    # Filling the missing data of Age column with the median of Age
    titanic.Age = titanic.Age.fillna(titanic.Age.median())

Here, titanic is a pandas dataframe and we are using the median of the 'Age' feature to fill the rest. We have used the 'fillna()' to fill the empty values.

### 2. Encoding te data.

No matter how much processing power you have on a computer or how many hours you spent to build an intelligent model, the cold truth is that computers only deal with numbers. Weather it is comparison between two elements or doing some logical calculations. It is always the numerical data that take participation in this process and not the categorical data (Non-numerical data such as names, address, job-type, gender and so on). To deal with this we have two solutions, either we make dummy columns for each category in a non-numerical feature like this
    
    # Making dummy variables for Pclass, Embarked
    pclass_dummies = pd.get_dummies(titanic.Pclass, prefix = 'Pclass')
    embarked_dummies = pd.get_dummies(titanic.Embarked, prefix = 'Embarked')

or we can label encode the feature when there are just two categories like sex (male or female).

    # Label Encoder for the Sex column 
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    titanic.Sex = le.fit_transform(titanic.Sex)

We can also do one hot encoding instead of making dummy columns but it is really situation dependend and we won't discuss it here.

### 3. Splitting the data 

Splitting the data is pretty simple, you decide the portion based on the number of entries and other properties and then you split the data into two parts which are, the training set and the test set. You might be wondering, "why are we doing this?". 

Well, for any machine learning model to perform well on our set standards. First, it has to learn and it can do that by learning from the training set which has to be bigger than the test set. To give you an idea, the test set is usually 1/4 or 1/5 of the whole data set, the rest is training data. After we feed our model with large amounts of training data, we say that our model is trained and ready to predict some results and to do that we use the test set for that. Where the final result is often denoted as 'y_pred'. 

You can split the data by using these functions.

    data_train = dataset[ :train_idx]
    data_test = dataset[train_idx: ]    

Here, the 'dataset' is the total dataset after encoding and 'train_idx' is the number of training data entries can be decided earlier at the starting of the program or later. 

### 4. Feature Scaling 

Now, in the last part of data pre-processing what we try to achieve is the same order of scaling for every feature. For example, suppose we have features like 'Age' and 'Height'. Both being a type of numerical data, makes it difficult for the model when it comes to comparison. So to overcome that, we feature scale every element of the feature set (X_train, X_test) so that the model can compare them for it's learning.  

You can do feature scaling by using following functions.

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    

## 3. Model selection and training 

Model selection depends on certain properties of the model. First being the most accurate compare to the rest. Second is the type of parameters they take.

Now, it is true that some models are powerful than others because they use more computing power and more complex methods to learn. At the end of the day, it boils down to to type of data you are feeding it, if it is a simple dataset with fewer than 5 features and fewer than 500 entries. Then you can get away by using just the "Logistic Regression Model".

But, I can't really say otherwise.

Here are some of the Machine learning models that we have used on this 'Titanic' dataset

### Logistic Regression Model

    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

### Random Forest Model

    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators = 100, max_depth = 5, random_state = 1)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    
### Kernel SVM Model

    from sklearn.svm import SVC
    classifier = SVC(kernel = 'rbf', random_state = 0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    
## 4. Accuracy measurement and saving final results.

Finally, in the last part after training our model, what we do is find the accuracy of each model and cross check to confirm the highest accuracy and which model has achived it. We do this by this set of commands.

    from sklearn.model_selection import cross_val_score
    accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
    print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
    print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
    
These set of commands are also known as the 'k-fold cross validation' 

In our case of 'titanic' dataset. We found out that the 'Kernel SVM Model' gave the best accuracy of '83.05 %' . So, what we do now is we save the 'y_pred' or 'Survived' column of Kernel SVM and we save it in a final adjacent to the 'PassengerID' to know whether the passenger has surivied or not according to the model.

We can create a '.csv' file of this sort by using the following funtions.

    output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': y_pred})
    output.to_csv('submission.csv', index=False)
    
You can find the submission.csv file in the repository. 
