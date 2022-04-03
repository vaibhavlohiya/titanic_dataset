# Titanic Dataset
Building a machine learning model for the famous titanic dataset using the several different classifiers. However, at the end we found that the Kernel SVM has the best accuracy which is about 83.05 % and a standard deviation of 3.31 %.

Now, this whole process can be devided into 4 parts, which are :-

1. Importing.
2. Data preprocessing
3. Model selection and training 
4. Accuracy measurement and saving final results.

## 1. Importing.

This is the most easiest and less time consuming part, here we just have to import the basic libraries to handle the data and to build our machine learning models. After, doing that we have to import our training and testing dataset. 

Now, after importing the dataset and putting it into a pandas dataframe it would look somewhat like this 

      "text/plain": [
       "   PassengerId  Survived  Pclass  \\\n",
       "0            1       0.0       3   \n",
       "1            2       1.0       1   \n",
       "2            3       1.0       3   \n",
       "3            4       1.0       1   \n",
       "4            5       0.0       3   \n",
       "\n",
       "                                                Name     Sex   Age  SibSp  \\\n",
       "0                            Braund, Mr. Owen Harris    male  22.0      1   \n",
       "1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
       "2                             Heikkinen, Miss. Laina  female  26.0      0   \n",
       "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   \n",
       "4                           Allen, Mr. William Henry    male  35.0      0   \n",
       "\n",
       "   Parch            Ticket     Fare Cabin Embarked  \n",
       "0      0         A/5 21171   7.2500   NaN        S  \n",
       "1      0          PC 17599  71.2833   C85        C  \n",
       "2      0  STON/O2. 3101282   7.9250   NaN        S  \n",
       "3      0            113803  53.1000  C123        S  \n",
       "4      0            373450   8.0500   NaN        S  "
      ]

## 2. Data preprocessing

## 3. Model selection and training 

## 4. Accuracy measurement and saving final results.
