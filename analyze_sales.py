import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


url = "https://community.watsonanalytics.com/wp-content/uploads/2015/04/WA_Fn-UseC_-Sales-Win-Loss.csv"
pd.set_option('display.expand_frame_repr', False)
sales_data = pd.read_csv(url)

# print(sales_data.head())  # view the first few records of the data set

le = preprocessing.LabelEncoder()  # create label encoder

#convert the categorical columns into numeric
sales_data['Supplies Subgroup'] = le.fit_transform(sales_data['Supplies Subgroup'])
sales_data['Region'] = le.fit_transform(sales_data['Region'])
sales_data['Route To Market'] = le.fit_transform(sales_data['Route To Market'])
sales_data['Opportunity Result'] = le.fit_transform(sales_data['Opportunity Result'])
sales_data['Competitor Type'] = le.fit_transform(sales_data['Competitor Type'])
sales_data['Supplies Group'] = le.fit_transform(sales_data['Supplies Group'])

# select columns other than 'Opportunity Number','Opportunity Result'
cols = [col for col in sales_data.columns if col not in ['Opportunity Number','Opportunity Result']]
# split into the training and testing sets
data = sales_data[cols]
target = sales_data['Opportunity Result']
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.30, random_state=10)

# train on a model
# gnb = GaussianNB()
# pred = gnb.fit(data_train, target_train).predict(data_test)
# print("Gaussian Naive-Bayes result (accuracy score): ", accuracy_score(target_test, pred, normalize = True))

knc = KNeighborsClassifier(n_neighbors=3)
#Train the algorithm
knc.fit(data_train, target_train)
#predict the response
pred = knc.predict(data_test)
#evaluate accuracy
print("KNeighbors accuracy score: ", accuracy_score(target_test, pred))  # 85%

# feed test data 
new_data_instance = ["Exterior Accessories", "Car Accessories", "Northwest", "Fields Sales", 76, 13, 104, 101, 0, 5, 5, 0, "Unknown", 0.69636, 0.113985, 0.154215, 1]

new_data_instance = le.fit_transform(new_data_instance)

inference = knc.predict([new_data_instance])
if inference[0] == 0:
    print("Prediction: this is going to be a LOSS.")
elif inference[0] == 1:
    print("Prediction: this is going to be a WIN.")
else:
    print(inference)
