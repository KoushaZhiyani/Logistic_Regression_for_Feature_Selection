# counts best number of Feature and select Feature


import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# address and target
# C:\Users\Ashil-Rayan\PycharmProjects\PCOS\PCOS_edit.csv
# PCOS (Y/N)


while True:
    status = input("1)start\n0)Exit!")
    if status == "0":
        exit()
    # Address your csv file
    address = input("address : \n")
    # select Target columns
    target = input("target : \n")
    # A maximum of columns ( Feature )should be checked
    columns_num = int(input("number of columns"))
    data = pd.read_csv(address)
    # find best Features
    corr = data.corr()[target].sort_values(ascending=False)
    corr = list(corr.index)
    corr.remove(target)
    # Variable for columns ( Feature )
    columns = []
    columns = corr[0:columns_num]
    # Highest score
    all_sets = []
    # Variable for average score after 100 times
    avg_score = []
    for i in range(1, columns_num + 1):
        score = []
        data = pd.read_csv(address)
        X = data[columns]
        y = data[target]
        model = LogisticRegression()
        rfe = RFE(model, n_features_to_select=i)
        rfe.fit(X, y)
        # Variable, show  which columns selected
        sets = rfe.support_
        all_sets.append(sets)
        print(sets)
        for j in range(0, columns_num):
            if sets[j] == False:
                X = X.drop(columns[j], axis=1)
        # selecting different samples group
        for j in range(0, 100):
            model = LogisticRegression(max_iter=160)
            X_train, X_test, y_train, y_tset = train_test_split(X, y, test_size=0.2, random_state=j)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score.append(model.score(X_test, y_tset))

        avg_score.append(sum(score) / 100)
    # find Features from max score
    highest_scored = max(avg_score)
    highest_scored_index = avg_score.index(highest_scored)
    c = 0
    print("Features :")
    for j in all_sets[highest_scored_index]:
        if j:
            print(columns[c])
        c = c + 1
    # Data visualization
    plt.scatter(range(1, columns_num + 1), avg_score, c="green")
    plt.xlabel("number of columns")
    plt.ylabel("score")
    plt.show()
    exit()
