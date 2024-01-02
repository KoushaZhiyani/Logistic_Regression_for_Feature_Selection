# counts best number of Feature and select Feature


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

while True:
    status = input("1)start\n0)Exit!\n")
    if status == "0":
        exit()
    # Address your csv file
    address = input("address : \n")
    # select Target columns
    target = input("target : \n")
    # A maximum of columns ( Feature )should be checked
    columns_num = int(input("number of columns : \n"))
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
        X = data[columns[:i + 1]]
        y = data[target]
        # selecting different samples group
        for j in range(0, 100):
            model = LogisticRegression(max_iter=2000)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=j)
            model.fit(X_train, y_train)
            # print(model.feature_names_in_)
            y_pred = model.predict(X_test)
            score.append(model.score(X_test, y_test))
        avg_score.append(sum(score) / 100)

    # find Features from max score
    highest_scored = max(avg_score)
    highest_scored_index = avg_score.index(highest_scored)
    print(print(round(np.mean(max(avg_score)), 3), round(np.std(max(avg_score)),3)))
    print("Features :")
    print(columns[:highest_scored_index + 1])
    # Data visualization
    plt.scatter(range(1, columns_num + 1), avg_score, c="green")
    plt.xlabel("number of columns")
    plt.ylabel("score")
    plt.show()
