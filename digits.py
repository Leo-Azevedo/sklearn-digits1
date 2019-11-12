#### Import datasets and some other important stuff ####
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn import svm, neighbors, ensemble

#### Initialize Digits ####
digits = load_digits()
expected = digits.target[1::2] #Expected Result (Target)

#### SVC ####
clf = svm.SVC(gamma=0.001, C=100.) 
clf.fit(digits.data[::2], digits.target[::2])

results1 = clf.predict(digits.data[1::2])
score1 = clf.score(digits.data[1::2], digits.target[1::2])

#### KNeighbors ####
clf2 = neighbors.KNeighborsClassifier()
clf2.fit(digits.data[::2], digits.target[::2])

results2 = clf2.predict(digits.data[1::2])
score2 = clf2.score(digits.data[1::2], digits.target[1::2])

#### Ensemble ####
clf3 = ensemble.RandomForestClassifier()
clf3.fit(digits.data[::2], digits.target[::2])

results3 = clf3.predict(digits.data[1::2])
score3 = clf3.score(digits.data[1::2], digits.target[1::2])

names = ["SVC", "KNeighbors", "Ensemble"]
values = [score1, score2, score3]

print(values) #Debug print. Prints each classifier's score

plt.bar(names, values)
plt.title("Comparando Algoritmos")
plt.show()
