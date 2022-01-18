# ACM Research Coding Challenge (Spring 2022)

## Sources
Types of Binary Classification: https://www.learndatasci.com/glossary/binary-classification/#APythonExampleforBinaryClassification

Choose what estimator to use: https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html

Label Encoder: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html

SVC: https://scikit-learn.org/stable/tutorial/basic/tutorial.html#learning-and-predicting

SVC: https://stackoverflow.com/questions/38584829/svc-support-vector-classification-with-categorical-string-data-as-labels


## My Method

After I read the problem statement, I wanted to see which estimator works best for classification and after coming across the source listed above (Choose what estimator to use), I decided to stick with SVC Linear. However, I ran into problems since Strings would not work as they weren't float64 (which were the datatype that was functional with the classifier). As a result, I researched on the different methods of converting strings into float64 and I came across OneHotEncoder and Label Encoder. After trial and error, I decided to stick with LabelEncoder since it worked well with SVC Linear. After a couple of tests, I noticed that the average of accuracies was around 0.95 (as shown in the image below), which is pretty high. After that, I tried Logistic Regression and other classifiers, only to realize that SVC was the most convenient out of all the classifiers. If it was not randomized, logistic regression had the highest accuracy than the other classifiers from the specific training size I used.


## Image
![avg_accuracy](https://user-images.githubusercontent.com/31449837/149862687-59dd211b-d4f3-43dc-b856-f0b70f0433b5.png)
