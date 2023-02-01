## Supreme Court Judgement Prediction
This project attempts to predict the outcome of Supreme Court Cases using natural language processing and a convolutional neural network. We cleaned the data by removing rows with blank values and adding justice ideologies to the data to account for potential ideological biases when predicting case outcomes. After some exploratory data analysis, we processed the facts through tokenization and produced word vectors, allowing us to put them through our model.

Ultimately, our model had an accuracy of about 65%, which isn't too great considering that roughly that same number of cases all resulted in the first-party winning. We think this is due to the nature of our data and the nature of the Supreme Court. First, our data had only aboud 3k cases, and the facts blurb was usually only a paragraph long. If we wanted to make a more accurate model in the future, we would likely seek to find a much larger data set and take a different approach regarding which text we analyze. Additionally, the Supreme Court in general is not very predictable, with there constantly being close decisions on crucial topics. This too likely contributed to why our model was not able to perform well.

In the end, future projects can expand on this work by implementing a larger data set, examining more text, and potentially using a pre-trained model, such as BERT.

This project was submitted to the 2022 DATA Club Expo.

## Contributors
Aayush Turakhia, David O’Brien, and Rudra Sett

## Sources
Supreme Court Data Set: https://www.kaggle.com/deepcontractor/supreme-court-judgment-prediction
 
