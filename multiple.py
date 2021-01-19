import pandas as pd


df=pd.read_csv('cwurData.csv')


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle



x_train, x_test, y_train, y_test= train_test_split(df[['score',
       'quality_of_education', 'alumni_employment', 'quality_of_faculty',
       'publications', 'influence', 'citations','patents']],df[['world_rank']], test_size=0.3, random_state=109)



linear=LinearRegression()
linear.fit(x_train, y_train)
y_pred= linear.predict(x_test)


pickle_out= open("lin_mult.pkl", "wb")
pickle.dump(linear, pickle_out)
pickle_out.close()








