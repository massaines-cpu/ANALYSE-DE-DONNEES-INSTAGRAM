import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns

X = insta_data[['perceived_stress_score']]
y = insta_data['daily_active_minutes_instagram']
plt.scatter(X, y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=1/3, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X_train, y_train)
print(regressor.coef_)
print(regressor.intercept_)

ordonnee = np.linspace(0, 50, 1000)
plt.scatter(X, y)
plt.plot(ordonnee, regressor.coef_[0]*ordonnee+regressor.intercept_, color='red')
plt.xlabel('Stress')
plt.ylabel('Minutes instagram')

y_predict = regressor.predict(X_test)

from sklearn import metrics
print('mean absolute error:', metrics.mean_absolute_error(y_test, y_predict))
print('mean squared error:', metrics.mean_squared_error(y_test, y_predict))
print('racine de mean squared error:', np.sqrt(metrics.mean_squared_error(y_test, y_predict)))
print('racine au carré:', metrics.r2_score(y_test, y_predict))

nouvelles_infos = [30, 250, 700]
nouveau_test = pd.DataFrame(nouvelles_infos, columns=['daily_active_minutes_instagram'])
y_nouveau = regressor.predict(nouveau_test)
y_nouveau

--------------------------------------
#entrainement modele
X = insta_data[['daily_active_minutes_instagram', 'reels_watched_per_day', 'stories_viewed_per_day', 'age']]
y = insta_data['age']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=1/3, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print(regressor.coef_)
print(regressor.intercept_)

ordonnee = np.linspace(0, 600, 1000)
plt.scatter(X, y)
plt.plot(ordonnee, regressor.coef_[0]*ordonnee+regressor.intercept_, color='red')
plt.xlabel('TEMPS INSTA')
plt.ylabel('AGE')