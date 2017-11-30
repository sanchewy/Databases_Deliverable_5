import mysql.connector as sql
import numpy as np
import scipy.stats as spicy
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import normalize

class main():
	def __init__(self):
		pass
		#do something here

	def solveQuestions(self):
		q = query()
		v = visualize()
		#solve question 1 (spearman correlation coefficient between movie runtime and movie revenue)
		duration_gross = np.asarray(q.querydb(1,0))
		duration, gross = zip(*q.querydb(1,0))
		duration = np.asarray(duration)
		gross = np.asarray(gross)
		print(spicy.spearmanr(duration_gross))
		v.scatter(duration, gross, 'Duration', 'Gross')

		#solve question 2

		#solve question 3
		#Normalize data from database.
		attr = normalize(np.asarray(q.querydb(3,0)[0]))
		target = normalize(np.asarray(q.querydb(3,0)[1]))
		# print(attr, target)
		model = Sequential()
		model.add(Dense(10, input_dim=5, activation='relu'))
		model.add(Dense(10, activation='relu'))
		model.add(Dense(1, activation='linear'))
		#Compile model for efficient use of tensorflow underlying.
		model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
		#Training the model, tune batch size and epochs.
		model.fit(attr, target, epochs=20, batch_size=10)
		#Evaluate our model
		scores = model.evaluate(attr, target)
		print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

		#Solve question 4

		#Solve question 5

		#visualize.barGraph(analyze.spearman(query.querydb(1, False)))

	def disconnectDatabase(self):
		q = query()
		q.querydb('whatever', 1)

	#Normalizes data
	def normalize(self, v):
		norm=np.linalg.norm(v, ord=1)
		if norm==0:
			norm=np.finfo(v.dtype).eps
		return v/norm

class query():
	def __init__(self):
		config = {
			'user' : 'root',		#username here (*)
			'password' : 'mushroom',		#password here (*)
			'host' : '127.0.0.1',	#connect on local host
			'database' : 'MovieDataModel',		#db name here (*)
			'raise_on_warnings' : True,
		}

		#link to our databse
		global cnx, cursor
	 	cnx = sql.connect(**config)
		cursor = cnx.cursor()

	def querydb(self, num, done):
		if done == True:
			#if we are done, disconnect from the database
			cursor.close()
			cnx.close()

		if num == 1:
			a = []
			#query our database for q1 data
			query1 = ("SELECT duration, gross "
					  "FROM Movie;")
			cursor.execute(query1)
			for title in cursor:
				a.append(title)
			return a
				#do something with the data (*)
		elif num == 2:
			a = []
			#query our database for q2 data
			query2 = ()
			cursor.execute(query2)
			for title in cursor:
				a.append(title)
				#do something with the data (*)
			return a
		elif num == 3:
			attr = []
			target = []
			#query our database for q3 data
			query3 = ()
			cursor.execute("SELECT actor_1_facebook_likes, actor_2_facebook_likes, actor_3_facebook_likes, director_facebook_likes, duration, gross"
							" FROM Movie;")
			for title in cursor:
				attr.append(title[:len(title)-1])
				target.append(title[len(title)-1:])
			return(attr,target)
		elif num == 4:
			#query our database for q4 data
			query4 = ()
			cursor.execute(query4)
			for x in cursor:
				pass
				#do something with the data (*)
		else:
			#query our database for q4 data
			query5 = ()
			cursor.execute(query5)
			for x in cursor:
				pass
				#do something with the data (*)

class analyze():
	# have some different analysis functions
	def spearman(data):
		pass
		#an example of a analysis function

class visualize():
	# have some different analysis functions
	def scatter(self, x, y, xaxis, yaxis):
		# Create data
		N = 500
		colors = (0,0,40)
		print("%s max = %d, %s max = %d" % (xaxis, np.amax(x), yaxis, np.amax(y)))

		# Plot
		plt.scatter(x, y)
		plt.title('Correlation between movie %s and %s' % (xaxis, yaxis))
		plt.xlabel(xaxis)
		plt.ylabel(yaxis)
		plt.show()

if __name__ == '__main__':
	main = main()
	main.solveQuestions()
	main.disconnectDatabase()
