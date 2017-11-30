import mysql.connector as sql
import numpy as np
import scipy.stats as spicy
import matplotlib.pyplot as plt

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
		v.scatter(duration, gross)

		#solve question 2

		#visualize.barGraph(analyze.spearman(query.querydb(1, False)))

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
		if num == 1:
			a = []
			#query our database for q1 data
			query1 = ("SELECT duration, gross "
					  "FROM Movie")
			cursor.execute(query1)
			for title in cursor:
				a.append(title)
			return a
				#do something with the data (*)
		elif num == 2:
			#query our database for q2 data
			query2 = ()
			cursor.execute(query2)
			for x in cursor:
				pass
				#do something with the data (*)
		elif num == 3:
			#query our database for q3 data
			query3 = ()
			cursor.execute(query3)
			for x in cursor:
				pass
				#do something with the data (*)
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
		if done == True:
			#if we are done, disconnect from the database
			cursor.close()
			cnx.close()

class analyze():
	# have some different analysis functions
	def spearman(data):
		pass
		#an example of a analysis function

class visualize():
	# have some different analysis functions
	def scatter(self,x,y):
		# Create data
		N = 500
		colors = (0,0,40)
		print(np.amax(x),np.amax(y))
		area = max(np.amax(x),np.amax(y))

		# Plot
		plt.scatter(x, y)
		plt.title('Correlation between movie duration and revenue')
		plt.xlabel('duration')
		plt.ylabel('gross')
		plt.show()
		#an example of a analysis function

if __name__ == '__main__':
	main = main()
	main.solveQuestions()
