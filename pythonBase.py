import mysql.connector as sql
import numpy as np
import scipy.stats as spicy
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors.kde import KernelDensity
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import normalize
import time

class main():

	#Initialization function for the main method (not used)
	def __init__(self):
		pass

	#Main driving function. Uses Query class and Visualize class to answer our 5 questions.
	def solveQuestions(self):

		#Class instantiations
		q = query()
		v = visualize()

		#Solve question 1 (spearman correlation coefficient between movie runtime and movie revenue)
		#Version without data cleaning
		duration_gross = np.asarray(q.querydb(1,0))
		duration, gross = zip(*q.querydb(1,0))
		duration = np.asarray(duration)
		gross = np.asarray(gross)
		#Calculate coefficient and p-value
		print("\nSpearman Correlation Coefficient Without Data Cleaning: %.8f, P-Value: %.2f\n" % (spicy.spearmanr(duration_gross)[0],spicy.spearmanr(duration_gross)[1]))
		#Scatter plot of the data
		v.scatter(duration, gross, 'Duration dirty', 'Gross dirty')

		#Version with data cleaning
		duration_gross = np.asarray(q.querydb(1,0))
		print("Size duration_gross before cleaning"+str(duration_gross.size))
		duration_gross = duration_gross[np.all(duration_gross != 0, axis=1)]
		print("Size duration_gross after cleaning"+str(np.size(duration_gross)))
		duration = duration_gross.T[0]
		gross = duration_gross.T[1]
		#Calculate coefficient and p-value
		print("\nSpearman Correlation Coefficient With Data Cleaning: %.8f, P-Value: %.2f\n" % (spicy.spearmanr(duration_gross)[0],spicy.spearmanr(duration_gross)[1]))
		#Scatter plot of the data
		v.scatter(duration, gross, 'Duration clean', 'Gross clean')

		#Solve question 2

		#Solve question 3 (Neural network predicting revenue from facebook likes)
		#5 fold cross validation is used
		num_folds = 5
		#Normalize data from database (both the input vectors and the target output vector)
		attr = normalize(np.asarray(q.querydb(3,0)[0]))
		target = normalize(np.asarray(q.querydb(3,0)[1]))
		#Create model (2 hidden layer neural network relu activation on the hidden neurons, 1 linear output neuron)
		model = Sequential()
		model.add(Dense(10, input_dim=5, activation='relu'))
		model.add(Dense(10, activation='relu'))
		model.add(Dense(1, activation='linear'))
		#Compile model for efficient use of underlying tensorflow libraries.
		model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
		fold_accuracy = [] #Holds the accuracy results of each of the 5 folds.
		#Train network on the training set, evaluate it on the test set, record accuracy.
		for x in range(num_folds):
			print("%d fold cross validation on fold: %d" % (num_folds,x+1))
			start_bound = len(attr)/num_folds * x
			end_bound = start_bound + len(attr)/num_folds
			#Training the model, tune batch size and epochs.
			model.fit(np.concatenate((attr[:start_bound], attr[end_bound:len(attr)]), axis=0), np.concatenate((target[:start_bound], target[end_bound:len(target)]), axis=0), epochs=20, batch_size=10)
			#Evaluate our model using mean squared error as the accuracy measure.
			scores = model.evaluate(attr[start_bound:end_bound], target[start_bound:end_bound])
			fold_accuracy.append(scores[1]*100)
			print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
		print("Final accuracy over all %d folds = %.2f%%" %(num_folds, sum(fold_accuracy)/float(num_folds)))

		#Solve question 4 (Spearman correlation between Movie budget and revenue).
		#Version without data cleaning
		budget_gross = np.asarray(q.querydb(4,0))
		budget, gross = zip(*q.querydb(4,0))
		budget = np.asarray(budget)
		gross = np.asarray(gross)
		#Calculate coefficient and p-value
		print("\nSpearman Correlation Coefficient Without Data Cleaning: %.8f, P-Value: %.2f\n" % (spicy.spearmanr(budget_gross)[0],spicy.spearmanr(budget_gross)[1]))
		#Scatter plot of the data
		v.scatter(budget, gross, 'Budget dirty', 'Gross dirty')

		#Version with data cleaning
		budget_gross = np.asarray(q.querydb(4,0))
		print("Size budget_gross before cleaning"+str(np.size(budget_gross)))
		budget_gross = budget_gross[np.all(budget_gross != 0, axis=1)]
		print("Size budget_gross after cleaning"+str(np.size(budget_gross)))
		budget = budget_gross.T[0]
		gross = budget_gross.T[1]
		#Calculate coefficient and p-value
		print("\nSpearman Correlation Coefficient With Data Cleaning: %.8f, P-Value: %.2f\n" % (spicy.spearmanr(duration_gross)[0],spicy.spearmanr(duration_gross)[1]))
		#Scatter plot of the data
		v.scatter(budget, gross, 'Budget clean', 'Gross clean')

		#Solve question 5 (Probability distribution number of critic reviews / duration)
		#Duration was way more interesting than critic reviews.
		num_reviews = np.asarray(q.querydb(5,0))
		N = 100
		np.random.seed(1)
		X_plot = np.linspace(0, max(num_reviews), len(num_reviews))[:, np.newaxis]
		fig, ax = plt.subplots()

		#Plot the fit of each kernel over the extracted data.
		#This is for the density kernel estimator (obviously)
		for kernel in ['gaussian','tophat', 'epanechnikov']:
		    kde = KernelDensity(kernel=kernel, bandwidth=5).fit(num_reviews)
		    log_dens = kde.score_samples(X_plot)
		    ax.plot(X_plot[:, 0], np.exp(log_dens),'-',
		            label="kernel = '{0}'".format(kernel), linewidth=1)

		#Plot the individual point values below the axis of the graph.
		ax.text(500, 0.2, "N={0} points".format(N))
		#Legend
		ax.legend(loc='upper right')
		ax.plot(num_reviews[:, 0], -0.005 - 0.01 * np.random.random(num_reviews.shape[0]), '+k')
		#Set axis values for the graph
		ax.set_xlim(-100,max(num_reviews))
		ax.set_ylim(-0.02, 0.03)
		plt.show()

		#This is for the plain old histogram with no kernel density estimation.
		fig, ax = plt.subplots()
		fig.subplots_adjust(hspace=0.05, wspace=0.05)
		#Tune number of bins for the histogram.
		bins = np.linspace(0,max(num_reviews), 100)
		ax.hist(num_reviews[:, 0], bins=bins, fc='#AAAAFF', normed=True)
		ax.text(-3.5, 0.31, "Histogram")
		#Plot the individual data points below the axis of the graph.
		ax.plot(num_reviews[:, 0], np.zeros(num_reviews.shape[0]) - 0.001, '+k')
		ax.set_xlim(-100, max(num_reviews))
		ax.set_ylim(-0.005, 0.03)
		plt.show()

	#Sever the connection with the database.
	def disconnectDatabase(self):
		q = query()
		q.querydb('whatever', True)

	#Normalizes data (linearly) using the numpy libraries.
	def normalize(self, v):
		norm=np.linalg.norm(v, ord=1)
		if norm==0:
			norm=np.finfo(v.dtype).eps
		return v/norm

#Class which manages queries to the MySql database.
class query():

	#Configuration for the database name, username, and password.
	def __init__(self):
		config = {
			'user' : 'root',		#username here (*)
			'password' : 'mushroom',		#password here (*)
			'host' : '127.0.0.1',	#connect on local host
			'database' : 'MovieDataModel',		#db name here (*)
			'raise_on_warnings' : True,
		}
		#Begin connection to our databse.
		global cnx, cursor
	 	cnx = sql.connect(**config)
		cursor = cnx.cursor()

	#Takes in the query number (1-5 for each question) and a "done" indactor which ends the database connection.
	def querydb(self, num, done):

		#If we are done, disconnect from the database
		if done == True:
			cursor.close()
			cnx.close()
			return

		#Query for question 1.
		if num == 1:
			a = []
			query1 = ("SELECT duration, gross "
					  "FROM Movie;")
			cursor.execute(query1)
			for title in cursor:
				a.append(title)
			return a

		#Query for question 2.
		elif num == 2:
			a = []
			query2 = ()
			cursor.execute(query2)
			for title in cursor:
				a.append(title)
			return a

		#Query for question 3.
		elif num == 3:
			attr = []
			target = []
			query3 = ()
			cursor.execute("SELECT actor_1_facebook_likes, actor_2_facebook_likes, actor_3_facebook_likes, director_facebook_likes, duration, gross"
							" FROM Movie;")
			for title in cursor:
				attr.append(title[:len(title)-1])
				target.append(title[len(title)-1:])
			return(attr,target)

		#Query for question 4.
		elif num == 4:
			a = []
			query4 = ("SELECT budget, gross "
					  "FROM Movie;")
			cursor.execute(query4)
			for title in cursor:
				a.append(title)
			return a

		#Query for question 5.
		elif num == 5:
			a = []
			query5 = ("SELECT duration "
						"FROM Movie;")
			cursor.execute(query5)
			for title in cursor:
				a.append(title)
			return a

		#Query number not recognized.
		else:
			print("Error: Unrecognized query number.")

#Visualization class.
class visualize():

	#Graphs a scatter plot.
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

#Main script call function.
if __name__ == '__main__':
	main = main()
	main.solveQuestions()
	main.disconnectDatabase()
