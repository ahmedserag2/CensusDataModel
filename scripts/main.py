import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.cluster import KMeans
from numpy.random import randint
from numpy.random import rand

path = "../adult.data.csv"
df = pd.read_csv(path)



def one_hot_encoding(df):
    #one hot encoding
    df_predictors = df.loc[:, df.columns != 'salary']
    #df_predictors
    #pd.get_dummies(df['race']).head()
    
    #df_predictors.concat(pd.get_dummies(df['race']) ,axis = 1)
    df_merged = pd.concat([df_predictors, pd.get_dummies(df['race'])], axis=1)
    df_merged = df_merged.drop('race' ,axis = 1)
    
    
    df_merged = pd.concat([df_merged, pd.get_dummies(df['education'])], axis=1)
    df_merged = df_merged.drop('education', axis = 1)
    
    
    df_merged = pd.concat([df_merged, pd.get_dummies(df['marital-status'])], axis=1)
    df_merged = df_merged.drop('marital-status', axis = 1)
    
    
    df_merged = pd.concat([df_merged, pd.get_dummies(df['sex'])], axis=1)
    df_merged = df_merged.drop('sex', axis = 1)
    
    df_merged = pd.concat([df_merged, pd.get_dummies(df['relationship'])], axis=1)
    df_merged = df_merged.drop('relationship', axis = 1)
    
    
    df_merged = pd.concat([df_merged, pd.get_dummies(df['occupation'])], axis=1)
    df_merged = df_merged.drop('occupation', axis = 1)
    
    df_merged = pd.concat([df_merged, pd.get_dummies(df['workclass'])], axis=1)
    df_merged = df_merged.drop('workclass', axis = 1)
    
    
    df_merged = pd.concat([df_merged, pd.get_dummies(df['native-country'])], axis=1)
    df_merged = df_merged.drop('native-country', axis = 1)
    return df_merged


df_merged = one_hot_encoding(df)

def knn(neigbors):
    X_train, X_test, y_train, y_test = train_test_split(df_merged, df['salary'], test_size=0.33, random_state=42)
    # Create KNN classifier
    knn = KNeighborsClassifier(n_neighbors = neigbors)
    # Fit the classifier to the data
    knn.fit(X_train,y_train)
    return knn.score(X_test, y_test)

def decision_tree(max_depth):
    X_train, X_test, y_train, y_test = train_test_split(df_merged, df['salary'], test_size=0.66, random_state=30)
    dt = DecisionTreeClassifier(max_depth=max_depth)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    
def k_means(no_clusters):
    kmeans = KMeans(n_clusters=no_clusters, random_state=0).fit(df_merged)
    return kmeans.inertia_

print(k_means(3))

















#genetic algorithm 

def onemax(x):
	#print(np.asarray(x) > 0)
#	df_merged[df_merged.columns[np.asarray(x) > 0]]
	X_train, X_test, y_train, y_test = train_test_split(df_merged[df_merged.columns[np.asarray(x) > 0]], df['salary'], test_size=0.66)
    # Create KNN classifier
	knn = KNeighborsClassifier(n_neighbors = 7)
# Fit the classifier to the data
	knn.fit(X_train,y_train)
	return -1 *knn.score(X_test, y_test)

# tournament selection
def selection(pop, scores, k=3):
	# first random selection
	selection_ix = randint(len(pop))
	for ix in randint(0, len(pop), k-1):
		# check if better (e.g. perform a tournament)
		if scores[ix] < scores[selection_ix]:
			selection_ix = ix
	return pop[selection_ix]

# crossover two parents to create two children
def crossover(p1, p2, r_cross):
	# children are copies of parents by default
	c1, c2 = p1.copy(), p2.copy()
	# check for recombination
	if rand() < r_cross:
		# select crossover point that is not on the end of the string
		pt = randint(1, len(p1)-2)
		# perform crossover
		c1 = p1[:pt] + p2[pt:]
		c2 = p2[:pt] + p1[pt:]
	return [c1, c2]

# mutation operator
def mutation(bitstring, r_mut):
	for i in range(len(bitstring)):
		# check for a mutation
		if rand() < r_mut:
			# flip the bit
			bitstring[i] = 1 - bitstring[i]

# genetic algorithm
def genetic_algorithm(objective, n_bits, n_iter, n_pop, r_cross, r_mut):
	# initial population of random bitstring
	pop = [randint(0, 2, n_bits).tolist() for _ in range(n_pop)]
	# keep track of best solution
	best, best_eval = 0, objective(pop[0])
	# enumerate generations
	for gen in range(n_iter):
		# evaluate all candidates in the population
		scores = [objective(c) for c in pop]
		# check for new best solution
		for i in range(n_pop):
			if scores[i] < best_eval:
				best, best_eval = pop[i], scores[i]
				print(">%d, new best f(%s) = %.3f" % (gen,  pop[i], scores[i]))
		# select parents
		selected = [selection(pop, scores) for _ in range(n_pop)]
		# create the next generation
		children = list()
		for i in range(0, n_pop, 2):
			# get selected parents in pairs
			p1, p2 = selected[i], selected[i+1]
			# crossover and mutation
			for c in crossover(p1, p2, r_cross):
				# mutation
				mutation(c, r_mut)
				# store for next generation
				children.append(c)
		# replace population
		pop = children
	return [best, best_eval]

# define the total iterations
n_iter = 100
# bits
n_bits = 108
# define the population size
n_pop = 100
# crossover rate
r_cross = 0.9
# mutation rate
r_mut = 1.0 / float(n_bits)
# perform the genetic algorithm search
#best, score = genetic_algorithm(onemax, n_bits, n_iter, n_pop, r_cross, r_mut)
#print('Done!')
#print('f(%s) = %f' % (best, score))


