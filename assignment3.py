# Name: Yinyin Jiang
# Net ID: yj1438

import numpy as np

class KNN:
	def __init__(self, k):
		#KNN state here
		#Feel free to add methods
		self.k = k
		self.trainset = []

	def distance(self, featureA, featureB):
		diffs = (featureA - featureB)**2
		return np.sqrt(diffs.sum())

	def train(self, X, y):
		#training logic here
		#input is an array of features and labels
		self.trainset = list(zip(X, y))

	def predict(self, X):
		#Run model here
		#Return array of predictions where there is one prediction for each set of features
		res = []
		for i in range(len(X)):
			difference = []
			for j in range(len(self.trainset)):
				difference.append(self.distance(X[i], self.trainset[j][0]))
			idx = np.argpartition(np.array(difference), self.k)
			tmp = idx[:self.k]

			count = 0
			for m in range(len(tmp)):
				if self.trainset[tmp[m]][1] == 0:
					count += 1
			if count > self.k / 2:
				res.append(0)
			else:
				res.append(1)
		return np.array(res)

class ID3:
	def __init__(self, nbins, data_range):
		#Decision tree state here
		#Feel free to add methods
		self.bin_size = nbins
		self.range = data_range
		self.root = None

	def preprocess(self, data):
		#Our dataset only has continuous data
		norm_data = np.clip((data - self.range[0]) / (self.range[1] - self.range[0]), 0, 1)
		categorical_data = np.floor(self.bin_size*norm_data).astype(int)
		return categorical_data

	def train(self, X, y):
		#training logic here
		#input is array of features and labels
		categorical_data = self.preprocess(X)
		examples = list(zip(categorical_data, y))
		attributes = range(0, len(categorical_data[0]))
		default = None
		self.root = self.DTL(examples, attributes, default)

	def predict(self, X):
		#Run model here
		#Return array of predictions where there is one prediction for each set of features
		categorical_data = self.preprocess(X)
		res = []
		for i in range(len(categorical_data)):
			cur = self.root
			while cur.label is None:
				cur = cur.children[categorical_data[i][cur.attribute]]
			res.append(cur.label)
		return np.array(res)

	def DTL(self, examples, attributes, default):
		if examples is None or len(examples) == 0:
			return default
		elif sum([example[1] for example in examples]) == 0:
			return TreeNode(None, 0)
		elif sum([example[1] for example in examples]) == len(examples):
			return TreeNode(None, 1)
		elif attributes is None or len(attributes) == 0:
			return self.mode(examples)
		else:
			best = self.choose_attribute(attributes, examples)
			tree = TreeNode(best, None)
			for i in range(self.bin_size + 1):
				new_example = []
				for example in examples:
					if example[0][best] == i:
						new_example.append(example)
				subtree = self.DTL(new_example, list(set(attributes) - set(list([best]))), self.mode(examples))
				# different branches are in different indices
				tree.children.append(subtree)
			return tree

	def choose_attribute(self, attributes, examples):
		best_gain = -100000
		best_attribute = None
		info_D = self.calculate_entropy(examples)
		for i in range(len(attributes)):
			info_attribute_D = self.calculate_attribute_entropy(examples, attributes[i])
			if info_D - info_attribute_D > best_gain:
				best_gain = info_D - info_attribute_D
				best_attribute = attributes[i]
		return best_attribute

	def calculate_entropy(self, examples):
		Y = 0.
		N = 0.
		total = len(examples)
		info_D = 0.
		for i in range(len(examples)):
			if examples[i][1] == 1:
				Y += 1
			else:
				N += 1
		if total != 0:
			info_D = - Y/total * np.log2(Y/total) - N/total * np.log2(N/total)
		return info_D

	def calculate_attribute_entropy(self, examples, attribute):
		Y = []
		N = []
		total = len(examples)
		for i in range(self.bin_size + 1):
			Yi = 0.
			Ni = 0.
			for j in range(len(examples)):
				if examples[j][0][attribute] == i:
					if examples[j][1] == 1:
						Yi += 1
					else:
						Ni += 1
			Y.append(Yi)
			N.append(Ni)
		info_attribute_D = 0.
		for i in range(len(Y)):
			total_attribute = Y[i] + N[i]
			if total_attribute != 0:
				if Y[i] != 0:
					first = -1. * Y[i] / total_attribute * np.log2(Y[i] / total_attribute)
				else:
					first = 0
				if N[i] != 0:
					second = -1. * N[i] / total_attribute * np.log2(N[i] / total_attribute)
				else:
					second = 0
				info_attribute_D += total_attribute / total * (first + second)
		return info_attribute_D

	def mode(self, examples):
		count = 0
		for i in range(len(examples)):
			if examples[i][1] == 0:
				count += 1
		if count > len(examples) / 2:
			return TreeNode(None, 0)
		else:
			return TreeNode(None, 1)

class TreeNode:
	def __init__(self, attribute, label):
		self.attribute = attribute
		self.label = label
		self.children = []

class Perceptron:
	def __init__(self, w, b, lr):
		#Perceptron state here, input initial weight matrix
		#Feel free to add methods
		self.lr = lr
		self.w = w
		self.b = b

	def train(self, X, y, steps):
		#training logic here
		#input is array of features and labels
		for i in range(round(steps / len(X))):
			for j in range(len(X)):
				tmp = np.dot(X[j], self.w) + self.b
				predict = 1 if tmp > 0 else 0
				self.w = self.w + self.lr * (y[j] - predict) * X[j]
				self.b = self.b + self.lr * (y[j] - predict)

	def predict(self, X):
		#Run model here
		#Return array of predictions where there is one prediction for each set of features
		res = []
		for i in range(len(X)):
			tmp = X[i] * self.w + self.b
			if np.sum(tmp) > 0:
				res.append(1)
			else:
				res.append(0)
		return np.array(res)

class MLP:
	def __init__(self, w1, b1, w2, b2, lr):
		self.l1 = FCLayer(w1, b1, lr)
		self.a1 = Sigmoid()
		self.l2 = FCLayer(w2, b2, lr)
		self.a2 = Sigmoid()

	def MSE(self, prediction, target):
		return np.square(target - prediction).sum()

	def MSEGrad(self, prediction, target):
		return - 2.0 * (target - prediction)

	def shuffle(self, X, y):
		idxs = np.arange(y.size)
		np.random.shuffle(idxs)
		return X[idxs], y[idxs]

	def train(self, X, y, steps):
		for s in range(steps):
			i = s % y.size
			if(i == 0):
				X, y = self.shuffle(X,y)
			xi = np.expand_dims(X[i], axis=0)
			yi = np.expand_dims(y[i], axis=0)

			pred = self.l1.forward(xi)
			pred = self.a1.forward(pred)
			pred = self.l2.forward(pred)
			pred = self.a2.forward(pred)
			loss = self.MSE(pred, yi) 
			#print(loss)

			grad = self.MSEGrad(pred, yi)
			grad = self.a2.backward(grad)
			grad = self.l2.backward(grad)
			grad = self.a1.backward(grad)
			grad = self.l1.backward(grad)

	def predict(self, X):
		pred = self.l1.forward(X)
		pred = self.a1.forward(pred)
		pred = self.l2.forward(pred)
		pred = self.a2.forward(pred)
		pred = np.round(pred)
		return np.ravel(pred)

class FCLayer:

	def __init__(self, w, b, lr):
		self.lr = lr
		self.w = w	#Each column represents all the weights going into an output node
		self.b = b
		self.input = None

	def forward(self, input):
		#Write forward pass here
		self.input = input
		return np.dot(input, self.w) + self.b

	def backward(self, gradients):
		#Write backward pass here
		w1 = np.dot(np.transpose(self.input), gradients)
		x1 = np.dot(gradients, np.transpose(self.w))
		self.w = self.w - self.lr * w1
		self.b = self.b - self.lr * gradients
		return x1

class Sigmoid:

	def __init__(self):
		self.pred = None

	def forward(self, input):
		# Write forward pass here
		self.pred = 1.0 / (1 + np.exp(-input))
		return self.pred

	def backward(self, gradients):
		#Write backward pass
		return gradients * (1 - self.pred) * self.pred