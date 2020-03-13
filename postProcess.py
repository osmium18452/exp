import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from dataloader import DataLoader
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from utils import TQDM_ASCII
import seaborn as sns


class TrainProcess:
	def __init__(self, path):
		self.trainLoss = np.array([])
		self.trainAcc = np.array([])
		self.testLoss = np.array([])
		self.testAcc = np.array([])
		self.dataDir = os.path.join(path, "data")
		self.imgDir = os.path.join(path, "img")
		if not os.path.exists(self.imgDir):
			os.makedirs(self.imgDir)

	def addData(self, trainLoss, trainAcc, testLoss, testAcc):
		self.trainLoss = np.append(self.trainLoss, trainLoss)
		self.trainAcc = np.append(self.trainAcc, trainAcc)
		self.testLoss = np.append(self.testLoss, testLoss)
		self.testAcc = np.append(self.testAcc, testAcc)

	def save(self):
		f = open(os.path.join(self.dataDir, "trainLoss.pkl"), "wb")
		pickle.dump(self.trainLoss, f)
		f = open(os.path.join(self.dataDir, "trainAcc.pkl"), "wb")
		pickle.dump(self.trainAcc, f)
		f = open(os.path.join(self.dataDir, "testLoss.pkl"), "wb")
		pickle.dump(self.testLoss, f)
		f = open(os.path.join(self.dataDir, "testAcc.pkl"), "wb")
		pickle.dump(self.testAcc, f)

	def restore(self):
		f = open(os.path.join(self.dataDir, "trainLoss.pkl"), "rb")
		self.trainLoss = pickle.load(f)
		f = open(os.path.join(self.dataDir, "trainAcc.pkl"), "rb")
		self.trainAcc = pickle.load(f)
		f = open(os.path.join(self.dataDir, "testLoss.pkl"), "rb")
		self.testLoss = pickle.load(f)
		f = open(os.path.join(self.dataDir, "testAcc.pkl"), "rb")
		self.testAcc = pickle.load(f)

	def draw(self):
		plt.figure(figsize=(8, 4.5))
		ax1 = plt.subplot()
		plt.title("loss and accuracy of training and testing")
		ax1.set_xlabel("epochs")
		x = range(len(self.trainLoss))
		ax1.set_ylabel("loss")
		ax2 = ax1.twinx()
		ax2.set_ylabel("accuracy")

		kwargs = {
			"marker": None,
			"lw": 2,
		}
		l1, = ax1.plot(x, self.trainLoss, color="tab:blue", label="train loss", **kwargs)
		l2, = ax2.plot(x, self.trainAcc, color="tab:orange", label="train accuracy", **kwargs)
		l3, = ax1.plot(x, self.testLoss, color="tab:green", label="test loss", **kwargs)
		l4, = ax2.plot(x, self.testAcc, color="tab:red", label="test accuracy", **kwargs)

		plt.legend(handles=[l1, l2, l3, l4], loc="center right")
		sv = plt.gcf()
		sv.savefig(os.path.join(self.imgDir, "lossAndAcc.eps"), format="eps", dpi=300)

	def drawLoss(self):
		plt.figure(figsize=(8, 4.5))
		x = range(len(self.trainLoss))
		plt.title("Loss of training and testing")
		plt.plot(x, self.trainLoss, label="train loss")
		plt.plot(x, self.testLoss, label="test loss")
		plt.legend()
		sv = plt.gcf()
		sv.savefig(os.path.join(self.imgDir, "loss.eps"), format="eps", dpi=300)

	def drawAcc(self):
		plt.figure(figsize=(8, 4.5))
		x = range(len(self.trainAcc))
		plt.title("accuracy of training and testing")
		plt.plot(x, self.trainAcc, label="train loss")
		plt.plot(x, self.testAcc, label="test loss")
		plt.legend()
		sv = plt.gcf()
		sv.savefig(os.path.join(self.imgDir, "acc.eps"), format="eps", dpi=300)


class ProbMap:
	def __init__(self, numClasses, path, groundTruth, index, height, width, trainIndex):
		self.map = np.zeros((1, numClasses))
		self.groundTruth = groundTruth
		self.index = index
		self.dataDir = os.path.join(path, "data")
		self.height = height
		self.width = width
		self.groundTruth = np.argmax(self.groundTruth, axis=1)
		self.trainIndex = trainIndex
		self.imgDir = os.path.join(path, "img")
		self.dic = ["black", "snow", "red", "tomato", "chocolate", "orange", "wheat", "gold", "yellow", "chartreuse",
		            "limegreen", "aquamarine", "cyan", "dodgerblue", "slateblue", "violet", "pink"]
		self.cmap = mpl.colors.ListedColormap(self.dic[0:len(np.unique(self.groundTruth))])
		self.palette = {0: (0, 0, 0)}
		for k, color in enumerate(sns.color_palette("hls", len(np.unique(self.groundTruth)))):
			self.palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype='uint8'))
		# print(k,color)

	def convertToColor(self, map):
		map = np.array(map, dtype=int)
		unique = np.unique(map)
		lut = np.zeros((int)(np.max(unique) + 1), dtype=np.int)
		for it, i in enumerate(unique):
			lut[i] = it
		map = lut[map]
		a = np.zeros(shape=(np.shape(map)[0], np.shape(map)[1], 3), dtype=np.uint8)
		for i in range(np.shape(map)[0]):
			for j in range(np.shape(map)[1]):
				a[i][j] = self.palette[map[i][j]]
		return a

	def addData(self, data):
		self.map = np.concatenate((self.map, data), axis=0)

	def finish(self):
		self.map = np.delete(self.map, (0), axis=0)

	def save(self):
		f = open(os.path.join(self.dataDir, "probmap.pkl"), "wb")
		pickle.dump(self.map, file=f)
		print("probmap saved!")

	def restore(self):
		f = open(os.path.join(self.dataDir, "probmap.pkl"), "rb")
		self.map = pickle.load(f)

	def drawGt(self):
		# f=open("txt/wtf.txt", "w+")
		plt.figure()
		groundTruth = np.zeros(shape=(self.height, self.width), dtype=int)
		# print(self.height,self.width,"8888888888")
		with tqdm(total=np.shape(self.groundTruth)[0], desc="processing gt", ascii=TQDM_ASCII) as pbar:
			for i in range(np.shape(self.groundTruth)[0]):
				index = self.index[i]
				h = index // self.width
				w = index % self.width
				# print("%5d%5d%5d%5d"%(h,w,index,i),file=f)
				# if groundTruth[h][w]!=0:
				# 	print("*****",groundTruth[h][w],h,w,index,i,file=f)
				groundTruth[h][w] += (self.groundTruth[i] + 1)
				# print(groundTruth[h][w],file=f)
				pbar.update()
		# # print(np.unique(groundTruth),"*******")
		# f.close()

		groundTruth = self.convertToColor(groundTruth)
		plt.imshow(groundTruth)
		# plt.show()
		sv = plt.gcf()
		sv.savefig(os.path.join(self.imgDir, "gt.eps"), format="eps", dpi=300)

	# print(2)

	def drawPredictedMap(self):
		pred = np.argmax(self.map, axis=1)
		# print(np.shape(self.map),np.shape(pred),"******")
		probMap = np.zeros(shape=(self.height, self.width))
		with tqdm(total=np.shape(self.groundTruth)[0], desc="processing gt", ascii=TQDM_ASCII) as pbar:
			for i in range(np.shape(self.groundTruth)[0]):
				index = self.index[i]
				h = index // self.width
				w = index % self.width
				probMap[h][w] = (pred[i] + 1)
				# print(h,w,i,np.shape(pred),np.shape(self.groundTruth)[0])
				pbar.update()

		probMap = self.convertToColor(probMap)
		plt.figure(figsize=np.shape(probMap))
		plt.imshow(probMap)
		ticks=False
		plt.tick_params(axis='both', which='both', bottom=ticks, top=ticks, labelbottom=ticks, right=ticks, left=ticks,
		                labelleft=ticks)
		sv = plt.gcf()
		sv.savefig(os.path.join(self.imgDir, "predictMap.eps"), format="eps", dpi=300)

	def drawProbMap(self):
		plt.figure()
		probMap = np.zeros(shape=[self.height, self.width])
		for i in range(np.shape(self.groundTruth)[0]):
			index = self.index[i]
			h = index // self.width
			w = index % self.width
			probMap[h][w] += self.map[i][self.groundTruth[i]]

		plt.imshow(probMap, cmap="hot")
		plt.colorbar()
		# plt.show()
		sv = plt.gcf()
		sv.savefig(os.path.join(self.imgDir, "probMap.eps"), format="eps", dpi=300)

	def drawTrainMap(self):
		plt.figure()
		groundTruth = np.zeros(shape=(self.height, self.width))
		with tqdm(total=np.shape(self.groundTruth)[0], desc="processing gt", ascii=TQDM_ASCII) as pbar:
			for i in range(np.shape(self.groundTruth)[0]):
				index = self.index[i]
				h = index // self.width
				w = index % self.width
				groundTruth[h][w] += (self.groundTruth[i] + 1)
				pbar.update()

		trainMap = np.zeros(shape=[self.height, self.width])
		for i in range(np.shape(self.trainIndex)[0]):
			index = self.trainIndex[i]
			h = index // self.width
			w = index % self.width
			trainMap[h][w] = 1

		for i in range(self.height):
			for j in range(self.width):
				trainMap[i][j] *= groundTruth[i][j]

		trainMap = self.convertToColor(trainMap)
		plt.imshow(trainMap, cmap=self.cmap)
		# plt.show()
		sv = plt.gcf()
		sv.savefig(os.path.join(self.imgDir, "trainMap.eps"), format="eps", dpi=300)

	def drawTestMap(self):
		plt.figure()
		groundTruth = np.zeros(shape=(self.height, self.width))
		with tqdm(total=np.shape(self.groundTruth)[0], desc="processing gt", ascii=TQDM_ASCII) as pbar:
			for i in range(np.shape(self.groundTruth)[0]):
				index = self.index[i]
				h = index // self.width
				w = index % self.width
				groundTruth[h][w] += (self.groundTruth[i] + 1)
				pbar.update()

		testMap = np.ones(shape=[self.height, self.width])
		# print(np.shape(self.groundTruth)[0])
		for i in range(np.shape(self.trainIndex)[0]):
			index = self.trainIndex[i]
			h = index // self.width
			w = index % self.width
			testMap[h][w] = 0

		for i in range(self.height):
			for j in range(self.width):
				testMap[i][j] *= groundTruth[i][j]

		testMap = self.convertToColor(testMap)
		plt.imshow(testMap)
		# plt.show()
		sv = plt.gcf()
		sv.savefig(os.path.join(self.imgDir, "testMap.eps"), format="eps", dpi=300)


if __name__ == "__main__":
	# trainProcess = TrainProcess(os.path.join(".", "save", "feb28"))
	# trainProcess.restore()
	# trainProcess.draw()
	# exit(0)

	pathName = []
	pathName.append("./data/SalinasA_corrected.mat")
	pathName.append("./data/SalinasA_gt.mat")
	matName = []
	matName.append("salinasA_corrected")
	matName.append("salinasA_gt")
	dataloader = DataLoader(pathName, matName, 9, 0.05, 1)
	print(dataloader.numClasses)
	allLabeledPatch, allLabeledSpectrum, allLabeledLabel, allLabeledIndex = dataloader.loadAllLabeledData()
	probMap = ProbMap(dataloader.numClasses, ".\\save\\formal0311\\data\\DCCN\\p5\\sa\\1",
	                  allLabeledLabel, allLabeledIndex, dataloader.height, dataloader.width, dataloader.trainIndex)
	probMap.restore()
	with open("txt/seeData.txt", "w+") as f:
		for i in dataloader.label:
			for j in i:
				print("%3d" % j, end="", file=f)
			print(file=f)
	print(dataloader.numClasses)
	probMap.drawGt()
	print(1)
	probMap.drawPredictedMap()
	print(1)
	probMap.drawProbMap()
	print(1)
	# probMap.drawGt()
	probMap.drawTestMap()
	print(1)
	probMap.drawTrainMap()
	print(1)
