from dataloader import DataLoader
from utils import selectData
import numpy as np
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import scipy.io as scio


def load(path):
	f = open(path, "rb")
	return pkl.load(f)


def convertToColor(map):
	palette = {0: (0, 0, 0)}
	for k, color in enumerate(sns.color_palette("hls", len(np.unique(map)))):
		palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype='uint8'))
	map = np.array(map, dtype=int)
	unique = np.unique(map)
	lut = np.zeros((int)(np.max(unique) + 1), dtype=np.int)
	for it, i in enumerate(unique):
		lut[i] = it
	map = lut[map]
	a = np.zeros(shape=(np.shape(map)[0], np.shape(map)[1], 3), dtype=np.uint8)
	for i in range(np.shape(map)[0]):
		for j in range(np.shape(map)[1]):
			a[i][j] = palette[map[i][j]]
	return a


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-d", "--data", default="sa")
	parser.add_argument("-p", "--path", default="result/nn/hamida/sa/")
	parser.add_argument("-s", "--save", default="result/img/hamida/sa/")
	parser.add_argument("-f", "--format", default="png")
	args = parser.parse_args()
	probMapPath = args.save + "probmap" + "." + args.format
	predMapPath = args.save + "predmap" + "." + args.format
	dic = {"pc": 3, "pu": 2, "sl": 4, "sa": 5}
	pathName, matName = selectData(dic[args.data])
	gt = scio.loadmat(pathName[1])[matName[1]]

	f = open(args.path + "predmap.pkl", "rb")
	predmap = pkl.load(f)
	print(np.unique(predmap))
	f = open(args.path + "probmap.pkl", "rb")
	probmap = pkl.load(f)

	dpi = 100
	pd=np.zeros(np.shape(gt))
	for i in range(np.shape(gt)[0]):
		for j in range(np.shape(gt)[1]):
			if gt[i][j]!=0:
				pd[i][j]=predmap[i][j]
	plt.figure()
	plt.imshow(convertToColor(pd))
	plt.show()
