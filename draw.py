from dataloader import DataLoader
from utils import selectData
import numpy as np
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os


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


def draw(pic):
	pass


# python ./draw.py -d sa -p result/my/cao/sa/probmap.pkl -s result/img/cao/sa/ -f png
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-d", "--data", default="sa")
	parser.add_argument("-p", "--path",default="result/my/cao/sa/probmap.pkl")
	parser.add_argument("-s", "--save", default="result/img/cao/sa/")
	parser.add_argument("-f", "--format", default="png")
	args = parser.parse_args()
	probMapPath = args.save + "probmap" + "."+args.format
	predMapPath = args.save + "predmap" + "."+args.format
	dpi = 300
	dic = {"pc": 3, "pu": 2, "sl": 4, "sa": 5}
	if not os.path.exists(args.save):
		os.makedirs(args.save)
	for i in (args.data,):
		pmap = load(args.path)
		print(np.shape(pmap))
		map = np.argmax(pmap, axis=1)
		currentPx = 0
		pathName, matName = selectData(dic[i])
		dl = DataLoader(pathName, matName, 7, 0.1, 1)
		index = dl.loadAllLabeledData()[-1]
		grdt = dl.label
		gt = np.zeros((dl.height, dl.width))
		showTicks=True
		for i in range(len(map)):
			idx = index[i]
			h = idx // dl.width
			w = idx % dl.width
			gt[h][w] = map[i] + 1
		plt.figure(figsize=(dl.width * 50 / dpi, dl.height * 50 / dpi))
		plt.imshow(convertToColor(gt))
		# plt.savefig(probMapPath, format=args.format, dpi=dpi)
		plt.show()
		pm = np.zeros((dl.height, dl.width))
		for i in range(len(map)):
			idx = index[i]
			h = idx // dl.width
			w = idx % dl.width
			# print(h,w)
			pm[h][w] = pmap[i][grdt[h][w] - 1]

		plt.figure(figsize=(dl.width * 50 / dpi, dl.height * 50 / dpi))
		plt.imshow(pm)
		plt.colorbar()
		# plt.savefig(predMapPath, format=args.format, dpi=dpi)
		plt.show()
