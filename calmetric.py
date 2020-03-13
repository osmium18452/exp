import argparse
import numpy as np
import re


def cal(path):
	with open(path, "r") as f:
		aa = []
		oa = []
		kappa = []
		for line in f.readlines():
			wordlist = re.split(", |:", line.strip())
			# print(wordlist)
			oa.append((float)(wordlist[3]))
			aa.append((float)(wordlist[5]))
			kappa.append((float)(wordlist[7]))
		# print(aa)
		# print(oa)
		# print(kappa)
		# exit()
		oamid = (np.max(oa) + np.min(oa)) / 2
		oadiff = oamid - np.min(oa)
		aamid = (np.max(aa) + np.min(aa)) / 2
		aadiff = aamid - np.min(aa)
		kpmid = (np.max(kappa) + np.min(kappa)) / 2
		kpdiff = kpmid - np.min(kappa)
		oamid *= 100
		oadiff *= 100
		aamid *= 100
		aadiff *= 100
		kpmid *= 100
		kpdiff *= 100
		return oamid, aamid, kpmid


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-d", "--directory")
	args = parser.parse_args()
	oa = []
	aa = []
	kappa = []
	for i in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10):
		path = "./save/03122/data/DCCN/r%02d/pu/sum/sum.txt" % i
		rt = cal(path)
		oa.append(rt[0])
		aa.append(rt[1])
		kappa.append(rt[2])
	print(aa)
	print(oa)
	print(kappa)
