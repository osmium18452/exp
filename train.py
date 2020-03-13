import tensorflow as tf
import numpy as np
from tqdm import tqdm
import capslayer as cl
import os
import argparse
from dataloader import DataLoader
from model import DCCapsNet, CapsNet, conv_net
from utils import LENGTH, calOA, selectData, calMixMatrix, calAA, calKappa, TQDM_ASCII
from postProcess import TrainProcess, ProbMap

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", default=50, type=int)
parser.add_argument("-b", "--batch_size", default=50, type=int)
parser.add_argument("-l", "--lr", default=0.001, type=float)
parser.add_argument("-g", "--gpu", default="0")
parser.add_argument("-r", "--ratio", default=0.1, type=float)
parser.add_argument("-a", "--aug", default=1, type=float)
parser.add_argument("-p", "--patch_size", default=9, type=int)
parser.add_argument("-m", "--model", default=1, type=int)
parser.add_argument("-d", "--directory", default="./save/default")
parser.add_argument("--model_path", default="./save/default/model")
parser.add_argument("--sum_path",default="./save/default/sum")
parser.add_argument("--drop", default=1, type=float)
parser.add_argument("--data", default=0, type=int)
parser.add_argument("--first_layer", default=6, type=int)
parser.add_argument("--second_layer", default=8, type=int)
parser.add_argument("--predict_only", action="store_true")
parser.add_argument("--restore", action="store_true")
parser.add_argument("--use_best_model", action="store_true")
parser.add_argument("--dont_save_data", action="store_true")
parser.add_argument("--no_detailed_summary", action="store_true")
parser.add_argument("--draw", action="store_true")
args = parser.parse_args()
print(args)

EPOCHS = args.epochs
LEARNING_RATE = args.lr
BATCH_SIZE = args.batch_size
RATIO = args.ratio
AUGMENT_RATIO = args.aug
PATCH_SIZE = args.patch_size
DROP_OUT = args.drop
DATA = args.data
DIRECTORY = args.directory
RESTORE = args.restore
PREDICT_ONLY = args.predict_only
MODEL_DIRECTORY = args.model_path
USE_BEST_MODEL = args.use_best_model
DONT_SAVE_DATA = args.dont_save_data
NO_DETAILED_SUMMARY=args.no_detailed_summary
DRAW = args.draw
FIRST_LAYER, SECOND_LAYER = args.first_layer, args.second_layer
SUM_PATH=args.sum_path

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
if not os.path.exists(DIRECTORY):
	os.makedirs(os.path.join(DIRECTORY, "img"))
	os.makedirs(os.path.join(DIRECTORY, "data"))
if not os.path.exists(MODEL_DIRECTORY):
	os.makedirs(MODEL_DIRECTORY)
if not os.path.exists(SUM_PATH):
	os.makedirs(SUM_PATH)

modelSavePath = os.path.join(MODEL_DIRECTORY, "model.ckpt")
imgSavePath = os.path.join(DIRECTORY, "img")
dataSavePath = os.path.join(DIRECTORY, "data")
sumSavePath=SUM_PATH

pathName, matName = selectData(DATA)
dataloader = DataLoader(pathName, matName, PATCH_SIZE, RATIO, AUGMENT_RATIO)

trainPatch, trainSpectrum, trainLabel = dataloader.loadTrainData()
testPatch, testSpectrum, testLabel = dataloader.loadTestData()
allLabeledPatch, allLabeledSpectrum, allLabeledLabel, allLabeledIndex = dataloader.loadAllLabeledData()

w = tf.placeholder(shape=[None, dataloader.bands, 1], dtype=tf.float32)
x = tf.placeholder(shape=[None, dataloader.patchSize, dataloader.patchSize, dataloader.bands], dtype=tf.float32)
y = tf.placeholder(shape=[None, dataloader.numClasses], dtype=tf.float32)
k = tf.placeholder(dtype=tf.float32)

if args.model == 1:
	pred = DCCapsNet(x, w, k, dataloader.numClasses, FIRST_LAYER, SECOND_LAYER)
	print("USING DCCAPS***************************************")
elif args.model==2:
	pred = CapsNet(x, dataloader.numClasses)
	print("USING CAPS*****************************************")
else:
	pred=conv_net(x,dataloader.numClasses)
	print("USING CONV*****************************************")
pred = tf.divide(pred, tf.reduce_sum(pred, 1, keep_dims=True))

loss = tf.reduce_mean(cl.losses.margin_loss(y, pred))
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
correctPredictions = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correctPredictions, "float"))
init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
	leastLoss = 100.0
	if RESTORE or PREDICT_ONLY:
		saver.restore(sess, modelSavePath)
	else:
		sess.run(init)

	if not PREDICT_ONLY:
		trainProcess = TrainProcess(DIRECTORY)
		for epoch in range(EPOCHS):
			if epoch % 5 == 0:
				permutation = np.random.permutation(trainPatch.shape[0])
				trainPatch = trainPatch[permutation, :, :, :]
				trainSpectrum = trainSpectrum[permutation, :]
				trainLabel = trainLabel[permutation, :]

			iter = dataloader.trainNum // BATCH_SIZE
			with tqdm(total=iter, desc="epoch %3d/%3d" % (epoch + 1, EPOCHS), ncols=LENGTH, ascii=TQDM_ASCII) as pbar:
				for i in range(iter):
					batch_w = trainSpectrum[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, :, :]
					batch_x = trainPatch[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, :, :, :]
					batch_y = trainLabel[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, :]
					_, batchLoss, trainAcc = sess.run([optimizer, loss, accuracy],
													  feed_dict={w: batch_w, x: batch_x, y: batch_y, k: DROP_OUT})
					pbar.set_postfix_str(
						"loss: %.6f, accuracy:%.2f, testLoss:-.---, testAcc:-.--" % (batchLoss, trainAcc))
					pbar.update(1)

					if i == 0 and epoch == 0:
						idx = np.random.choice(dataloader.testNum, size=BATCH_SIZE, replace=False)
						test_batch_w = testSpectrum[idx, :, :]
						test_batch_x = testPatch[idx, :, :, :]
						test_batch_y = testLabel[idx, :]
						ac, ls = sess.run([accuracy, loss],
										  feed_dict={w: test_batch_w, x: test_batch_x, y: test_batch_y, k: 1})
						trainProcess.addData(batchLoss, trainAcc, ls, ac)

				if batchLoss < leastLoss:
					saver.save(sess, save_path=modelSavePath)
					leastLoss = batchLoss

				if iter * BATCH_SIZE < dataloader.trainNum:
					batch_w = trainSpectrum[iter * BATCH_SIZE:, :, :]
					batch_x = trainPatch[iter * BATCH_SIZE:, :, :, :]
					batch_y = trainLabel[iter * BATCH_SIZE:, :]
					_, bl, ta = sess.run([optimizer, loss, accuracy],
										 feed_dict={w: batch_w, x: batch_x, y: batch_y, k: DROP_OUT})

				idx = np.random.choice(dataloader.testNum, size=BATCH_SIZE, replace=False)
				test_batch_w = testSpectrum[idx, :, :]
				test_batch_x = testPatch[idx, :, :, :]
				test_batch_y = testLabel[idx, :]
				ac, ls = sess.run([accuracy, loss], feed_dict={w: test_batch_w, x: test_batch_x, y: test_batch_y, k: 1})
				pbar.set_postfix_str(
					"loss: %.6f, accuracy:%.2f, testLoss:%.3f, testAcc:%.2f" % (batchLoss, trainAcc, ls, ac))
				trainProcess.addData(batchLoss, trainAcc, ls, ac)

	if USE_BEST_MODEL:
		saver.restore(sess, modelSavePath)
	iter = dataloader.allLabeledNum // BATCH_SIZE
	probMap = ProbMap(dataloader.numClasses, DIRECTORY, allLabeledLabel, allLabeledIndex, dataloader.height,
					  dataloader.width, dataloader.trainIndex)
	with tqdm(total=iter, desc="predicting...", ascii=TQDM_ASCII) as pbar:
		for i in range(iter):
			batch_w = allLabeledSpectrum[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, :, :]
			batch_x = allLabeledPatch[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, :, :, :]
			batch_y = allLabeledLabel[i * BATCH_SIZE:(i + 1) * BATCH_SIZE, :]
			tmp = sess.run(pred, feed_dict={w: batch_w, x: batch_x, y: batch_y, k: 1})
			probMap.addData(tmp)
			pbar.update()

		if iter * BATCH_SIZE < dataloader.allLabeledNum:
			batch_w = allLabeledSpectrum[iter * BATCH_SIZE:, :, :]
			batch_x = allLabeledPatch[iter * BATCH_SIZE:, :, :, :]
			batch_y = allLabeledLabel[iter * BATCH_SIZE:, :]
			tmp = sess.run(pred, feed_dict={w: batch_w, x: batch_x, y: batch_y, k: 1})
			probMap.addData(tmp)

	probMap.finish()
	print(np.shape(probMap.map))

	if not DONT_SAVE_DATA:
		if not PREDICT_ONLY:
			trainProcess.save()
		probMap.save()


	OA = calOA(probMap.map, allLabeledLabel)
	AA = calAA(probMap.map, allLabeledLabel)
	kappa = calKappa(probMap.map, allLabeledLabel)
	mixMatrix = calMixMatrix(probMap.map, allLabeledLabel)

	with open(os.path.join(sumSavePath, "sum.txt"), "a+") as f:
		print("data:%d, oa: %4f, aa:%4f, kappa:%4f" % \
		      (DATA, OA, AA, kappa), file=f)

	if NO_DETAILED_SUMMARY:
		with open(os.path.join(dataSavePath, "sum.txt"), "a+") as f:
			print("first layer: %2d, second layer:%2d, epochs:%3d, data:%d, oa: %4f, aa:%4f, kappa:%4f" %\
				  (FIRST_LAYER, SECOND_LAYER, EPOCHS, DATA, OA, AA, kappa),file=f)
	else:
		with open(os.path.join(dataSavePath, "summary.txt"), "w+") as f:
			print(args, file=f)
			print("OA: %4f, AA: %4f, KAPPA: %4f" % (OA, AA, kappa), file=f)
			print("******* MIX MAP *******", file=f)
			print("   |", end="", file=f)
			for i in range(dataloader.numClasses):
				print("%6d" % i, end="", file=f)
			print(file=f)
			for i in range(6 * dataloader.numClasses + 4):
				print("-", end="", file=f)
			print(file=f)
			for i in range(dataloader.numClasses):
				print("%2d" % i, end=" |", file=f)
				for j in range(dataloader.numClasses):
					print("%6d" % mixMatrix[i][j], end="", file=f)
				print(file=f)

	if DRAW:
		trainProcess.draw()
		trainProcess.drawLoss()
		trainProcess.drawAcc()
		probMap.drawProbMap()
		probMap.drawPredictedMap()
		probMap.drawTrainMap()
		probMap.drawTestMap()
		probMap.drawGt()
