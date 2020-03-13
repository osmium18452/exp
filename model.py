import tensorflow as tf
import capslayer as cl
import numpy as np
from tensorflow.contrib import slim


def DCCapsNet(patch, spectrum, k, output, firstDimension=6,secondDimension=8):
	pt = tf.layers.conv2d(
		patch,
		filters=50,
		kernel_size=3,
		strides=1,
		padding="same",
		activation=tf.nn.relu
	)
	pt = tf.nn.dropout(pt, k)

	pt, ptAct = cl.layers.primaryCaps(
		pt,
		filters=64,
		kernel_size=3,
		strides=1,
		out_caps_dims=[firstDimension, 1],
		method="logistic",
		name="pt"
	)

	ptNumInput = np.prod(cl.shape(pt)[1:4])
	pt = tf.reshape(pt, shape=[-1, ptNumInput, firstDimension, 1])
	ptAct = tf.reshape(ptAct, shape=[-1, ptNumInput])

	sp = tf.layers.conv1d(
		spectrum,
		filters=30,
		kernel_size=32,
		strides=8,
		padding="valid",
		activation=tf.nn.relu
	)
	sp = tf.nn.dropout(sp, k)

	sp = tf.reshape(sp, [-1, sp.shape[1], 1, sp.shape[2]])

	sp, spAct = cl.layers.primaryCaps(
		sp,
		filters=64,
		kernel_size=(3, 1),
		strides=1,
		out_caps_dims=[firstDimension, 1],
		method="logistic",
		name="sp"
	)

	spNumInput = np.prod(cl.shape(sp)[1:4])
	sp = tf.reshape(sp, shape=[-1, spNumInput, firstDimension, 1])
	spAct = tf.reshape(spAct, shape=[-1, spNumInput])

	net = tf.concat([pt, sp], 1)
	act = tf.concat([ptAct, spAct], 1)

	net, act = cl.layers.dense(
		net, act,
		num_outputs=output,
		out_caps_dims=[secondDimension, 1],
		routing_method="DynamicRouting"
	)
	return act


def CapsNet(net, output):
	conv1 = tf.layers.conv2d(
		net,
		filters=64,
		kernel_size=3,
		strides=1,
		padding="same",
		activation=tf.nn.relu,
		name="convLayer"
	)

	convCaps, activation = cl.layers.primaryCaps(
		conv1,
		filters=64,
		kernel_size=3,
		strides=1,
		out_caps_dims=[8, 1],
		method="logistic"
	)

	n_input = np.prod(cl.shape(convCaps)[1:4])
	convCaps = tf.reshape(convCaps, shape=[-1, n_input, 8, 1])
	activation = tf.reshape(activation, shape=[-1, n_input])

	rt_poses, rt_probs = cl.layers.dense(
		convCaps,
		activation,
		num_outputs=output,
		out_caps_dims=[16, 1],
		routing_method="DynamicRouting"
	)
	return rt_probs

def conv_net(net,num_classes):
	with slim.arg_scope([slim.conv2d, slim.fully_connected],
	                    activation_fn=tf.nn.relu):
		net = slim.conv2d(net, 300, 3, padding='VALID',
		                  weights_initializer=tf.contrib.layers.xavier_initializer())
		net = slim.max_pool2d(net,2,padding='SAME')
		net = slim.conv2d(net, 200, 3, padding='VALID',
		                  weights_initializer=tf.contrib.layers.xavier_initializer())
		net = slim.max_pool2d(net,2,padding='SAME')
		net = slim.flatten(net)
		net = slim.fully_connected(net,200)
		net = slim.fully_connected(net,100)
		logits = slim.fully_connected(net, num_classes, activation_fn=None)
	return logits
