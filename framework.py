from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from config import cfg
from dataprocess import split_data, LoadImg
from lib import tflib as tl
from basic_model.model import img_feature_ext, imghash, txthash
import scipy.io as sio
from utils.calc import calc_map, calc_neighbor, calc_loss

class Model(object):
    def __init__(self, sess, cfg):
        self.dataset = split_data(cfg.data)
        self.sess = sess
        self.loadimg = LoadImg
        self.crop_size = cfg.data.crop_size
        self.batchsize = cfg.train.batch_size
        self.epochs = cfg.train.epochs
        self.num_train = self.dataset.train_x.shape[0]
        self.text_dim = self.dataset.train_y.shape[1]
        self.unupdated_size = self.num_train - self.batchsize
        self.bit = cfg.bit
        self.gamma = cfg.train.gamma
        self.eta = cfg.train.eta
        self.lr = cfg.train.lr
        self.lr_step = cfg.train.lr_step
        self.continue_train = cfg.continue_train
        self.prtrained_model = cfg.network.pretrain
        self.checkpoint = cfg.checkpoint
        self.task = '{}_{}'.format(cfg.data.dataset, str(cfg.bit))
        self.model_name = cfg.model_name
        self.it_cnt, self.update_cnt = tl.counter()
        self.besti2tmap = 0.0
        self.bestt2imap = 0.0
        self._var()
        self._build()


    def _var(self):
        self.var_F = np.random.randn(self.bit, self.num_train)
        self.var_G = np.random.randn(self.bit, self.num_train)
        self.var_B = np.sign(self.var_F + self.var_G)
        self.Sim = calc_neighbor(self.dataset.train_L, self.dataset.train_L)

    def _build(self):
        self.ph_image = tf.placeholder(tf.float32, [None, 224, 224, 3], name='image')
        self.ph_text = tf.placeholder(tf.float32, [None, 1, 1, self.text_dim], name='text')
        self.ph_phase = tf.placeholder(tf.bool, name='is_training')
        self.ph_lr = tf.placeholder('float32', (), name='lr')
        self.ph_S_x = tf.placeholder('float32', [self.batchsize, self.num_train], name='pS_x')
        self.ph_S_y = tf.placeholder('float32', [self.num_train, self.batchsize], name='pS_y')
        self.ph_F = tf.placeholder('float32', [self.bit, self.num_train], name='pF')
        self.ph_G = tf.placeholder('float32', [self.bit, self.num_train], name='pG')
        self.ph_F_ = tf.placeholder('float32', [self.bit, self.unupdated_size], name='unupdated_F')
        self.ph_G_ = tf.placeholder('float32', [self.bit, self.unupdated_size], name='unupdated_G')
        self.ph_b_batch = tf.placeholder('float32', [self.bit, self.batchsize], name='b_batch')
        self.ph_ones_ = tf.constant(np.ones([self.unupdated_size, 1], 'float32'))
        self.ph_ones_batch = tf.constant(np.ones([self.batchsize, 1], 'float32'))
        # model structure

        ## feature = img_feature_ext(img=self.ph_image, is_training=self.ph_phase)
        feature = img_feature_ext(img=self.ph_image, pretrained=self.prtrained_model, name='vggf')

        self.cur_f_batch = imghash(feature, self.bit, name='img_hash')
        self.cur_g_batch = txthash(self.ph_text, self.bit, name='txt_hash')


        theta_x = 1.0 / 2 * tf.matmul(tf.transpose(self.cur_f_batch), self.ph_G)
        theta_y = 1.0 / 2 * tf.matmul(tf.transpose(self.ph_F), self.cur_g_batch)

        logloss_x = -tf.reduce_sum(tf.multiply(self.ph_S_x, theta_x) - tf.log(1.0 + tf.exp(theta_x)))
        quantization_x = tf.reduce_sum(tf.pow((self.ph_b_batch - self.cur_f_batch), 2))
        balance_x = tf.reduce_sum(tf.pow(tf.matmul(self.cur_f_batch, self.ph_ones_batch) + tf.matmul(self.ph_F_, self.ph_ones_), 2))
        loss_x = tf.div(logloss_x + self.gamma * quantization_x + self.eta * balance_x, float(self.num_train * self.batchsize))

        logloss_y = -tf.reduce_sum(tf.multiply(self.ph_S_y, theta_y) - tf.log(1.0 + tf.exp(theta_y)))
        quantization_y = tf.reduce_sum(tf.pow((self.ph_b_batch - self.cur_g_batch), 2))
        balance_y = tf.reduce_sum(tf.pow(tf.matmul(self.cur_g_batch, self.ph_ones_batch) + tf.matmul(self.ph_G_, self.ph_ones_), 2))
        loss_y = tf.div(logloss_y + self.gamma * quantization_y + self.eta * balance_y, float(self.num_train * self.batchsize))

        self.train_step_x = tf.train.AdamOptimizer(self.ph_lr, beta1=0.5).minimize(loss_x)
        self.train_step_y = tf.train.AdamOptimizer(self.ph_lr, beta1=0.5).minimize(loss_y)



    def train_img_net(self, X, L, var_F, var_G, var_B, lr):
        F = var_F
        for iter in tqdm(range(int(self.num_train / self.batchsize))):
            index = np.random.permutation(self.num_train)
            ind = index[0: self.batchsize]
            unupdated_ind = np.setdiff1d(range(self.num_train), ind)
            sample_L = L[ind, :]
            image = self.loadimg(pathList=X[ind]).astype(np.float32)
            S = calc_neighbor(sample_L, L)
            cur_f = self.cur_f_batch.eval(feed_dict={self.ph_image: image, self.ph_phase: False})
            F[:, ind] = cur_f

            self.train_step_x.run(feed_dict={self.ph_S_x: S,
                                             self.ph_G: var_G,
                                             self.ph_b_batch: var_B[:, ind],
                                             self.ph_F_: F[:, unupdated_ind],
                                             self.ph_lr: lr,
                                             self.ph_phase: True,
                                             self.ph_image: image})
        return F


    def train_txt_net(self, Y, L, var_F, var_G, var_B, lr):
        G = var_G
        for iter in tqdm(range(int(self.num_train / self.batchsize))):
            index = np.random.permutation(self.num_train)
            ind = index[0: self.batchsize]
            unupdated_ind = np.setdiff1d(range(self.num_train), ind)
            sample_L = L[ind, :]
            text = Y[ind, :].astype(np.float32)
            text = text.reshape([text.shape[0], 1, 1, text.shape[1]])
            S = calc_neighbor(L, sample_L)
            cur_g = self.cur_g_batch.eval(feed_dict={self.ph_text: text, self.ph_phase: False})
            G[:, ind] = cur_g

            self.train_step_y.run(feed_dict={self.ph_S_y: S,
                                             self.ph_F: var_F,
                                             self.ph_b_batch: var_B[:, ind],
                                             self.ph_G_: G[:, unupdated_ind],
                                             self.ph_lr: lr,
                                             self.ph_phase: True,
                                             self.ph_text: text})
        return G


    def generate_image_code(self, X):
        num_data = X.shape[0]
        index = np.linspace(0, num_data - 1, num_data).astype(int)
        B = np.zeros([num_data, self.bit], dtype=np.float32)
        for iter in tqdm(range(int(num_data / self.batchsize + 1))):
            ind = index[iter * self.batchsize: min((iter + 1) * self.batchsize, num_data)]
            image = self.loadimg(pathList=X[ind]).astype(np.float32)
            cur_f = self.cur_f_batch.eval(feed_dict={self.ph_image: image, self.ph_phase: False})
            B[ind, :] = cur_f.transpose()
        B = np.sign(B)
        return B

    def generate_text_code(self, Y):
        num_data = Y.shape[0]
        index = np.linspace(0, num_data - 1, num_data).astype(int)
        B = np.zeros([num_data, self.bit], dtype=np.float32)
        for iter in tqdm(range(int(num_data / self.batchsize + 1))):
            ind = index[iter * self.batchsize: min((iter + 1) * self.batchsize, num_data)]
            text = Y[ind, :].astype(np.float32)
            text = text.reshape([text.shape[0], 1, 1, text.shape[1]])
            cur_g = self.cur_g_batch.eval(feed_dict={self.ph_text: text, self.ph_phase: False})
            B[ind, :] = cur_g.transpose()
        B = np.sign(B)
        return B





    def train(self):
        exclusions = ['img_hash', 'txt_hash', 'counter']
        variables_to_restore = []

        # for var in tf.trainable_variables():
        #     print(var.op.name)

        for var in tf.trainable_variables():
            excluded = False
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    excluded = True
            if not excluded:
                variables_to_restore.append(var)

        ## saver_restore = tf.train.Saver(var_list=variables_to_restore)
        saver = tf.train.Saver(max_to_keep=1, var_list=tf.global_variables())

        self.sess.run(tf.global_variables_initializer())

        if not self.continue_train:
            ## saver_restore.restore(self.sess, self.prtrained_model)
            print('Weight is loaded!')
        else:
            tl.load(self.sess, saver, self.checkpoint, self.task)

        init_loss = calc_loss(self.var_B, self.var_F, self.var_G, self.Sim, self.gamma, self.eta)
        print('Initial Loss: %3.3f'%(init_loss))
        print('Start training processing...')



        for epoch in range(self.epochs):
            decay = 0.5 ** (sum(epoch >= np.array(self.lr_step)))
            lr = self.lr * decay

            # update F
            self.var_F = self.train_img_net(self.dataset.train_x, self.dataset.train_L, self.var_F, self.var_G, self.var_B, lr)
            # update G
            self.var_G = self.train_txt_net(self.dataset.train_y, self.dataset.train_L, self.var_F, self.var_G, self.var_B, lr)
            # update B
            self.var_B = np.sign(self.var_F + self.var_G)
            # calculate loss
            loss = calc_loss(self.var_B, self.var_F, self.var_G, self.Sim, self.gamma, self.eta)
            print('============Epoch: {}==Task: {}=============='.format(epoch, self.task))
            print('loss: %3.3f, comment: update B'% (loss))
            print('=============================================')

            if (epoch + 1) % 5 == 0:
                self.test()
                if (self.mapi2t + self.mapt2i) >= (self.besti2tmap + self.bestt2imap):
                    # save model
                    tl.save(self.sess, saver, self.checkpoint, self.task,
                            self.model_name, self.sess.run(self.it_cnt))
                    # save code
                    if os.path.exists('hashcodes/{}_{}_{}.mat'.format(self.model_name, cfg.data.dataset, str(self.bit))):
                        os.remove('hashcodes/{}_{}_{}.mat'.format(self.model_name, cfg.data.dataset, str(self.bit)))
                    sio.savemat('hashcodes/{}_{}_{}.mat'.format(self.model_name, cfg.data.dataset, str(self.bit)),
                                {'Qi': self.qBX, 'Qt': self.qBY, 'Di': self.rBX, 'Dt': self.rBY, 'retrieval_L': self.dataset.retrieval_L,
                                 'query_L': self.dataset.query_L})
                    print('Hash code "{}_{}_{}.mat" are saved...'.format(self.model_name, cfg.data.dataset, str(self.bit)))
                    self.besti2tmap = self.mapi2t
                    self.bestt2imap = self.mapt2i

            self.sess.run(self.update_cnt)


    def test(self):
        self.qBX = self.generate_image_code(self.dataset.query_x)
        self.qBY = self.generate_text_code(self.dataset.query_y)
        self.rBX = self.generate_image_code(self.dataset.retrieval_x)
        self.rBY = self.generate_text_code(self.dataset.retrieval_y)
        self.mapi2t = calc_map(self.qBX, self.rBY, self.dataset.query_L, self.dataset.retrieval_L)
        self.mapt2i = calc_map(self.qBY, self.rBX, self.dataset.query_L, self.dataset.retrieval_L)
        print('Test MAP: MAP(i->t): %5.5f, MAP(t->i): %5.5f' % (self.mapi2t, self.mapt2i))


