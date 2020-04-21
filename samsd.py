#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 18:31:42 2019

@author: siat-x
"""

import os
import re
import math
import numpy as np
import tensorflow as tf
from datetime import datetime
import cv2
from glob import glob
from ops import *
from utils import *
from tensorflow.data.experimental import prefetch_to_device, shuffle_and_repeat, map_and_batch


class SAMSD:

    def __init__(self, sess, args):
        self.model_name = 'SAMSD'
        self.sess = sess
        self.gan_type = args.gan_type
        self.epoch = args.epoch
        self.decay_epoch = args.decay_epoch
        self.save_freq = args.save_freq
        self.iteration = args.iteration
        self.init_lr = args.lr
        self.beta1 = 0.5
        
        """ Dir """
        self.checkpoint_dir = args.checkpoint_dir
        self.sample_dir = args.sample_dir
        self.result_dir = args.result_dir
        self.summaries_dir = args.summaries_dir

        """ Weight """
        self.rec_weight = args.rec_weight
        self.cls_weight = args.cls_weight
        self.ld = args.ld
        
        """ Channels """
        self.batch_size = args.batch_size
        self.resn = args.n_res
        self.disn = args.n_dis
        self.img_size = args.img_size
        self.img_ch = args.img_ch
        self.img_shape = (args.img_size, args.img_size, args.img_ch)
        self.ch = args.ch
            
        """ Dataset """
        self.label_size = args.label_size
        self.vector_size = args.label_size - 3
        self.augment_flag = args.augment_flag
        self.train_dataset_type = args.dataset_type
        self.train_dataset_img_type = args.dataset_img_type
        self.train_dataset_path = args.dataset_path
           
        """ Test """
        self.test_path = args.test_path
        self.CelebaA_attrs = args.CelebaA_attrs
        
##################################################################################
# Generator
##################################################################################
        
    def generator(self, x_init, label, reuse=False, train=True):
        skip = []
        channel = self.ch
        norm='instance'
        with tf.variable_scope('generator', reuse=reuse):

            x = conv(x_init, channel, kernel=7, stride=1, pad=3, activation_fn='relu', 
                     is_training=train, norm_fn=norm, use_bias=False, scope='conv_b')
            
            """ Weight sharing down sampling """
            for i in range(0, 2) :
                skip.append(x)
                channel = channel*2
                x = conv(x, channel, kernel=4, stride=2, pad=1, activation_fn='relu', 
                         is_training=train, norm_fn=norm, use_bias=False, scope='conv_'+str(i))
                
            """ Down sampling for attention mask """
            y = tf.identity(x)
            for i in range(0, 3) :
                skip.append(y)
                y = conv(y, channel, kernel=4, stride=2, pad=1, activation_fn='relu', 
                         is_training=train, norm_fn=norm, use_bias=False, scope='conv_y'+str(i))
                
            """ Up sampling for attention mask """
            for i in range(0, 5) :
                channel = skip[-(i+1)].shape[-1]
                y = up_sample(y, scale_factor=2)
                y = conv(y, channel, kernel=5, stride=1, pad=2, activation_fn='relu', 
                         is_training=train, norm_fn='layer', use_bias=False, scope='deconv_y'+str(i))
                y = tf.concat([y, skip[-(i+1)]], axis=-1)
                
            """ Adaptive resblock for color mask """
            channel = 256
            mu, sigma = self.MLP(label, channel, scope='MLP')
            for i in range(0, self.resn) :
                x = adaptive_resblock(x, channel, mu, sigma, use_bias=False, scope='adaptive_resblock'+str(i))
                
            """ Up sampling for color mask """
            for i in range(0, 2) :
                channel = channel/2
                x = up_sample(x, scale_factor=2)
                x = conv(x, channel, kernel=5, stride=1, pad=2, activation_fn='relu', 
                         is_training=train, norm_fn='layer', use_bias=False, scope='deconv_x'+str(i))
                
            """ Color mask """
            x = conv(x, self.img_ch, kernel=7, stride=1, pad=3, activation_fn='tanh', 
                     is_training=train, norm_fn='None', use_bias=False, scope='conv_x')
            
            """ Attention mask group """
            mask_group = conv(y, self.vector_size, kernel=7, stride=1, pad=3, activation_fn='sigmoid', 
                              is_training=train, norm_fn='None', use_bias=False, scope='conv_mask')
            
            return x, mask_group

    def MLP(self, x, channel, scope='MLP'):
        with tf.variable_scope(scope) :
            
            for i in range(0, 4) :
                x = linear(x, x.shape[-1]*2, activation_fn='relu', norm_fn='None',scope='linear_'+str(i))
            x = linear(x, channel*2, activation_fn='None', norm_fn='None',scope='linear_x')
            mu, sigma = tf.split(x, 2, axis=1)
            mu = tf.reshape(mu, shape=[-1, 1, 1, channel])
            sigma = tf.reshape(sigma, shape=[-1, 1, 1, channel])

            return mu, sigma
        
##################################################################################
# Discriminator
##################################################################################

    def discriminator(self, x, reuse=False, train=True, dropout=False):
        norm='None'#'spectral#instance
        with tf.variable_scope("discriminator", reuse=reuse) :
        
            channel = self.ch
            for i in range(0, int(self.disn/2)):
                x = conv(x, channel, kernel=4, stride=2, pad=1, activation_fn='leaky', 
                         is_training=train, norm_fn=norm, scope='conv_' + str(i))
                channel = channel * 2
                
            x = self.attention(x, channel/2, scope='attention', reuse=reuse)
            
            for i in range(int(self.disn/2), self.disn):
                x = conv(x, channel, kernel=4, stride=2, pad=1, activation_fn='leaky', 
                         is_training=train, norm_fn=norm, scope='conv_' + str(i))
                channel = channel * 2

            logit =  conv(x, 1, 3, stride=1, pad=1, activation_fn='None', 
                          is_training=train, norm_fn='None', use_bias=False, scope='D_logit')

            x = conv(x, self.label_size, kernel=int(x.shape[1]), stride=1, pad=0, activation_fn='None', 
                     is_training=train, norm_fn='None', use_bias=False, scope='D_label')
            
            x = tf.reshape(x, shape=[-1, self.label_size])

            return logit, tf.nn.sigmoid(x)
        
##################################################################################
# Model
##################################################################################

    def AWM(self, mask_group, label, scope='AWM'):
        with tf.variable_scope(scope) :
            
            label = tf.abs(tf.reshape(label, shape=[-1, 1, 1, self.label_size]))
            label0, label1, label2 = tf.split(label, [2, 4, 3], axis=-1)
            label1 = tf.clip_by_value(tf.reduce_sum(label1, axis=-1, keepdims=True), 0., 1.)
            label = tf.concat([label0, label1, label2], axis=-1)
            mask_group = mask_group*label
            
            mask_sum = tf.reduce_sum(mask_group, axis=-1, keepdims=True) + 0.01
            mask = tf.reduce_sum(tf.pow(mask_group, 1.6)/mask_sum, axis=-1, keepdims=True)

            return mask

    def attention(self, x, ch, scope='attention', reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            f = conv(x, ch // 8, kernel=1, stride=1, activation_fn='None', norm_fn='None', scope='f_conv') # [bs, h, w, c']
            g = conv(x, ch // 8, kernel=1, stride=1, activation_fn='None', norm_fn='None', scope='g_conv') # [bs, h, w, c']
            h = conv(x, ch, kernel=1, stride=1, activation_fn='None', norm_fn='None', scope='h_conv') # [bs, h, w, c]
    
            # N = h * w
            s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True) # # [bs, N, N]
    
            beta = tf.nn.softmax(s)  # attention map
    
            o = tf.matmul(beta, hw_flatten(h)) # [bs, N, C]
            gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
    
            o = tf.reshape(o, shape=x.shape) # [bs, h, w, C]
            x = gamma * o + x
    
            return x

    def gradient_panalty(self, real, fake, scope="discriminator"):
            
        if self.gan_type == 'dragan' :
            shape = tf.shape(real)
            eps = tf.random_uniform(shape=shape, minval=0., maxval=1.)
            x_mean, x_var = tf.nn.moments(real, axes=[0, 1, 2, 3])
            x_std = tf.sqrt(x_var)  # magnitude of noise decides the size of local region
            noise = 0.5 * x_std * eps  # delta in paper
            alpha = tf.random_uniform(shape=[shape[0], 1, 1, 1], minval=-1., maxval=1.)
            interpolated = tf.clip_by_value(real + alpha * noise, -1., 1.)  # x_hat should be in the space of X
        else :
            shape = tf.shape(real)
            alpha = tf.random_uniform(shape=[shape[0], 1, 1, 1], minval=0., maxval=1.)
            interpolated = alpha*real + (1. - alpha)*fake

        logit, _ = self.discriminator(interpolated, reuse=True)

        GP = 0

        grad = tf.gradients(logit, interpolated)[0] # gradient of D(interpolated)
        grad_norm = tf.norm(tf.layers.flatten(grad), axis=1) # l2 norm

        # WGAN - LP
        if self.gan_type == 'wgan-lp' :
            GP = tf.reduce_mean(tf.square(tf.maximum(0.0, grad_norm - 1.)))

        if self.gan_type == 'wgan-gp' or self.gan_type == 'dragan':
            GP = tf.reduce_mean(tf.square(grad_norm - 1.))

        return GP

    def optimizer_graph_generator(self, gen_loss, dis_loss, learning_rate_g, learning_rate_d, beta1):
        train_vars = tf.trainable_variables()
        gen_vars = [var for var in train_vars if var.name.startswith('generator')]
        dis_vars = [var for var in train_vars if var.name.startswith('discriminator')]
        # optimizer
        gen_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_g,beta1=beta1,beta2=0.999).minimize(gen_loss, var_list=gen_vars)
        dis_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_d,beta1=beta1,beta2=0.999).minimize(dis_loss, var_list=dis_vars)
        return gen_optimizer, dis_optimizer
    
    def build_model(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.lr_g = tf.placeholder(tf.float32, name='lr_g')
        self.lr_d = tf.placeholder(tf.float32, name='lr_d')
        """ Dataset """
        self.Image_Data = ImageData(self.train_dataset_path, img_shape = self.img_shape, augment_flag = self.augment_flag, 
                                    data_type = self.train_dataset_type, img_type = self.train_dataset_img_type, label_size=self.label_size)

        train = tf.data.Dataset.from_tensor_slices((self.Image_Data.train_dataset, self.Image_Data.train_label))

        dataset_num = len(self.Image_Data.train_dataset)
        gpu_device = '/gpu:0'
        train = train.\
            apply(shuffle_and_repeat(dataset_num)).\
            apply(map_and_batch(self.Image_Data.image_processing, self.batch_size, num_parallel_batches=8, drop_remainder=True)).\
            apply(prefetch_to_device(gpu_device, self.batch_size))

        train_iterator = train.make_one_shot_iterator()

        self.real_imgs, self.label_o = train_iterator.get_next()
        
        """ Translation """
        self.label_t = tf.random_shuffle(self.label_o)
        col_mask, mask_group = self.generator(self.real_imgs, self.label_t-self.label_o)
        att_mask = self.AWM(mask_group, self.label_t-self.label_o)
        self.fake_imgs = col_mask*att_mask + self.real_imgs*(1-att_mask)
        
        """ Reconstruction """
        col_mask, mask_group = self.generator(self.fake_imgs, self.label_o-self.label_t, reuse=True)
        att_mask = self.AWM(mask_group, self.label_o-self.label_t)
        self.recon = col_mask*att_mask + self.fake_imgs*(1-att_mask)
        
        """ Discriminator for real """
        real_logits, real_label = self.discriminator(self.real_imgs)
        
        """ Discriminator for fake """
        fake_logits, fake_label = self.discriminator(self.fake_imgs, reuse=True)
        
        """ Define Loss """
        if self.gan_type.__contains__('wgan') or self.gan_type == 'dragan' :
            grad_pen = self.gradient_panalty(self.real_imgs, self.fake_imgs)
        else :
            grad_pen = 0
        
        cyc_loss = L1_loss(self.real_imgs, self.recon)
        
        g_cls_loss = classification_loss2(label_p=fake_label, label_o=self.label_t)
        d_cls_loss = classification_loss2(label_p=real_label, label_o=self.label_o)
        
        dis_loss = discriminator_loss(self.gan_type, real_logits, fake_logits) + self.ld * grad_pen
        gen_loss = generator_loss(self.gan_type, fake_logits)
        
        D_loss = dis_loss + self.cls_weight * d_cls_loss
        G_loss = gen_loss + self.rec_weight * cyc_loss + self.cls_weight * g_cls_loss
        """ Optimizer """
        D_loss += regularization_loss('discriminator')
        G_loss += regularization_loss('generator')
        self.gen_optimizer, self.dis_optimizer = self.optimizer_graph_generator(G_loss, D_loss, self.lr_g, self.lr_d, self.beta1)
        """ Summaries """
        self.g_summary = summary({G_loss:'G_loss',
                                  gen_loss: 'gen_loss',
                                  cyc_loss: 'cyc_loss', 
                                  g_cls_loss:'g_cls_loss'})
        self.d_summary = summary({D_loss:'D_loss',
#                                  grad_pen_t:'grad_pen_t',
                                  dis_loss: 'dis_loss', 
                                  d_cls_loss:'d_cls_loss'})
    
        """ validate """
        test_label = tf.constant(self.Image_Data.CelebaA.CreateLabel(self.CelebaA_attrs))
        test_label = tf.reshape(test_label, shape=[1, self.label_size])
        test_label = tf.tile(test_label, [self.batch_size, 1])
        self.col_mask, self.mask_group = self.generator(self.real_imgs, test_label-self.label_o, reuse=True)
        self.att_mask = self.AWM(self.mask_group, test_label-self.label_o)
        self.fake_imgs_test = self.col_mask*self.att_mask + self.real_imgs*(1-self.att_mask)
        
        
    def train(self):
        
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
    
        # summary writer
        self.writer = tf.summary.FileWriter(os.path.join(self.summaries_dir), self.sess.graph)
        
        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.iteration)
            start_batch_id = checkpoint_counter - start_epoch * self.iteration
            step = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)
            start_epoch = 0
            start_batch_id = 0
            step = 1
            print(" [!] Load failed...")
        self.variables_count()
                
        lr = self.init_lr
        for epoch in range(start_epoch, self.epoch):
            lr = self.init_lr if epoch < self.decay_epoch else self.init_lr * (self.epoch - epoch) / (self.epoch - self.decay_epoch) # linear decay
    
            for idx in range(start_batch_id, self.iteration):
                train_feed_dict = {self.lr_g : lr, self.lr_d : lr}
                    
                _,d_summary_opt = self.sess.run([self.dis_optimizer, self.d_summary], feed_dict = train_feed_dict)
                self.writer.add_summary(d_summary_opt, step)
                if (step-1)%5==0:
                    _,g_summary_opt = self.sess.run([self.gen_optimizer, self.g_summary], feed_dict = train_feed_dict)
                    self.writer.add_summary(g_summary_opt, step)
                step += 1
                    
                if np.mod(idx + 1, self.save_freq) == 0:
                    samples_a, samples_t, samples_m, samples_s = self.sess.run([self.real_imgs,self.fake_imgs_test, self.att_mask, self.mask_group])
                    test_shape = (self.batch_size*self.img_size, self.img_size, self.img_ch)
                    samples_a=np.uint8(127.5*(np.reshape(samples_a,test_shape)+1.0))
                    samples_t=np.uint8(127.5*(np.reshape(samples_t,test_shape)+1.0))
                    samples_m=np.uint8(255*(np.reshape(samples_m,(self.batch_size*self.img_size, self.img_size, 1))))
                    samples_m = cv2.cvtColor(samples_m, cv2.COLOR_GRAY2RGB) 
                    sample = np.concatenate([samples_a,samples_t,samples_m],axis=1)
                    sample = cv2.cvtColor(sample, cv2.COLOR_RGB2BGR) 
                    samples_s=np.uint8(255*(np.reshape(samples_s,(self.batch_size*self.img_size, self.img_size, self.vector_size))))
                    for i in range(self.vector_size):
                        m = samples_s[:,:,i]
                        m = cv2.cvtColor(m, cv2.COLOR_GRAY2RGB) 
                        sample = np.hstack((sample,m))
                    cv2.imwrite(os.path.join(self.sample_dir,str(step)+'.jpg'), sample)
        
                    print(datetime.now().strftime('%c'), ' epoch:', epoch, ' idx:', idx, '/', self.iteration, ' step:', step)
                    self.saver.save(self.sess, os.path.join(self.checkpoint_dir, self.model_name + '.model'), global_step=step)

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0
    
            # save model for final step
            self.saver.save(self.sess, os.path.join(self.checkpoint_dir, self.model_name + '.model'), global_step=step)

    def test(self):
        
        self.label_vector = tf.placeholder(tf.float32, [1, self.label_size], name='label_vector')
        self.test_img = tf.placeholder(tf.float32, (1,)+self.img_shape, name='test_images')
        self.col_mask, self.mask_group = self.generator(self.test_img, self.label_vector, reuse=True, train=False)
        self.att_mask = self.AWM(self.mask_group, self.label_vector)
        self.test_output = self.col_mask*self.att_mask + self.test_img*(1-self.att_mask)
               
        # restore check-point if it exits
        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        
        path = np.sort(glob(os.path.join(os.getcwd(), self.test_path, '*.*')))
        self.vector = np.zeros((1, self.label_size))
        img = np.zeros((1,)+self.img_shape)
        self.I = img
        
        def callback(object):
            for i in range(self.label_size):
                self.vector[0][i] = cv2.getTrackbarPos(self.Image_Data.CelebaA.labels[i], 'vector')
            
            self.v = (self.vector-100.)/100.
            
            img_o, img_a, mask, ms = self.sess.run([self.test_output, self.col_mask, self.att_mask, self.mask_group], feed_dict={self.test_img: img, self.label_vector: self.v})#
            img_i = np.uint8(127.5*(np.reshape(img,(self.img_size,self.img_size,3))+1.0))
            img_a = np.uint8(127.5*(np.reshape(img_a,(self.img_size, self.img_size, 3))+1.0))
            img_o = np.uint8(127.5*(np.reshape(img_o,(self.img_size, self.img_size, 3))+1.0))
            ms=np.uint8(255*(np.reshape(ms,(self.img_size, self.img_size, self.vector_size))))
            I = np.hstack((img_i,img_o))
            I = np.hstack((I,img_a))
            mask = np.uint8(255*np.reshape(mask,(self.img_size, self.img_size, 1)))
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) 
            I = np.hstack((I,mask))
            for i in range(self.vector_size):
                m = ms[:,:,i]
                m = cv2.cvtColor(m, cv2.COLOR_GRAY2RGB) 
                I = np.hstack((I,m))
            self.I = cv2.cvtColor(I, cv2.COLOR_RGB2BGR) 
            cv2.imshow('image', self.I)
        
        cv2.namedWindow('vector')
        for i in range(self.label_size):
            cv2.createTrackbar(self.Image_Data.CelebaA.labels[i], 'vector', 0, 200, callback)
            cv2.setTrackbarPos(self.Image_Data.CelebaA.labels[i], 'vector', 100)
        
        
        for file in path:
            img = load_test_data(file, size_h=self.img_size, size_w=self.img_size)
            while(1):
                for i in range(self.label_size):
                    cv2.setTrackbarPos(self.Image_Data.CelebaA.labels[i], 'vector', 100)
                self.v = (self.vector-100.)/100.
                
                img_o, img_a, mask, ms = self.sess.run([self.test_output, self.col_mask, self.att_mask, self.mask_group], feed_dict={self.test_img: img, self.label_vector: self.v})#
                img_i = np.uint8(127.5*(np.reshape(img,(self.img_size,self.img_size,3))+1.0))
                img_a = np.uint8(127.5*(np.reshape(img_a,(self.img_size, self.img_size, 3))+1.0))
                img_o = np.uint8(127.5*(np.reshape(img_o,(self.img_size, self.img_size, 3))+1.0))
                ms=np.uint8(255*(np.reshape(ms,(self.img_size, self.img_size, self.vector_size))))
                I = np.hstack((img_i,img_o))
                I = np.hstack((I,img_a))
                mask = np.uint8(255*np.reshape(mask,(self.img_size, self.img_size, 1)))
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) 
                I = np.hstack((I,mask))
                for i in range(self.vector_size):
                    m = ms[:,:,i]
                    m = cv2.cvtColor(m, cv2.COLOR_GRAY2RGB) 
                    I = np.hstack((I,m))
                self.I = cv2.cvtColor(I, cv2.COLOR_RGB2BGR) 
            
                cv2.imshow('image', self.I)
                k = cv2.waitKey(0) & 0xFF
                if k == 27:
                    break

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
#        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
        
    def set_value(self, matrix, x, y, val):
        w = int(matrix.get_shape()[0])
        h = int(matrix.get_shape()[1])
        mult_matrix = tf.sparse_tensor_to_dense(tf.SparseTensor(indices=[[x, y]], values=[-1.0], dense_shape=[w, h])) + 1.0
        diff_matrix = tf.sparse_tensor_to_dense(tf.SparseTensor(indices=[[x, y]], values=[val], dense_shape=[w, h]))
        matrix = tf.multiply(matrix, mult_matrix) 
        matrix = matrix + diff_matrix
        return matrix
    
    def variables_count(self):
        train_vars = tf.trainable_variables()
        gen_vars = [var for var in train_vars if var.name.startswith('generator')]
        dis_vars = [var for var in train_vars if var.name.startswith('discriminator')]
        print("Generator variables:", np.sum([np.prod(v.get_shape().as_list()) for v in gen_vars]))
        print("Discriminator variables:", np.sum([np.prod(v.get_shape().as_list()) for v in dis_vars]))
        print("Total variables:", np.sum([np.prod(v.get_shape().as_list()) for v in train_vars]))