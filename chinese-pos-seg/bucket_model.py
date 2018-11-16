# -*- coding: utf-8 -*-
from __future__ import division
import tensorflow as tf
from layers import EmbeddingLayer, BiRNN, HiddenLayer, TimeDistributed, Forward
from time import time
import losses
import toolbox
import batch as Batch
import numpy as np
import random
import cPickle as pickle
import shutil
from tensorflow_with_latest_papers import highway_network_modern
from my.tensorflow.nn import highway_network


class Model(object):
    def __init__(self, nums_chars, nums_tags, buckets_char, counts=None, batch_size=10, tag_scheme='BIES', crf=1,
                 metric='F1-score', ngram=None, co_train=False, highway_layers=1, lambda0=0, lambda1=0,
                 char_freq_loss=False):
        self.ngram = ngram
        self.gram_layers = []

        # 字符种类数目
        self.nums_chars = nums_chars
        # 标签种类数目，例如[18]，至于为什么要用数组多此一举不清楚
        self.nums_tags = nums_tags
        # 每个 bucket 中句子的长度
        self.buckets_char = buckets_char
        # 训练集中每个 bucket 中的句子个数
        self.counts = counts
        self.tag_scheme = tag_scheme
        # 默认为1，即使用一阶条件随机场
        self.crf = crf
        self.emb_layer = None
        self.batch_size = batch_size
        self.l_rate = None
        self.decay = None
        self.train_steps = None
        self.saver = None
        self.decode_holders = None
        self.scores = None
        self.params = None
        self.drop_out = None
        self.drop_out_v = None
        # 默认是 F1-score
        self.metric = metric
        self.updates = []
        self.bucket_dit = {}
        # shape = (bucket数量，每个 bucket 中的句子数量，句子长度)
        self.input_v = []
        self.input_w = []
        self.input_p = None
        # LSTM 经全连接后的输出
        self.output = []
        # 标签，ground truth
        self.output_ = []
        self.output_p = []
        self.output_w = []
        self.output_w_ = []

        self.co_train = co_train
        self.highway_layers = highway_layers
        self.char_freq_loss = char_freq_loss

        if self.char_freq_loss:
            self.char_freq_predictions = []
            self.char_freq_groundtruthes = []
            self.lambda1 = lambda1

        if self.co_train:
            self.lm_fw_predictions = []
            self.lm_bw_predictions = []
            self.lm_fw_groundtruthes = []
            self.lm_bw_groundtruthes = []
            self.lambda0 = lambda0

        self.summaries = []
        # 使用 viterbi 解码
        if self.crf > 0:
            self.transition_char = []
            for i in range(len(self.nums_tags)):
                self.transition_char.append(
                    # 标签转移矩阵，为什么要额外加一呢？
                    tf.get_variable('transitions_char' + str(i), [self.nums_tags[i] + 1, self.nums_tags[i] + 1])
                )

        self.all_metrics = ['Precision', 'Recall', 'F1-score', 'True-Negative-Rate', 'Boundary-F1-score']

        while len(self.buckets_char) > len(self.counts):
            self.counts.append(1)

        # 获取每一个 batch 的大小
        # 一个 bucket 就是一个 batch，如果 bucket 中句子的个数小于设定的 batch_size，则对应 batch 的大小就是 bucket 中的句子个数，否则是 batch_size。
        # 即限制了 batch 最大为 batch_size
        self.real_batches = toolbox.get_real_batch(self.counts, self.batch_size)
        self.losses = []

    def main_graph(self, trained_model, scope, emb_dim, gru, rnn_dim, rnn_num, drop_out=0.5, emb=None,
                   ngram_embedding=None):
        """
        :param trained_model:
        :param scope:
        :param emb_dim:
        :param gru:
        :param rnn_dim:
        :param rnn_num:
        :param drop_out:
        :param emb:
        :return:
        """
        # trained_model: 模型存储路径
        if trained_model is not None:
            param_dic = {'nums_chars': self.nums_chars, 'nums_tags': self.nums_tags, 'tag_scheme': self.tag_scheme,
                         'crf': self.crf, 'emb_dim': emb_dim, 'gru': gru, 'rnn_dim': rnn_dim,
                         'rnn_num': rnn_num, 'drop_out': drop_out,
                         'buckets_char': self.buckets_char,
                         'ngram': self.ngram
                         }
            print "RNN dimension is %d" % rnn_dim
            print "RNN number is %d" % rnn_num
            print "Character embedding size is %d" % emb_dim
            # 存储模型超参数
            if self.metric == 'All':
                # rindex() 返回子字符串 str 在字符串中最后出现的位置
                # 截取模型文件名
                pindex = trained_model.rindex('/') + 1
                for m in self.all_metrics:
                    f_model = open(trained_model[:pindex] + m + '_' + trained_model[pindex:], 'w')
                    pickle.dump(param_dic, f_model)
                    f_model.close()
            else:
                f_model = open(trained_model, 'w')
                pickle.dump(param_dic, f_model)
                f_model.close()

        # define shared weights and variables

        dr = tf.placeholder(tf.float32, [], name='drop_out_holder')
        self.drop_out = dr
        self.drop_out_v = drop_out

        # 字向量层
        # 为什么字符数要加 500 ？
        # emb_dim 是每个字符的特征向量维度，可以通过命令行参数设置
        # weights 表示预训练的字向量，可以通过命令行参数设置
        self.emb_layer = EmbeddingLayer(self.nums_chars + 500, emb_dim, weights=emb, name='emb_layer')

        if self.ngram is not None:
            if ngram_embedding is not None:
                assert len(ngram_embedding) == len(self.ngram)
            else:
                ngram_embedding = [None for _ in range(len(self.ngram))]
            for i, n_gram in enumerate(self.ngram):
                self.gram_layers.append(EmbeddingLayer(n_gram + 1000 * (i + 2), emb_dim, weights=ngram_embedding[i],
                                                       name=str(i + 2) + 'gram_layer'))

        # 隐藏层，输入是前向 RNN 的输出加上 后向 RNN 的输出，所以输入维度为 rnn_dim * 2
        # 输出维度即标签个数
        tag_output_wrapper = TimeDistributed(
            HiddenLayer(rnn_dim * 2, self.nums_tags[0], activation='linear', name='tag_hidden'),
            name='tag_output_wrapper')

        if self.char_freq_loss:
            freq_output_wrapper = TimeDistributed(
                HiddenLayer(rnn_dim * 2, 1, activation='sigmoid', name='freq_hidden'),
                name='freq_output_wrapper')

        if self.co_train:
            lm_fw_wrapper = TimeDistributed(
                HiddenLayer(rnn_dim, self.nums_chars + 2, activation='linear', name='lm_fw_hidden'),
                name='lm_fw_wrapper')
            lm_bw_wrapper = TimeDistributed(
                HiddenLayer(rnn_dim, self.nums_chars + 2, activation='linear', name='lm_bw_hidden'),
                name='lm_bw_wrapper')

        # define model for each bucket
        # 每一个 bucket 中的句子长度不一样，所以需要定义单独的模型
        # bucket: bucket 中的句子长度
        for idx, bucket in enumerate(self.buckets_char):
            if idx == 1:
                # scope 是 tf.variable_scope("tagger", reuse=None, initializer=initializer)
                # 只需要设置一次 reuse，后面就都 reuse 了
                scope.reuse_variables()
            t1 = time()

            # 输入的句子，one-hot 向量
            # shape = （batch_size, 句子长度）
            input_sentences = tf.placeholder(tf.int32, [None, bucket], name='input_' + str(bucket))

            self.input_v.append([input_sentences])

            emb_set = []
            word_out = self.emb_layer(input_sentences)
            emb_set.append(word_out)

            if self.ngram is not None:
                for i in range(len(self.ngram)):
                    input_g = tf.placeholder(tf.int32, [None, bucket], name='input_g' + str(i) + str(bucket))
                    self.input_v[-1].append(input_g)
                    gram_out = self.gram_layers[i](input_g)
                    emb_set.append(gram_out)

            if len(emb_set) > 1:
                # 各种字向量直接 concat 起来（字向量、偏旁部首、n-gram、图像信息等）
                word_embeddings = tf.concat(axis=2, values=emb_set)

            else:
                word_embeddings = emb_set[0]

            # rnn_out 是前向 RNN 的输出和后向 RNN 的输出 concat 之后的值
            rnn_out_fw, rnn_out_bw = BiRNN(rnn_dim, p=dr, concat_output=False, gru=gru,
                                           name='BiLSTM' + str(bucket), scope='Tag-BiRNN')(word_embeddings,
                                                                                           input_sentences)

            tag_rnn_out_fw, tag_rnn_out_bw = rnn_out_fw, rnn_out_bw
            if self.co_train:
                if self.highway_layers > 0:
                    tag_rnn_out_fw = highway_network(rnn_out_fw, self.highway_layers, True, is_train=True,
                                                     scope="tag_fw")
                    tag_rnn_out_bw = highway_network(rnn_out_bw, self.highway_layers, True, is_train=True,
                                                     scope="tag_bw")
            tag_rnn_out = tf.concat(values=[tag_rnn_out_fw, tag_rnn_out_bw], axis=2)

            # 应用全连接层，Wx+b 得到最后的输出
            output = tag_output_wrapper(tag_rnn_out)
            # 为什么要 [output] 而不是 output 呢？
            self.output.append([output])

            self.output_.append([tf.placeholder(tf.int32, [None, bucket], name='tags' + str(bucket))])

            self.bucket_dit[bucket] = idx

            if self.co_train:
                # language model
                lm_rnn_out_fw, lm_rnn_out_bw = rnn_out_fw, rnn_out_bw
                if self.highway_layers > 0:
                    lm_rnn_out_fw = highway_network(rnn_out_fw, self.highway_layers, True, is_train=True,
                                                    scope="lm_fw")
                    lm_rnn_out_bw = highway_network(rnn_out_bw, self.highway_layers, True, is_train=True,
                                                    scope="lm_bw")

                self.lm_fw_predictions.append([lm_fw_wrapper(lm_rnn_out_fw)])
                self.lm_bw_predictions.append([lm_bw_wrapper(lm_rnn_out_bw)])
                self.lm_fw_groundtruthes.append(
                    [tf.placeholder(tf.int32, [None, bucket], name='lm_fw_targets' + str(bucket))])
                self.lm_bw_groundtruthes.append(
                    [tf.placeholder(tf.int32, [None, bucket], name='lm_bw_targets' + str(bucket))])

            if self.char_freq_loss:
                freq_rnn_out_fw, freq_rnn_out_bw = rnn_out_fw, rnn_out_bw
                if self.highway_layers > 0:
                    freq_rnn_out_fw = highway_network(rnn_out_fw, self.highway_layers, True, is_train=True,
                                                      scope="freq_fw")
                    freq_rnn_out_bw = highway_network(rnn_out_bw, self.highway_layers, True, is_train=True,
                                                      scope="freq_bw")
                freq_rnn_out = tf.concat(values=[freq_rnn_out_fw, freq_rnn_out_bw], axis=2)

                self.char_freq_groundtruthes.append(
                    [tf.placeholder(tf.float32, [None, bucket], name='freq_targets_%d' % bucket)])
                self.char_freq_predictions.append(
                    [freq_output_wrapper(freq_rnn_out)])

            print 'Bucket %d, %f seconds' % (idx + 1, time() - t1)

        assert \
            len(self.input_v) == len(self.output) and \
            len(self.output) == len(self.output_) and \
            len(self.output) == len(self.counts)

        self.params = tf.trainable_variables()

        self.saver = tf.train.Saver()

    def config(self, optimizer, decay, lr_v=None, momentum=None, clipping=False, max_gradient_norm=5.0):

        """

        :param optimizer: 优化函数，Adagrad
        :param decay: 学习率衰减率，0.05
        :param lr_v:  学习率，0.1
        :param momentum:
        :param clipping: 是否运用梯度裁剪（给梯度设置最大阈值）
        :param max_gradient_norm:
        """
        self.decay = decay
        print 'Training preparation...'

        print 'Defining loss...'

        if self.crf > 0:
            for i in range(len(self.input_v)):
                # 根据第 i 个 bucket 的输出和 ground truth，用 CRF 损失函数，计算损失函数值
                tagging_loss = losses.loss_wrapper(self.output[i], self.output_[i],
                                                   losses.crf_loss,
                                                   transitions=self.transition_char, nums_tags=self.nums_tags,
                                                   batch_size=self.real_batches[i])
                tagging_loss_summary = tf.summary.scalar('tagging loss %s' % i, tf.reduce_mean(tagging_loss))

                loss = tagging_loss
                loss_summary = [tagging_loss_summary]

                if self.co_train:
                    lm_loss = []
                    masks = tf.reshape(tf.cast(tf.sign(self.output_[i]), dtype=tf.float32), shape=[-1, self.buckets_char[i]])
                    for lm_fw_y, lm_fw_y_, lm_bw_y, lm_bw_y_ in zip(self.lm_fw_predictions[i],
                                                                    self.lm_fw_groundtruthes[i],
                                                                    self.lm_bw_predictions[i],
                                                                    self.lm_bw_groundtruthes[i]):
                        lm_fw_loss = tf.contrib.seq2seq.sequence_loss(lm_fw_y, lm_fw_y_, masks)
                        lm_bw_loss = tf.contrib.seq2seq.sequence_loss(lm_bw_y, lm_bw_y_, masks)
                        # lm_fw_loss = tf.reduce_sum(losses.sparse_cross_entropy(lm_fw_y, lm_fw_y_) * masks)
                        # lm_bw_loss = tf.reduce_sum(losses.sparse_cross_entropy(lm_bw_y, lm_bw_y_) * masks)
                        lm_loss.append(lm_fw_loss + lm_bw_loss)
                    lm_loss = tf.stack(lm_loss)
                    lm_loss_summary = tf.summary.scalar('language model loss %s' % i, tf.reduce_mean(lm_loss))

                    loss += self.lambda0 * lm_loss
                    loss_summary.append(lm_loss_summary)
                if self.char_freq_loss:
                    freq_loss = []
                    masks = tf.cast(tf.sign(self.output_[i]), dtype=tf.float32)
                    for freq_y, freq_y_ in zip(self.char_freq_predictions[i], self.char_freq_groundtruthes[i]):
                        freq_loss.append(tf.losses.mean_squared_error(freq_y_, tf.reshape(freq_y, tf.shape(freq_y_)),
                                                                      weights=tf.reshape(masks, tf.shape(freq_y_))))
                    freq_loss = tf.stack(freq_loss)
                    freq_loss_summary = tf.summary.scalar('char freq loss %s' % i, tf.reduce_mean(freq_loss))
                    loss += self.lambda1 * freq_loss
                    loss_summary.append(freq_loss_summary)
                self.losses.append(loss)
                self.summaries.append(loss_summary)

        else:
            # todo
            loss_function = losses.sparse_cross_entropy
            for output, output_ in zip(self.output, self.output_):
                bucket_loss = losses.loss_wrapper(output, output_, loss_function)
                self.losses.append(bucket_loss)

        l_rate = tf.placeholder(tf.float32, [], name='learning_rate_holder')
        self.l_rate = l_rate

        if optimizer == 'sgd':
            if momentum is None:
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=l_rate)
            else:
                optimizer = tf.train.MomentumOptimizer(learning_rate=l_rate, momentum=momentum)
        elif optimizer == 'adagrad':
            assert lr_v is not None
            optimizer = tf.train.AdagradOptimizer(learning_rate=l_rate)
        elif optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=l_rate)
        else:
            raise Exception('optimiser error')

        self.train_steps = []

        print 'Computing gradients...'

        for idx, l in enumerate(self.losses):
            t2 = time()
            if clipping:
                gradients = tf.gradients(l, self.params)
                clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
                train_step = optimizer.apply_gradients(zip(clipped_gradients, self.params))
            else:
                train_step = optimizer.minimize(l)
            print 'Bucket %d, %f seconds' % (idx + 1, time() - t2)
            self.train_steps.append(train_step)

    def decode_graph(self):
        self.decode_holders = []
        self.scores = []
        for bucket in self.buckets_char:
            decode_holders = []
            scores = []
            for nt in self.nums_tags:
                ob = tf.placeholder(tf.float32, [None, bucket, nt])
                trans = tf.placeholder(tf.float32, [nt + 1, nt + 1])
                nums_steps = ob.get_shape().as_list()[1]
                length = tf.placeholder(tf.int32, [None])
                b_size = tf.placeholder(tf.int32, [])
                small = -1000
                class_pad = tf.stack(small * tf.ones([b_size, nums_steps, 1]))
                observations = tf.concat(axis=2, values=[ob, class_pad])
                b_vec = tf.tile(([small] * nt + [0]), [b_size])
                b_vec = tf.cast(b_vec, tf.float32)
                b_vec = tf.reshape(b_vec, [b_size, 1, -1])
                e_vec = tf.tile(([0] + [small] * nt), [b_size])
                e_vec = tf.cast(e_vec, tf.float32)
                e_vec = tf.reshape(e_vec, [b_size, 1, -1])
                observations = tf.concat(axis=1, values=[b_vec, observations, e_vec])
                transitions = tf.reshape(tf.tile(trans, [b_size, 1]), [b_size, nt + 1, nt + 1])
                observations = tf.reshape(observations, [-1, nums_steps + 2, nt + 1, 1])
                observations = tf.transpose(observations, [1, 0, 2, 3])
                previous = observations[0, :, :, :]
                max_scores = []
                max_scores_pre = []
                alphas = [previous]
                for t in xrange(1, nums_steps + 2):
                    previous = tf.reshape(previous, [-1, nt + 1, 1])
                    current = tf.reshape(observations[t, :, :, :], [-1, 1, nt + 1])
                    alpha_t = previous + current + transitions
                    max_scores.append(tf.reduce_max(alpha_t, axis=1))
                    max_scores_pre.append(tf.argmax(alpha_t, axis=1))
                    alpha_t = tf.reshape(Forward.log_sum_exp(alpha_t, axis=1), [-1, nt + 1, 1])
                    alphas.append(alpha_t)
                    previous = alpha_t
                max_scores = tf.stack(max_scores, axis=1)
                max_scores_pre = tf.stack(max_scores_pre, axis=1)
                decode_holders.append([ob, trans, length, b_size])
                scores.append((max_scores, max_scores_pre))
            self.decode_holders.append(decode_holders)
            self.scores.append(scores)

    def train(self, t_x, t_y, v_x, v_y, idx2tag, idx2char, sess,
              epochs, trained_model, lr=0.05, decay=0.05, decay_step=1, tag_num=1):
        """
        :param t_x: b_train_x
        :param t_y: b_train_y
        :param v_x: b_dev_x
        :param v_y: b_dev_y
        :param idx2tag:
        :param idx2char:
        :param sess:
        :param epochs: 训练轮数
        :param trained_model: 训练好的模型参数
        :param lr: 学习率
        :param decay: 学习率衰减率
        :param decay_step:
        :param tag_num: 标签种类个数
        """
        log_dir = "./train_log"
        shutil.rmtree(log_dir)
        train_writer = tf.summary.FileWriter(log_dir, sess[0].graph)

        lr_r = lr

        best_epoch, best_score, best_seg, best_pos, c_tag, c_seg, c_score = {}, {}, {}, {}, {}, {}, {}

        pindex = 0

        metric = self.metric

        # 每种衡量标准下都有对应的最佳结果
        for m in self.all_metrics:
            best_epoch[m] = 0
            best_score[m] = 0

            best_seg[m] = 0
            best_pos[m] = 0

            c_tag[m] = 0
            c_seg[m] = 0
            c_score[m] = 0

        v_y = toolbox.merge_bucket(v_y)
        v_y = toolbox.unpad_zeros(v_y)

        gold = toolbox.decode_tags(v_y, idx2tag, self.tag_scheme)
        # 0 是字符本身，1 是偏旁部首，2、3 分别是 2gram 和 3gram
        input_chars = toolbox.merge_bucket([v_x[0]])

        chars = toolbox.decode_chars(input_chars[0], idx2char)
        # 正确答案，实际上直接读取 dev.txt 即可得到，不知为何还要这么麻烦通过各种 ID 转换获取
        gold_out = toolbox.generate_output(chars, gold, self.tag_scheme)

        for epoch in range(epochs):
            print 'epoch: %d' % (epoch + 1)
            t = time()
            # 在 decay_step 轮之后，衰减学习率
            if epoch % decay_step == 0 and decay > 0:
                lr_r = lr / (1 + decay * (epoch / decay_step))
            # data_list: shape=(5,bucket 数量，bucket 中句子个数，句子长度)
            data_list = t_x + t_y
            # samples: shape=(bucket 数量，5, bucket 中句子个数，句子长度)，相当于置换了 data_list 中的 shape[0] 和 shape[1]
            samples = zip(*data_list)

            random.shuffle(samples)

            # 遍历每一个 bucket
            for sample in samples:
                # sample: shape=(5, bucket 中句子个数，句子长度)
                # 当前 bucket 中的句子长度
                c_len = len(sample[0][0])
                # 当前 bucket 的序号
                idx = self.bucket_dit[c_len]
                real_batch_size = self.real_batches[idx]
                # 当前 bucket 的模型的输入和输出（注意每个 bucket 都有一个单独的模型）
                if self.char_freq_loss:
                    model_placeholders = self.input_v[idx] + self.char_freq_groundtruthes[idx] + self.output_[idx]
                else:
                    model_placeholders = self.input_v[idx] + self.output_[idx]

                if self.co_train:
                    model_placeholders += self.lm_fw_groundtruthes[idx] + self.lm_bw_groundtruthes[idx]

                # sess[0] 是 main_sess, sess[1] 是 decode_sess(如果使用 CRF 的话)
                # 训练当前的 bucket，这个函数里面才真正地为模型填充了数据并运行(以 real_batch_size 为单位，将 bucket 中的句子依次喂给模型)
                # 被 sess.run 的是 config=self.train_step[idx]，train_step[idx] 就会触发 BP 更新参数了
                Batch.train(sess=sess[0], placeholders=model_placeholders, batch_size=real_batch_size,
                            train_step=self.train_steps[idx], loss=self.losses[idx],
                            lr=self.l_rate, lrv=lr_r, dr=self.drop_out, drv=self.drop_out_v, data=list(sample),
                            # debug_variable=[self.lm_output[idx], self.lm_output_[idx], self.output[idx], self.output_[idx]],
                            verbose=False,
                            log_writer=train_writer,
                            single_summary=self.summaries[idx], epoch_index=epoch)

            predictions = []
            # 遍历每个 bucket, 用开发集测试准确率
            for v_b_x in zip(*v_x):

                # v_b_x: shape=(4,bucket 中句子个数，句子长度)
                c_len = len(v_b_x[0][0])
                idx = self.bucket_dit[c_len]

                if self.char_freq_loss:
                    model_placeholders = self.input_v[idx] + self.char_freq_groundtruthes[idx] + self.output[idx]
                else:
                    model_placeholders = self.input_v[idx] + self.output[idx]

                b_prediction = self.predict(data=v_b_x, sess=sess, placeholders=model_placeholders,
                                            index=idx, batch_size=100)
                b_prediction = toolbox.decode_tags(b_prediction, idx2tag, self.tag_scheme)
                predictions.append(b_prediction)

            predictions = zip(*predictions)
            predictions = toolbox.merge_bucket(predictions)

            prediction_out = toolbox.generate_output(chars, predictions, self.tag_scheme)

            scores = toolbox.evaluator(prediction_out, gold_out, metric=metric, verbose=True, tag_num=tag_num)
            scores = np.asarray(scores)

            # Score_seg * Score_seg&tag
            c_seg['Precision'] = scores[0]
            c_seg['Recall'] = scores[1]
            c_seg['F1-score'] = scores[2]
            c_seg['True-Negative-Rate'] = scores[6]
            c_seg['Boundary-F1-score'] = scores[10]
            if self.tag_scheme != 'seg':
                c_tag['Precision'] = scores[3]
                c_tag['Recall'] = scores[4]
                c_tag['F1-score'] = scores[5]
                c_tag['True-Negative-Rate'] = scores[7]
                c_tag['Boundary-F1-score'] = scores[13]
            else:
                c_tag['Precision'] = 1
                c_tag['Recall'] = 1
                c_tag['F1-score'] = 1
                c_tag['True-Negative-Rate'] = 1
                c_tag['Boundary-F1-score'] = 1

            if metric == 'All':
                for m in self.all_metrics:
                    print 'Segmentation ' + m + ': %f' % c_seg[m]
                    print 'POS Tagging ' + m + ': %f\n' % c_tag[m]
                pindex = trained_model.rindex('/') + 1
            else:
                print 'Segmentation ' + metric + ': %f' % c_seg[metric]
                if self.tag_scheme != 'seg':
                    print 'POS Tagging ' + metric + ': %f\n' % c_tag[metric]

            for m in self.all_metrics:
                c_score[m] = c_seg[m] * c_tag[m]

            if metric == 'All':
                for m in self.all_metrics:
                    if c_score[m] > best_score[m] and epoch > 4:
                        best_epoch[m] = epoch + 1
                        best_score[m] = c_score[m]
                        best_seg[m] = c_seg[m]
                        best_pos[m] = c_tag[m]
                        self.saver.save(sess[0], trained_model[:pindex] + m + '_' + trained_model[pindex:],
                                        write_meta_graph=False)

            elif c_score[metric] > best_score[metric] and epoch > 4:
                best_epoch[metric] = epoch + 1
                best_score[metric] = c_score[metric]
                best_seg[metric] = c_seg[metric]
                best_pos[metric] = c_tag[metric]
                self.saver.save(sess[0], trained_model, write_meta_graph=False)
            print 'Time consumed: %d seconds' % int(time() - t)
        print 'Training is finished!'

        if metric == 'All':
            for m in self.all_metrics:
                print 'Best segmentation ' + m + ': %f' % best_seg[m]
                print 'Best POS Tagging ' + m + ': %f' % best_pos[m]
                print 'Best epoch: %d\n' % best_epoch[m]
        else:
            print 'Best segmentation ' + metric + ': %f' % best_seg[metric]
            print 'Best POS Tagging ' + metric + ': %f' % best_pos[metric]
            print 'Best epoch: %d\n' % best_epoch[metric]

    def define_updates(self, new_chars, emb_path, char2idx, new_grams=None, ng_emb_path=None, gram2idx=None):

        self.nums_chars += len(new_chars)

        if emb_path is not None:

            old_emb_weights = self.emb_layer.embeddings
            emb_dim = old_emb_weights.get_shape().as_list()[1]
            emb_len = old_emb_weights.get_shape().as_list()[0]
            new_emb = toolbox.get_new_embeddings(new_chars, emb_dim, emb_path)
            n_emb_sh = new_emb.get_shape().as_list()
            if len(n_emb_sh) > 1:
                new_emb_weights = tf.concat(axis=0, values=[old_emb_weights[:len(char2idx) - len(new_chars)], new_emb,
                                                            old_emb_weights[len(char2idx):]])
                if new_emb_weights.get_shape().as_list()[0] > emb_len:
                    new_emb_weights = new_emb_weights[:emb_len]
                assign_op = old_emb_weights.assign(new_emb_weights)
                self.updates.append(assign_op)

        if self.ngram is not None and ng_emb_path is not None:
            old_gram_weights = [ng_layer.embeddings for ng_layer in self.gram_layers]
            ng_emb_dim = old_gram_weights[0].get_shape().as_list()[1]
            new_ng_emb = toolbox.get_new_ng_embeddings(new_grams, ng_emb_dim, ng_emb_path)
            for i in range(len(old_gram_weights)):
                new_ng_weight = tf.concat(axis=0, values=[old_gram_weights[i][:len(gram2idx[i]) - len(new_grams[i])],
                                                          new_ng_emb[i], old_gram_weights[i][len(gram2idx[i]):]])
                assign_op = old_gram_weights[i].assign(new_ng_weight)
                self.updates.append(assign_op)

    def run_updates(self, sess, weight_path):
        self.saver.restore(sess, weight_path)
        for op in self.updates:
            sess.run(op)

        print 'Loaded.'

    def test(self, sess, t_x, t_y, idx2tag, idx2char, outpath=None, ensemble=None, batch_size=200, tag_num=1):

        t_y = toolbox.unpad_zeros(t_y)
        gold = toolbox.decode_tags(t_y, idx2tag, self.tag_scheme)
        chars = toolbox.decode_chars(t_x[0], idx2char)
        gold_out = toolbox.generate_output(chars, gold, self.tag_scheme)

        prediction = self.predict(data=t_x, sess=sess, placeholders=self.input_v[0] + self.output[0], index=0,
                                  ensemble=ensemble, batch_size=batch_size)
        prediction = toolbox.decode_tags(prediction, idx2tag, self.tag_scheme)
        prediction_out = toolbox.generate_output(chars, prediction, self.tag_scheme)

        scores = toolbox.evaluator(prediction_out, gold_out, metric='All', verbose=True, tag_num=tag_num)

        print 'Best scores: '

        print 'Segmentation F1-score: %f' % scores[2]
        print 'Segmentation Precision: %f' % scores[0]
        print 'Segmentation Recall: %f' % scores[1]
        print 'Segmentation True Negative Rate: %f' % scores[6]
        print 'Segmentation Boundary-F1-score: %f\n' % scores[10]

        print 'Joint POS tagging F-score: %f' % scores[5]
        print 'Joint POS tagging Precision: %f' % scores[3]
        print 'Joint POS tagging Recall: %f' % scores[4]
        print 'Joint POS True Negative Rate: %f' % scores[7]
        print 'Joint POS tagging Boundary-F1-score: %f\n' % scores[13]

        if outpath is not None:
            final_out = prediction_out[0]
            toolbox.printer(final_out, outpath)

    def tag(self, sess, r_x, idx2tag, idx2char, char2idx, outpath='out.txt', ensemble=None, batch_size=200,
            large_file=False):

        chars = toolbox.decode_chars(r_x[0], idx2char)

        char_num = len(set(char2idx.values()))

        r_x = np.asarray(r_x)

        r_x[0][r_x[0] > char_num - 1] = char2idx['<UNK>']

        c_len = len(r_x[0][0])
        idx = self.bucket_dit[c_len]

        real_batch = int(batch_size * 300 / c_len)

        prediction = self.predict(data=r_x, sess=sess, placeholders=self.input_v[idx] + self.output[idx], index=idx,
                                  ensemble=ensemble, batch_size=real_batch)
        prediction = toolbox.decode_tags(prediction, idx2tag, self.tag_scheme)
        prediction_out = toolbox.generate_output(chars, prediction, self.tag_scheme)

        final_out = prediction_out[0]
        if large_file:
            return final_out
        else:
            toolbox.printer(final_out, outpath)

    def predict(self, data, sess, placeholders, index=None, argmax=True, batch_size=100,
                pt_h=None, pt=None, ensemble=None, verbose=False):

        """
        预测标签
        :param data: 一个 bucket 中的所有句子
        :param sess: [tf.Session]，两个，一个是训练的，一个是解码的
        :param placeholders: [tf.placeholder]，接受 feed 给模型的数据
        :param index: 当前 bucket 的序号
        :param argmax:
        :param batch_size:
        :param pt_h:
        :param pt:
        :param ensemble:
        :param verbose:
        :return:
        """
        if self.crf:
            assert index is not None
            predictions = Batch.predict(sess=sess[0], decode_sess=sess[1], placeholders=placeholders,
                                        transitions=self.transition_char, crf=self.crf, scores=self.scores[index],
                                        decode_holders=self.decode_holders[index], argmax=argmax, batch_size=batch_size,
                                        data=data, dr=self.drop_out, ensemble=ensemble,
                                        verbose=verbose)
        else:
            predictions = Batch.predict(sess=sess[0], placeholders=placeholders, crf=self.crf, argmax=argmax,
                                        batch_size=batch_size,
                                        data=data, dr=self.drop_out, ensemble=ensemble, verbose=verbose)
        return predictions
