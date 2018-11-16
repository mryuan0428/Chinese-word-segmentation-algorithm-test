# -*- coding: utf-8 -*-
import random
import toolbox
import numpy as np
import datetime
import tensorflow as tf


def train(sess, placeholders, batch_size, train_step, loss, lr, lrv, data, debug_variable=None, dr=None, drv=None,
          verbose=False, log_writer=None, single_summary=None,
          epoch_index=0):
    """
    训练单个 bucket 的模型
    :param epoch_index:
    :param single_summary:
    :param log_writer:
    :param debug_variable:
    :param loss:
    :param sess: tf.Session
    :param placeholders: [tf.placeholder]，总共5个，表示一个句子的字符本身、偏旁部首、2gram、3gram、对应标签
    :param batch_size:
    :param train_step: 目标 bucket 的 train_step, optimizer.apply_gradients()
    :param lr: 初始学习率
    :param lrv: 衰减后的学习率
    :param data: 当前 bucket 中的所有句子，shape=(5, bucket 中句子数量，句子长度)
    :param dr: drop_out
    :param drv: =drop_out
    :param verbose:
    """
    if debug_variable is not None:
        session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        np.set_printoptions(threshold=10000)
    # len(data)=5，表示一个句子的字符本身、对应标签
    if len(data) != len(placeholders):
        lm_fw_target = []
        lm_bw_target = []
        for sentence in data[0]:
            lm_fw_target.append(np.append(sentence[1:], 0))
            lm_bw_target.append(np.append([0],sentence[:-1]))
        data.append(lm_fw_target)
        data.append(lm_bw_target)
    assert len(data) == len(placeholders)
    num_items = len(data)
    samples = zip(*data)
    random.shuffle(samples)
    start_idx = 0
    n_samples = len(samples)
    placeholders.append(lr)
    if dr is not None:
        placeholders.append(dr)
    while start_idx < len(samples):
        if verbose:
            print '%s : %d of %d' % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), start_idx, n_samples)
        next_batch_samples = samples[start_idx:start_idx + batch_size]
        real_batch_size = len(next_batch_samples)
        if real_batch_size < batch_size:
            next_batch_samples.extend(samples[:batch_size - real_batch_size])
        holders = []
        for item in range(num_items):
            holders.append([s[item] for s in next_batch_samples])
        holders.append(lrv)
        if dr is not None:
            holders.append(drv)
        feed_dict = {m: h for m, h in zip(placeholders, holders)}
        if debug_variable is not None:
            debug_info = sess.run([train_step, loss] + debug_variable, feed_dict=feed_dict)
            # masks = tf.cast(tf.sign(data[0]), dtype=tf.float32)
            print "data and loss"
            masks = tf.cast(tf.sign(data[0][0]), dtype=tf.float32)
            lengths = tf.reduce_sum(tf.sign(data[0][0]), axis=0)
            print "masks", tf.Tensor.eval(masks, session=session)
            print "lengths", tf.Tensor.eval(lengths, session=session)
            for i in range(0, np.min([len(data[0]), len(debug_info[2][0]), len(debug_info[3][0])])):
                for (y, y_) in zip(debug_info[2][0][i], debug_info[3][0][i]):
                    lm_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=y_)
                    lm_loss_value = tf.Tensor.eval(lm_loss, session=session)
                    if np.isnan(lm_loss_value) or True:
                        print "lm_loss:", lm_loss_value
                        print "y:", np.array2string(y, precision=2, separator=',', suppress_small=True)
                        print "y_:", y_

            print "total loss:", debug_info[1]

        else:
            if single_summary is not None and log_writer is not None:
                loss_value, summary, _ = sess.run([loss, single_summary, train_step], feed_dict=feed_dict)
                log_step = start_idx + len(samples)*epoch_index
                if verbose:
                    print "log step: %d" % log_step
                for item in summary:
                    log_writer.add_summary(item, log_step)
            else:
                _, loss_value = sess.run([train_step, loss], feed_dict=feed_dict)
        start_idx += batch_size


def predict(sess, placeholders, data, dr=None, transitions=None, crf=True, decode_sess=None, scores=None, decode_holders=None,
            argmax=True, batch_size=100, ensemble=False, verbose=False):
    en_num = None
    if ensemble:
        en_num = len(sess)
    # 输入向量是4个，字符、偏旁、2gram、3gram
    num_items = len(data)
    input_v = placeholders[:num_items]
    if dr is not None:
        input_v.append(dr)
    # 预测向量1个
    predictions = placeholders[num_items:]
    # output = [[]]
    output = [[] for _ in range(len(predictions))]
    samples = zip(*data)
    start_idx = 0
    n_samples = len(samples)
    if crf:
        trans = []
        for i in range(len(predictions)):
            if ensemble:
                en_trans = 0
                for en_sess in sess:
                    en_trans += en_sess.run(transitions[i])
                trans.append(en_trans/en_num)
            else:
                trans.append(sess.run(transitions[i]))
    while start_idx < n_samples:
        if verbose:
            print '%d' % (start_idx*100/n_samples) + '%'
        next_batch_input = samples[start_idx:start_idx + batch_size]
        batch_size = len(next_batch_input)
        holders = []
        for item in range(num_items):
            holders.append([s[item] for s in next_batch_input])
        if dr is not None:
            holders.append(0.0)
        # length_holder = tf.cast(tf.pack(holders[0]), dtype=tf.int32)
        # length = tf.reduce_sum(tf.sign(length_holder), reduction_indices=1)
        length = np.sum(np.sign(holders[0]), axis=1)
        length = length.astype(int)
        if crf:
            assert transitions is not None and len(transitions) == len(predictions) and len(scores) == len(decode_holders)
            for i in range(len(predictions)):
                if ensemble:
                    en_obs = 0
                    for en_sess in sess:
                        en_obs += en_sess.run(predictions[i], feed_dict={i: h for i, h in zip(input_v, holders)})
                    ob = en_obs/en_num
                else:
                    ob = sess.run(predictions[i], feed_dict={i: h for i, h in zip(input_v, holders)})
                # trans = sess.run(transitions[i])
                pre_values = [ob, trans[i], length, batch_size]
                assert len(pre_values) == len(decode_holders[i])
                max_scores, max_scores_pre = decode_sess.run(scores[i], feed_dict={i: h for i, h in zip(decode_holders[i], pre_values)})
                output[i].extend(toolbox.viterbi(max_scores, max_scores_pre, length, batch_size))
        elif argmax:
            for i in range(len(predictions)):
                pre = sess.run(predictions[i], feed_dict={i: h for i, h in zip(input_v, holders)})
                pre = np.argmax(pre, axis=2)
                pre = pre.tolist()
                pre = toolbox.trim_output(pre, length)
                output[i].extend(pre)
        else:
            for i in range(len(predictions)):
                pre = sess.run(predictions[i], feed_dict={i: h for i, h in zip(input_v, holders)})
                pre = pre.tolist()
                pre = toolbox.trim_output(pre, length)
                output[i].extend(pre)
        start_idx += batch_size
    return output


