# -*- coding: utf-8 -*-
import tensorflow as tf

from layers import Forward


def cross_entropy(y, y_, nums_tags):
    one_hot_y_ = tf.contrib.layers.one_hot_encoding(y_, nums_tags)
    one_hot_y_ = tf.reshape(one_hot_y_, [-1, nums_tags])
    y = tf.reshape(y, [-1, nums_tags])
    # Computes softmax cross entropy between logits and labels
    # While the classes are mutually exclusive, their probabilities need not be.
    # All that is required is that each row of labels is a valid probability distribution
    # WARNING: This op expects unscaled logits, since it performs a softmax on logits internally for efficiency.
    # Do not call this op with the output of softmax, as it will produce incorrect results.
    return tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=one_hot_y_)


def mean_square(y, y_):
    return tf.reduce_mean(tf.square(y_ - y))


def sparse_cross_entropy(y, y_):
    # Computes sparse softmax cross entropy between logits and labels
    # NOTE: For this operation, the probability of a given label is considered exclusive.
    # That is, soft classes are not allowed, and the labels vector must provide a single
    # specific index for the true class for each row of logits (each minibatch entry).
    return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=y_)


def sparse_cross_entropy_with_weights(y, y_, weights=None, average_cross_steps=True):
    if weights is None:
        weights = tf.cast(tf.sign(y_), tf.float32)
    out = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=y_)
    if average_cross_steps:
        weights_sum = tf.reduce_sum(weights, axis=0)
        return out * weights / (weights_sum + 1e-12)
    else:
        return out * weights


def sequence_loss_by_example(logits, targets, weights=None, average_across_timesteps=True,
                             softmax_loss_function=None, name=None):
    """Weighted cross-entropy loss for a sequence of logits (per example).
    Args:
    logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
    targets: List of 1D batch-sized int32 Tensors of the same length as logits.
    weights: List of 1D batch-sized float-Tensors of the same length as logits.
    average_across_timesteps: If set, divide the returned cost by the total label weight.
        softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
        to be used instead of the standard softmax (the default if this is None).
        name: Optional name for this operation, default: "sequence_loss_by_example".
    Returns:
        1D batch-sized float Tensor: The log-perplexity for each sequence.
    Raises:
        ValueError: If len(logits) is different from len(targets) or len(weights).
    """
    if len(targets) != len(logits) or len(weights) != len(logits):
        raise ValueError("Lengths of logits, weights, and targets must be the same " "%d, %d, %d." % (
        len(logits), len(weights), len(targets)))
    with tf.name_scope(name + "sequence_loss_by_example"):
        log_perp_list = []
        for logit, target, weight in zip(logits, targets, weights):
            if softmax_loss_function is None:
                target = tf.reshape(target, [-1])
                crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=target)
            else:
                crossent = softmax_loss_function(logit, target)
            log_perp_list.append(crossent * weight)
        log_perps = tf.add_n(log_perp_list)
        if average_across_timesteps:
            total_size = tf.add_n(weights)
            total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
            log_perps /= total_size
    return log_perps


def crf_loss(y, y_, transitions, nums_tags, batch_size):
    """
    计算 CRF 损失函数值
    :param y: tag 预测值，shape = (batch_size, 句子长度，标签数量)，即每个句子中各个字符对应的标签概率
    :param y_: tag ground truth，shape=(batch_size, 句子长度)
    :param transitions: 标签转移矩阵, shape=(标签数量+1， 标签数量+1)
    :param nums_tags: 标签数量
    :param batch_size:  real batch size
    :return:
    """
    tag_scores = y
    # 句子长度，即解码步数
    nums_steps = len(tf.unstack(tag_scores, axis=1))
    # shape = (batch_size, 句子长度)
    masks = tf.cast(tf.sign(y_), dtype=tf.float32)
    lengths = tf.reduce_sum(tf.sign(y_), axis=1)
    tag_ids = y_
    # shape = (batch_size, 1)，实际上就是将 a list of arrays/tensors 变成一个 tensor
    b_id = tf.stack([[nums_tags]] * batch_size)
    # e_id = tf.pack([[0]] * batch_size)
    # shape=(batch_size, 句子长度+1)，因为 tag_ids.shape=(batch_size, 句子长度), b_id.shape=(batch_size, 1)
    padded_tag_ids = tf.concat(axis=1, values=[b_id, tag_ids])
    # tf.slice() 的作用是将输入标签序列中的每个标签两两成对切开，表示的就是标签间的转移关系
    # 每一个 slice 得到的是 shape=(batch_size, 2) 的 tensor，总共有 nums_steps 个，stack 到一个 tensor 里得到
    # shape = (batch_size, 句子长度，2)
    idx_tag_ids = tf.stack([tf.slice(padded_tag_ids, [0, i], [-1, 2]) for i in range(nums_steps)], axis=1)
    tag_ids = tf.contrib.layers.one_hot_encoding(tag_ids, nums_tags)
    point_score = tf.reduce_sum(tag_scores * tag_ids, axis=2)
    point_score *= masks
    # Save for future
    # trans_score = tf.gather_nd(transitions, idx_tag_ids)
    trans_sh = tf.stack(transitions.get_shape())
    trans_sh = tf.cumprod(trans_sh, exclusive=True, reverse=True)
    flat_tag_ids = tf.reduce_sum(trans_sh * idx_tag_ids, axis=2)
    trans_score = tf.gather(tf.reshape(transitions, [-1]), flat_tag_ids)
    # extend_mask = tf.concat(1, [tf.ones([batch_size, 1]), masks])
    extend_mask = masks
    trans_score *= extend_mask
    target_path_score = tf.reduce_sum(point_score) + tf.reduce_sum(trans_score)
    total_path_score, _, _ = Forward(tag_scores, transitions, nums_tags, lengths, batch_size)()
    tagging_loss = - (target_path_score - total_path_score)

    return tagging_loss


def loss_wrapper(y, y_,
                 loss_function, transitions=None, nums_tags=None, batch_size=None, weights=None,
                 average_cross_steps=True):
    """

    :param y:
    :param y_:
    :param loss_function:
    :param transitions:
    :param nums_tags:
    :param batch_size:
    :param weights:
    :param average_cross_steps:
    :return:
    """
    assert len(y) == len(y_)
    total_loss = []
    if loss_function is crf_loss:
        assert len(y) == len(transitions) and len(transitions) == len(nums_tags) and batch_size is not None
        for sy, sy_,  transition, tags in \
                zip(y, y_, transitions, nums_tags):
            tagging_loss = loss_function(sy, sy_, transition, tags, batch_size)
            total_loss.append(tagging_loss)
    elif loss_function is cross_entropy:
        assert len(y) == len(nums_tags)
        for sy, sy_, tags in zip(y, y_, nums_tags):
            total_loss.append(loss_function(sy, sy_, tags))
    elif loss_function is sparse_cross_entropy:
        for sy, sy_ in zip(y, y_):
            total_loss.append(loss_function(sy, sy_))
    elif loss_function is sparse_cross_entropy_with_weights:
        assert len(y) == len(nums_tags)
        for sy, sy_, tags in zip(y, y_):
            total_loss.append(
                tf.reshape(loss_function(sy, sy_, weights=weights, average_cross_steps=average_cross_steps), [-1]))
    else:
        for sy, sy_ in zip(y, y_):
            total_loss.append(tf.reshape(loss_function(sy, sy_), [-1]))
    return tf.stack(total_loss)
