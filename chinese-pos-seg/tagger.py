# -*- coding: utf-8 -*-
import sys
import toolbox
import os
import argparse
import tensorflow as tf
from bucket_model import Model
from time import time
import cPickle as pickle
import codecs

parser = argparse.ArgumentParser(description='A tagger for joint Chinese segmentation and POS tagging.')
parser.add_argument('action', default='tag', choices=['train', 'test', 'tag'], help='train, test or tag')
parser.add_argument('-p', '--path', default=None, help='Path of the workstation')

parser.add_argument('-t', '--train', default=None, help='File for training')
parser.add_argument('-d', '--dev', default=None, help='File for validation')
parser.add_argument('-e', '--test', default=None, help='File for evaluation')
parser.add_argument('-r', '--raw', default=None, help='Raw file for tagging')

parser.add_argument('-m', '--model', default='trained_model', help='Name of the trained model')
parser.add_argument('-tg', '--tag_scheme', default='BIES', help='Tagging scheme')
parser.add_argument('-crf', '--crf', default=1, type=int, help='Using CRF interface')

parser.add_argument('-ng', '--ngram', default=3, type=int, help='Using ngrams')
parser.add_argument('--ngram_embeddings', default=None, help='Pre-trained ngram embeddings')
parser.add_argument('-emb', '--embeddings', default=None, help='Path and name of pre-trained char embeddings')
parser.add_argument('-ed', '--embeddings_dimension', default=64, type=int, help='Dimension of the embeddings')

parser.add_argument('-bt', '--bucket_size', default=10, type=int, help='Bucket size')

parser.add_argument('-gru', '--gru', default=False, help='Use GRU as the recurrent cell', action='store_true')
parser.add_argument('-rnn', '--rnn_cell_dimension', default=200, type=int, help='Dimension of the RNN cells')
parser.add_argument('-layer', '--rnn_layer_number', default=1, type=int, help='Numbers of the RNN layers')

parser.add_argument('-dr', '--dropout_rate', default=0.5, type=float, help='Dropout rate')

parser.add_argument('-iter', '--epochs', default=30, type=int, help='Numbers of epochs')
parser.add_argument('-op', '--optimizer', default='adagrad', help='Optimizer')
parser.add_argument('-lr', '--learning_rate', default=0.1, type=float, help='Initial learning rate')
parser.add_argument('-ld', '--decay_rate', default=0.05, type=float, help='Learning rate decay')
parser.add_argument('-mt', '--momentum', default=None, type=float, help='Momentum')

parser.add_argument('-om', '--op_metric', default='F1-score', help='Optimization metric')

parser.add_argument('-cp', '--clipping', default=False, help='Apply Gradient Clipping', action='store_true')

parser.add_argument("-tb", "--train_batch", help="Training batch size", default=10, type=int)
parser.add_argument("-eb", "--test_batch", help="Testing batch size", default=200, type=int)
parser.add_argument("-rb", "--tag_batch", help="Tagging batch size", default=200, type=int)

parser.add_argument('-opth', '--output_path', default=None, help='Output path')

parser.add_argument('-ens', '--ensemble', default=False, help='Ensemble several weights', action='store_true')

parser.add_argument('-tl', '--tag_large', default=False, help='Tag (very) large file', action='store_true')

parser.add_argument('-debug', '--debug', default=False, help='Print debug information', action='store_true')

parser.add_argument('-ls', '--large_size', default=200000, type=int, help='Tag (very) large file')

parser.add_argument('--co_train', action='store_true', default=False, help='cotrain language model')
parser.add_argument('--lambda0', type=float, default=1, help='language model loss weight')
parser.add_argument('--lambda1', type=float, default=1, help='char freq loss weight')
parser.add_argument('--char_freq_loss', action='store_true', default=False,
                    help="use character frequency as auxiliary loss")
parser.add_argument('--highway_layers', type=int, default=1, help='number of highway layers')

parser.add_argument('--patience', type=int, default=15, help='patience for early stop')
args = parser.parse_args()

sys = reload(sys)
sys.setdefaultencoding('utf-8')
print 'Encoding: ', sys.getdefaultencoding()

gpu_config = "/gpu:0"
if args.action == 'train':
    assert args.path is not None
    assert args.train is not None
    assert args.dev is not None
    path = args.path
    train_file = args.train
    dev_file = args.dev
    model_file = args.model
    print 'Reading data......'

    # 统计文本信息
    ngram = args.ngram
    if not os.path.isfile(path + '/' + str(ngram) + 'gram.txt') \
            or (not os.path.isfile(path + '/' + 'chars.txt')):
        toolbox.get_vocab_tag(path, [train_file, dev_file], ngram=ngram)
    # 读取文本信息
    chars, tags, ngram = toolbox.read_vocab_tag(path, ngram)

    emb = None
    emb_dim = args.embeddings_dimension
    if args.embeddings is not None:
        # 读取预训练字向量
        print 'Reading embeddings...'
        short_emb = args.embeddings[args.embeddings.index('/') + 1: args.embeddings.index('.')]
        if not os.path.isfile(path + '/' + short_emb + '_sub.txt'):
            toolbox.get_sample_embedding(path, args.embeddings, map(lambda x: x[0], chars))
        emb_dim, emb = toolbox.read_sample_embedding(path, short_emb)
        assert args.embeddings_dimension == emb_dim
    else:
        print 'Using random embeddings...'

    char2idx, idx2char, char2freq, tag2idx, idx2tag = toolbox.get_dic(chars, tags, args.char_freq_loss)

    # train_x: shape=(2,句子数量)，2 表示字符本身+偏旁部首
    train_x, train_y, train_max_slen_c, train_max_slen_w, train_max_wlen = \
        toolbox.get_input_vec(path, train_file, char2idx, tag2idx, char2freq, tag_scheme=args.tag_scheme)
    dev_x, dev_y, dev_max_slen_c, dev_max_slen_w, dev_max_wlen = \
        toolbox.get_input_vec(path, dev_file, char2idx, tag2idx, char2freq, tag_scheme=args.tag_scheme)

    # 读取 ngram 向量
    nums_grams = None
    ng_embeddings = None
    if ngram > 1:
        gram2idx = toolbox.get_ngram_dic(ngram)
        train_gram = toolbox.get_gram_vec(path, train_file, gram2idx)
        dev_gram = toolbox.get_gram_vec(path, dev_file, gram2idx)
        # 这一句后 train_x： shape=(4,句子数量)，因为加了 2gram 和 3gram
        train_x += train_gram
        dev_x += dev_gram
        nums_grams = []
        for dic in gram2idx:
            nums_grams.append(len(dic.keys()))

        if args.ngram_embeddings is not None:
            print 'Reading N-gram Embeddings...'
            short_ng_emb = args.ngram_embeddings[args.ngram_embeddings.index('/') + 1:]
            if not os.path.isfile(path + '/' + short_ng_emb + '_' + str(args.ngram) + 'gram_sub.txt'):
                toolbox.get_ngram_embedding(path, args.ngram_embeddings, ngram)
            ng_embeddings = toolbox.read_ngram_embedding(path, short_ng_emb, args.ngram)

    tag_map = {'seg': 0, 'BI': 1, 'BIE': 2, 'BIES': 3}

    max_step_c = max(train_max_slen_c, dev_max_slen_c)
    max_step_w = max(train_max_slen_w, dev_max_slen_w)
    max_w_len = max(train_max_wlen, dev_max_wlen)

    print 'Longest sentence by character is %d. ' % max_step_c
    print 'Longest sentence by word is %d. ' % max_step_w
    print 'Longest word is %d. ' % max_w_len

    # b_train_x: shape=(4, bucket 数量，)
    b_train_x, b_train_y = toolbox.buckets(train_x, train_y, size=args.bucket_size)
    b_dev_x, b_dev_y = toolbox.buckets(dev_x, dev_y, size=args.bucket_size)

    # 在句子在末尾填充 0 从而使得每个 bucket 内的句子长度保持一致
    # b_train_x: shape=(4, bucket 数量，每个 bucket 内的句子数量，句子长度)
    # b_train_y: shape=(1, bucket 数量，每个 bucket 内的句子数量，句子长度)，1表示每个字对应1个标签
    b_train_x, b_train_y, b_buckets, b_counts = toolbox.pad_bucket(b_train_x, b_train_y)
    # b_buckets：每一个 bucket 中句子的长度（已经对齐过，bucket 中所有句子的长度一致）
    # b_counts：每一个 bucket 中句子的个数
    b_dev_x, b_dev_y, b_buckets, _ = toolbox.pad_bucket(b_dev_x, b_dev_y, bucket_len_c=b_buckets)

    print 'Training set: %d instances; Dev set: %d instances.' % (len(train_x[0]), len(dev_x[0]))

    nums_tags = toolbox.get_nums_tags(tag2idx, args.tag_scheme)
    # 用来对session进行参数配置，allow_soft_placement 表示如果你指定的设备不存在，允许TF自动分配设备
    config = tf.ConfigProto(allow_soft_placement=True)

    print 'Initialization....'
    t = time()
    # Returns an initializer performing "Xavier" initialization for weights.
    # This initializer is designed to keep the scale of the gradients roughly the same in all layers
    initializer = tf.contrib.layers.xavier_initializer()
    # Returns a context manager that makes this Graph the default graph
    main_graph = tf.Graph()
    with main_graph.as_default():
        with tf.variable_scope("tagger", reuse=None, initializer=initializer) as scope:
            # 初始化 model
            model = Model(nums_chars=len(chars) + 2,
                          nums_tags=nums_tags, buckets_char=b_buckets, counts=b_counts,
                          tag_scheme=args.tag_scheme,
                          crf=args.crf,
                          ngram=nums_grams,
                          batch_size=args.train_batch, metric=args.op_metric,
                          co_train=args.co_train,
                          lambda0=args.lambda0,
                          lambda1=args.lambda1,
                          highway_layers=args.highway_layers,
                          char_freq_loss=args.char_freq_loss)
            # 构建 graph，即字符输入=>字向量=>BiLSTM=>全连接层的整个模型图
            model.main_graph(trained_model=path + '/' + model_file + '_model', scope=scope, emb_dim=emb_dim,
                             gru=args.gru, rnn_dim=args.rnn_cell_dimension, rnn_num=args.rnn_layer_number,
                             emb=emb, drop_out=args.dropout_rate, ngram_embedding=ng_embeddings)
        t = time()
        # 根据指定参数策略计算损失函数，计算梯度，应用梯度下降
        model.config(optimizer=args.optimizer, decay=args.decay_rate,
                     lr_v=args.learning_rate, momentum=args.momentum, clipping=args.clipping)
        init = tf.global_variables_initializer()
    # Finalizes this graph, making it read-only.
    main_graph.finalize()

    main_sess = tf.Session(config=config, graph=main_graph)

    if args.crf:
        # 如果是用 Viterbi 解码，则还需要构建解码图
        decode_graph = tf.Graph()
        with decode_graph.as_default():
            model.decode_graph()
        decode_graph.finalize()

        decode_sess = tf.Session(config=config, graph=decode_graph)

        sess = [main_sess, decode_sess]

    else:
        sess = [main_sess]

    # A context manager that specifies the default device to use for newly created ops
    with tf.device(gpu_config):
        main_sess.run(init)
        print 'Done. Time consumed: %d seconds' % int(time() - t)
        t = time()
        # run graph 进行训练
        model.train(t_x=b_train_x, t_y=b_train_y, v_x=b_dev_x, v_y=b_dev_y, idx2tag=idx2tag, idx2char=idx2char,
                    sess=sess, epochs=args.epochs, trained_model=path + '/' + model_file + '_weights',
                    lr=args.learning_rate, decay=args.decay_rate, tag_num=len(tags))
        print 'Done. Time consumed: %f hours' % (int(time() - t) / 60.0 / 60.0)

else:

    assert args.path is not None
    assert args.model is not None
    path = args.path
    assert os.path.isfile(path + '/' + 'chars.txt')

    model_file = args.model
    emb_path = args.embeddings
    ng_emb_path = args.ngram_embeddings

    if args.ensemble:
        if not os.path.isfile(path + '/' + model_file + '_1_model') or not os.path.isfile(
                                        path + '/' + model_file + '_1_weights.index'):
            raise Exception('Not any model file or weights file under the name of ' + model_file + '.')
        fin = open(path + '/' + model_file + '_1_model', 'rb')
    else:
        if not os.path.isfile(path + '/' + model_file + '_model') or not os.path.isfile(
                                        path + '/' + model_file + '_weights.index'):
            raise Exception('No model file or weights file under the name of ' + model_file + '.')
        fin = open(path + '/' + model_file + '_model', 'rb')

    weight_path = path + '/' + model_file

    param_dic = pickle.load(fin)
    fin.close()

    nums_chars = param_dic['nums_chars']
    nums_tags = param_dic['nums_tags']
    tag_scheme = param_dic['tag_scheme']
    crf = param_dic['crf']
    emb_dim = param_dic['emb_dim']
    gru = param_dic['gru']
    rnn_dim = param_dic['rnn_dim']
    rnn_num = param_dic['rnn_num']
    drop_out = param_dic['drop_out']
    buckets_char = param_dic['buckets_char']
    num_ngram = param_dic['ngram']

    ngram = 1
    gram2idx = None
    if num_ngram is not None:
        ngram = len(num_ngram) + 1

    chars, tags, grams = toolbox.read_vocab_tag(path, ngram)
    char2idx, idx2char, char2freq, tag2idx, idx2tag = toolbox.get_dic(chars, tags)

    new_chars, new_grams, new_gram_emb, gram2idx = None, None, None, None

    raw_file = None

    test_x, test_y, raw_x = None, None, None

    rad_dic, pixels = None, None

    unk_char2idx = None

    max_step = None

    s_time = time()
    if args.action == 'test':
        assert args.test is not None

        test_file = args.test
        new_chars = toolbox.get_new_chars(path + '/' + test_file, char2idx)

        valid_chars = None

        if args.embeddings is not None:
            valid_chars = toolbox.get_valid_chars(new_chars, args.embeddings)

        char2idx, idx2char, unk_char2idx = toolbox.update_char_dict(char2idx, new_chars, valid_chars)

        test_x, test_y, test_max_slen_c, test_max_slen_w, test_max_wlen = \
            toolbox.get_input_vec(path, test_file, char2idx, tag2idx, tag_scheme=tag_scheme)

        print 'Test set: %d instances.' % len(test_x[0])

        max_step = test_max_slen_c

        print 'Longest sentence by character is %d. ' % test_max_slen_c
        print 'Longest sentence by word is %d. ' % test_max_slen_w

        print 'Longest word is %d. ' % test_max_wlen

        if ngram > 1:
            gram2idx = toolbox.get_ngram_dic(grams)
            new_grams = toolbox.get_new_grams(path + '/' + test_file, gram2idx)
            if args.ngram_embeddings is not None:
                new_grams = toolbox.get_valid_grams(new_grams, args.ngram_embeddings)
                gram2idx = toolbox.update_gram_dicts(gram2idx, new_grams)

            test_gram = toolbox.get_gram_vec(path, test_file, gram2idx)
            test_x += test_gram

        for k in range(len(test_x)):
            test_x[k] = toolbox.pad_zeros(test_x[k], max_step)
        for k in range(len(test_y)):
            test_y[k] = toolbox.pad_zeros(test_y[k], max_step)

    elif args.action == 'tag':
        assert args.raw is not None

        raw_file = args.raw

        new_chars = toolbox.get_new_chars(raw_file, char2idx, type='raw')

        valid_chars = None

        if args.embeddings is not None:
            valid_chars = toolbox.get_valid_chars(new_chars, args.embeddings)

        char2idx, idx2char, unk_char2idx = toolbox.update_char_dict(char2idx, new_chars, valid_chars)

        if not args.tag_large:

            raw_x, raw_len = toolbox.get_input_vec_raw(None, raw_file, char2idx)
            print 'Numbers of sentences: %d.' % len(raw_x[0])
            max_step = raw_len

        else:
            max_step = toolbox.get_maxstep(raw_file, args.bucket_size)

        print 'Longest sentence is %d. ' % max_step
        if ngram > 1:

            gram2idx = toolbox.get_ngram_dic(grams)

            if args.ngram_embeddings is not None:
                new_grams = toolbox.get_new_grams(raw_file, gram2idx, type='raw')
                new_grams = toolbox.get_valid_grams(new_grams, args.ngram_embeddings)
                gram2idx = toolbox.update_gram_dicts(gram2idx, new_grams)

            if not args.tag_large:

                raw_gram = toolbox.get_gram_vec(None, raw_file, gram2idx, is_raw=True)
                raw_x += raw_gram

        if not args.tag_large:
            for k in range(len(raw_x)):
                raw_x[k] = toolbox.pad_zeros(raw_x[k], max_step)

    config = tf.ConfigProto(allow_soft_placement=True)
    print 'Initialization....'
    t = time()
    main_graph = tf.Graph()

    with main_graph.as_default():
        with tf.variable_scope("tagger") as scope:
            if args.action == 'test' or (args.action == 'tag' and not args.tag_large):
                model = Model(nums_chars=nums_chars, nums_tags=nums_tags, buckets_char=[max_step], counts=[200],
                              batch_size=args.tag_batch, tag_scheme=tag_scheme, crf=crf, ngram=num_ngram)
            else:
                bt_chars = []
                bt_len = args.bucket_size
                while bt_len <= min(300, max_step):
                    bt_chars.append(bt_len)
                    bt_len += args.bucket_size
                bt_chars.append(bt_len)
                if max_step > 300:
                    bt_chars.append(max_step)
                bt_counts = [200] * len(bt_chars)
                model = Model(nums_chars=nums_chars, nums_tags=nums_tags, buckets_char=bt_chars, counts=bt_counts,
                              batch_size=args.tag_batch, tag_scheme=tag_scheme, crf=crf, ngram=num_ngram)
            model.main_graph(trained_model=None, scope=scope, emb_dim=emb_dim, gru=gru, rnn_dim=rnn_dim,
                             rnn_num=rnn_num, drop_out=drop_out)
        model.define_updates(new_chars=new_chars, emb_path=emb_path, char2idx=char2idx, new_grams=new_grams,
                             ng_emb_path=ng_emb_path, gram2idx=gram2idx)

        init = tf.global_variables_initializer()

        print 'Done. Time consumed: %d seconds' % int(time() - t)
    main_graph.finalize()

    idx = None
    if args.ensemble:
        idx = 1
        main_sess = []
        while os.path.isfile(path + '/' + model_file + '_' + str(idx) + '_weights'):
            main_sess.append(tf.Session(config=config, graph=main_graph))
            idx += 1
    else:
        main_sess = tf.Session(config=config, graph=main_graph)

    if crf:
        decode_graph = tf.Graph()

        with decode_graph.as_default():
            model.decode_graph()
        decode_graph.finalize()

        decode_sess = tf.Session(config=config, graph=decode_graph)

        sess = [main_sess, decode_sess]

    else:
        sess = [main_sess]

    with tf.device(gpu_config):
        ens_model = None
        print 'Loading weights....'
        if args.ensemble:
            for i in range(1, idx):
                print 'Ensemble: ' + str(i)
                main_sess[i - 1].run(init)
                model.run_updates(main_sess[i - 1], weight_path + '_' + str(i) + '_weights')
        else:
            main_sess.run(init)
            model.run_updates(main_sess, weight_path + '_weights')

        if args.action == 'test':
            model.test(sess=sess, t_x=test_x, t_y=test_y, idx2tag=idx2tag, idx2char=idx2char, outpath=args.output_path,
                       ensemble=args.ensemble, batch_size=args.test_batch, tag_num=len(tags))

        elif args.action == 'tag':
            if not args.tag_large:
                model.tag(sess=sess, r_x=raw_x, idx2tag=idx2tag, idx2char=idx2char, char2idx=unk_char2idx,
                          outpath=args.output_path, ensemble=args.ensemble, batch_size=args.tag_batch,
                          large_file=args.tag_large)
            else:
                def tag_large_raw_file(l_raw_file, output_path, l_max_step):
                    l_writer = codecs.open(output_path, 'w', encoding='utf-8')
                    count = 0
                    with codecs.open(l_raw_file, 'r', encoding='utf-8') as l_file:
                        lines = []
                        for line in l_file:
                            line = line.strip()
                            lines.append("".join(line.split()))
                            if len(lines) >= args.large_size:
                                count += len(lines)
                                print count
                                raw_x, _ = toolbox.get_input_vec_line(lines, char2idx, rad_dic=rad_dic)

                                if ngram > 1:
                                    raw_gram = toolbox.get_gram_vec_raw(lines, gram2idx)
                                    raw_x += raw_gram

                                for k in range(len(raw_x)):
                                    raw_x[k] = toolbox.pad_zeros(raw_x[k], l_max_step)

                                out = model.tag(sess=sess, r_x=raw_x, idx2tag=idx2tag, idx2char=idx2char,
                                                char2idx=unk_char2idx, outpath=args.output_path, ensemble=args.ensemble,
                                                batch_size=args.tag_batch, large_file=args.tag_large)

                                for l_out in out:
                                    l_writer.write(l_out + '\n')
                                lines = []
                        if len(lines) > 0:
                            count += len(lines)
                            print count
                            raw_x, _ = toolbox.get_input_vec_line(lines, char2idx, rad_dic=rad_dic)

                            if ngram > 1:
                                raw_gram = toolbox.get_gram_vec_raw(lines, gram2idx)
                                raw_x += raw_gram

                            for k in range(len(raw_x)):
                                raw_x[k] = toolbox.pad_zeros(raw_x[k], l_max_step)

                            out = model.tag(sess=sess, r_x=raw_x, idx2tag=idx2tag, idx2char=idx2char,
                                            char2idx=unk_char2idx,
                                            outpath=args.output_path, ensemble=args.ensemble, batch_size=args.tag_batch,
                                            large_file=args.tag_large)

                            for l_out in out:
                                l_writer.write(l_out + '\n')

                    l_writer.close()


                bt_num = (min(300, max_step) - 1) / args.bucket_size + 1

                for i in range(bt_num):
                    print 'Tagging sentences in bucket %d: ' % (i + 1)
                    tag_large_raw_file(raw_file + '_' + str(i), args.output_path + '_' + str(i),
                                       (i + 1) * args.bucket_size)

                if max_step > 300:
                    print 'Tagging sentences in the last bucket: '
                    tag_large_raw_file(raw_file + '_' + str(bt_num), args.output_path + '_' + str(bt_num), max_step)
                    bt_num += 1

                print 'Merging...'
                toolbox.merge_files(args.output_path, raw_file, bt_num)

        print 'Done.'
        print 'Done. Time consumed: %d seconds' % int(time() - s_time)
