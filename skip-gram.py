from collections import Counter
import numpy as np
import tensorflow as tf
print(tf.__version__)


def preprocess(text, freq=5):
    """
    统计汉字出现的频率，将少于某个值的汉字过滤掉
    :param text:
    :param freq:
    :return:
    """
    text = text.lower()
    text = text.replace(',|，', '')
    text = text.replace('.|。', '')
    text = text.replace('\r', '')
    text = text.replace('\n', '')

    counter = Counter(text)
    filtered_words = [w for w in text if counter[w] > freq]

    return filtered_words


def get_targets(words, idx, window_size=5):
    """
    获取input word(idx下标对应的word)对应的上下文words
    :param words: 单词列表
    :param idx: input word的下标
    :param window_size: 窗口大小
    :return:
    """
    target_window = np.random.randint(1, window_size + 1)
    start_point = idx - target_window if (idx - target_window) > 0 else 0
    end_point = idx + target_window
    targets = set(words[start_point:idx] + words[idx+1:end_point+1])  # [start_point, idx) + [idx+1, endpoint+1)

    return targets


def get_batches(words, batch_size, window_size=5):
    """
    batch的生成器
    :param words:
    :param batch_size:
    :param window_size:
    :return:
    """
    n_batchs = len(words) // batch_size
    words = words[:n_batchs * batch_size]

    for idx in range(0, len(words), batch_size):
        x, y = [], []
        for j in range(idx, idx + batch_size):
            batch_x = [words[j]]
            batch_y = get_targets(words, j, window_size)

            x.extend(batch_x * len(batch_y))
            y.extend(batch_y)
        yield x, y


def main(file):
    with open(file) as f:
        text = f.read()

    # 构造字典
    words = preprocess(text, 2)
    print('total word:{}'.format(len(words)))
    vocab = set(words)
    word2id, id2word = {}, {}
    for i, word in enumerate(vocab):
        word2id[word] = i
        id2word[i] = word

    vocab_size = len(vocab)
    print('vocab size:{}'.format(vocab_size))

    # 对原文本数字化
    int_words = [word2id[word] for word in words]

    # Subsampling for frequent words   P(wi) = 1 - sqrt(t/f(wi))
    t = 1e-5
    threadhold = 0.9  # 剔除概率阀值
    int_word_counts = Counter(int_words)
    words_count = len(int_words)
    # 字出现的概率
    word_freq = {w: c/words_count for w, c in int_word_counts.items()}
    # 根据公式计算被剔除的概率
    prob_drop = {w: 1- np.sqrt(t / word_freq[w]) for w in int_word_counts}
    train_words = [w for w in int_words if prob_drop[w] < threadhold]
    print("train words len:{}".format(len(train_words)))

    # 输入
    train_graph = tf.Graph()
    with train_graph.as_default():
        inputs = tf.placeholder(tf.int32, shape=[None], name='inputs')
        labels = tf.placeholder(tf.int32, shape=[None, None], name='labels')

    # embedding层
    embed_size = 200
    with train_graph.as_default():
        # 嵌入式权重矩阵
        embedding = tf.Variable(tf.random_uniform([vocab_size, embed_size], -1, 1))
        embed = tf.nn.embedding_lookup(embedding, inputs)

    # 负采样
    n_sampled = 100
    with train_graph.as_default():
        softmax_w = tf.Variable(tf.truncated_normal([vocab_size, embed_size], mean=0, stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(vocab_size))

        # 计算negative sampling下的损失
        sampled_loss = tf.nn.sampled_softmax_loss(softmax_w, softmax_b, labels, embed, n_sampled, vocab_size)
        cost = tf.reduce_mean(sampled_loss)
        optimizer = tf.train.AdamOptimizer().minimize(cost)

    with train_graph.as_default():
        # 选出一些单词
        valid_samples = [
            word2id['国'],
            word2id['新'],
            word2id['a'],
            word2id['春'],
            word2id['早'],
            word2id['北']
        ]

        valid_size = len(valid_samples)
        valid_dataset = tf.constant(valid_samples, dtype=tf.int32)
        norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), axis=1, keep_dims=True))
        normalized_embedding = embedding / norm
        # 查找验证单词的词向量
        valid_embedd = tf.nn.embedding_lookup(normalized_embedding, valid_dataset)
        # 计算验证单词和字典所有单词的余弦相似度
        similarity = tf.matmul(valid_embedd, tf.transpose(normalized_embedding))

    # 开始训练
    epochs = 10
    batch_size = 1000
    window_size = 10

    with train_graph.as_default():
        saver = tf.train.Saver()

    with tf.Session(graph=train_graph) as sess:
        iteration = 1
        loss = 0

        sess.run(tf.initialize_all_variables())
        
        for e in range(1, epochs+1):
            batches = get_batches(train_words, batch_size, window_size)
            for x, y in batches:
                new_y = np.array(y)[:, None]   # y的个数是:N, new_y的shape是(N, 1)
                params = {
                   inputs: x,
                   labels: new_y}
                train_loss, _ = sess.run([cost, optimizer], feed_dict=params)
                loss += train_loss

                if iteration % 100 == 0:
                    print('epoch:{}, iterator:{}, loss:{:.4f}'.format(e, iteration, loss))
                    loss = 0

                if iteration % 1000 == 0:
                    sim = similarity.eval()
                    for i in range(valid_size):
                        valid_word = id2word[valid_samples[i]]
                        top_k = 8
                        nearest = (-sim[i, :]).argsort()[1:top_k+1]
                        log = 'Nearest to [%s]:' % valid_word
                        for k in range(top_k):
                            close_word = id2word[nearest[k]]
                            log = '%s %s,' % (log, close_word)
                        print(log)

                iteration += 1

        save_path = 'checkpoints/test.ckpt'
        saver.save(sess, save_path)


if __name__ == '__main__':
    # 文件是一个中文的文本文件即可
    data_path = 'data/Javasplittedwords'
    main(data_path)
