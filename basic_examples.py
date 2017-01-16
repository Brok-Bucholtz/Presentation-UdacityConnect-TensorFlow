import tensorflow as tf


def graphs():
    graph_g = tf.Graph()

    # Using Graph graph_g
    with graph_g.as_default():
        graph_a = tf.constant(10)
        graph_b = tf.constant(2)
        graph_y = tf.mul(graph_a, graph_b)

    assert graph_y.graph == graph_g

    with tf.Session(graph=graph_g) as sess:
        graph_out = sess.run(graph_y)
        print('graph_out: {}'.format(graph_out))

    # Using default graph
    a = tf.constant(10)
    b = tf.constant(5)
    y = tf.mul(a, b)

    assert y.graph == tf.get_default_graph()

    with tf.Session() as sess:
        out = sess.run(y)
        print('out: {}'.format(out))


def sessions():
    # Run on CPU 0
    with tf.device('/cpu:0'):
        a = tf.constant(10)
        b = tf.constant(5)
        c = tf.mul(a, b)

    # Run on CPU 0 as well, but this could be any resource including a different computer
    with tf.device('/cpu:0'):
        d = tf.constant(10)
        e = tf.constant(5)
        f = tf.mul(a, b)

    # Run on default device
    g = tf.add(c, f)

    print('y type: {}'.format(type(g)))

    # Execute graph
    with tf.Session() as sess:
        out = sess.run(g)
        print('out type: {}'.format(type(out)))
        print('out: {}'.format(out))


def math_operations():
    a = tf.constant(10)
    b = tf.constant(5)

    # The calling tf math operations are only required over *, -, etc. when no inputs are Tensors
    c = tf.mul(a, b)
    d = a * b
    e = a * 5
    f = 10 * b
    g = tf.mul(10, 5)
    h = 10 * 5

    print('c type: {}'.format(type(c)))
    print('d type: {}'.format(type(d)))
    print('e type: {}'.format(type(e)))
    print('f type: {}'.format(type(f)))
    print('g type: {}'.format(type(g)))
    print('h type: {}'.format(type(h)))

    with tf.Session() as sess:
        c_out = sess.run(c)
        d_out = sess.run(d)
        e_out = sess.run(e)
        f_out = sess.run(f)
        g_out = sess.run(g)

        # Make sure they are the same results
        assert c_out == d_out and c_out == e_out and c_out == f_out and c_out == g_out
        print('All TensorFlow Results: {}'.format(c_out))


def save(save_path):
    v1 = tf.Variable(tf.random_normal((1, 3)), name="v1")
    v2 = tf.Variable(tf.random_normal((1, 3)), name="v2")

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('v1: {}'.format(sess.run(v1)))
        print('v2: {}'.format(sess.run(v2)))
        save_path = saver.save(sess, save_path)


def load(save_path):
    tf.reset_default_graph()
    v1 = tf.Variable(tf.random_normal((1, 3)), name="v1")
    v2 = tf.Variable(tf.random_normal((1, 3)), name="v2")

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, save_path)
        print('v1: {}'.format(sess.run(v1)))
        print('v2: {}'.format(sess.run(v2)))


def run():
    save_path = './example_2'

    print('Graph Example')
    graphs()

    print('\nSession Example')
    sessions()

    print('\nMath Operations Example')
    math_operations()

    print('\nSave Example')
    save(save_path)

    print('\nLoad Example')
    load(save_path)


if __name__ == '__main__':
    run()
