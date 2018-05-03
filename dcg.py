import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import os

train_x_32 = []
for f in range(6):
    f += 1
    filepath = './'+str(f)+'/32x18/'
    for filename in os.listdir(filepath):
        image = mpimg.imread(filepath+filename)
        train_x_32.append(np.zeros((32, 32, 3)))
        train_x_32[-1][7:7+18] = image
train_x_32 = np.asarray(train_x_32)

train_x_64 = []
for f in range(6):
    f += 1
    filepath = './'+str(f)+'/64x36/'
    for filename in os.listdir(filepath):
        image = mpimg.imread(filepath+filename)
        train_x_64.append(np.zeros((64, 64, 3)))
        train_x_64[-1][14:14+36] = image
train_x_64 = np.asarray(train_x_64)

train_x_128 = []
for f in range(6):
    f += 1
    filepath = './'+str(f)+'/128x72/'
    for filename in os.listdir(filepath):
        image = mpimg.imread(filepath+filename)
        train_x_128.append(np.zeros((128, 128, 3)))
        train_x_128[-1][28:28+72] = image
        print(filename)
train_x_128 = np.asarray(train_x_128)



last_epoch = 0
total_epochs = 400
batch_size = 100
learning_rate = 0.0005
random_size = 256

init = tf.random_normal_initializer(mean=0.0, stddev = 0.01)


def generator1( z , reuse = False ):
    l = [1024, 256, 64, 32, 3]
    #l = [512, 256, 128, 64, 3]
    with tf.variable_scope(name_or_scope = "Gen1") as scope:
        gw1 = tf.get_variable(name = "w1", shape = [5, 5, l[1], l[0]], initializer = init)
        gw2 = tf.get_variable(name = "w2", shape = [5, 5, l[2], l[1]], initializer = init)
        gw3 = tf.get_variable(name = "w3", shape = [5, 5, l[3], l[2]], initializer = init)
        gw4 = tf.get_variable(name = "w4", shape = [5, 5, l[4], l[3]], initializer = init)


    z = tf.layers.dense(z, 2 * 2 * l[0])
    size = tf.shape(z)[0]
    z = tf.reshape(z, [size, 2, 2, l[0]])
    output = z

    output = tf.nn.conv2d_transpose(output, gw1, output_shape=[size, 4, 4, l[1]], strides=[1, 2, 2, 1], padding='SAME')
    output = tf.reshape(output, [-1, 4, 4, l[1]])
    output = tf.layers.batch_normalization(output, training=True)
    output = tf.nn.relu(output)


    output = tf.nn.conv2d_transpose(output, gw2, output_shape=[size, 8, 8, l[2]], strides=[1, 2, 2, 1], padding='SAME')
    output = tf.reshape(output, [-1, 8, 8, l[2]])
    output = tf.layers.batch_normalization(output, training=True)
    output = tf.nn.relu(output)

    output = tf.nn.conv2d_transpose(output, gw3, output_shape=[size, 16, 16, l[3]], strides=[1, 2, 2, 1], padding='SAME')
    output = tf.reshape(output, [-1, 16, 16, l[3]])
    output = tf.layers.batch_normalization(output, training=True)
    output = tf.nn.relu(output)

    output = tf.nn.conv2d_transpose(output, gw4, output_shape=[size, 32, 32, l[4]], strides=[1, 2, 2, 1], padding='SAME')
    output = tf.reshape(output, [-1, 32, 32, l[4]])
    output = tf.tanh(output)
    return output

def generator2( z , reuse = False ):
    dl = [3, 32, 64]
    l = [64, 32, 16, 3]
    with tf.variable_scope(name_or_scope = "Gen2") as scope:
        dw1 = tf.get_variable(name = "dw1", shape = [5, 5, dl[0], dl[1]], initializer = init)
        dw2 = tf.get_variable(name = "dw2", shape = [5, 5, dl[1], dl[2]], initializer = init)

        gw1 = tf.get_variable(name = "w1", shape = [5, 5, l[1], l[0]], initializer = init)
        gw2 = tf.get_variable(name = "w2", shape = [5, 5, l[2], l[1]], initializer = init)
        gw3 = tf.get_variable(name = "w3", shape = [5, 5, l[3], l[2]], initializer = init)

    output = tf.nn.conv2d(z, dw1, strides=[1, 2, 2, 1], padding='SAME')
    output = tf.reshape(output, [-1, 16, 16, dl[1]])
    output = tf.layers.batch_normalization(output, training=True)
    output = tf.nn.relu(output)

    output = tf.nn.conv2d(output, dw2, strides=[1, 2, 2, 1], padding='SAME')
    output = tf.reshape(output, [-1, 8, 8, dl[2]])
    output = tf.layers.batch_normalization(output, training=True)
    output = tf.nn.relu(output)


    size = tf.shape(z)[0]

    output = tf.nn.conv2d_transpose(output, gw1, output_shape=[size, 16, 16, l[1]], strides=[1, 2, 2, 1], padding='SAME')
    output = tf.reshape(output, [-1, 16, 16, l[1]])
    output = tf.layers.batch_normalization(output, training=True)
    output = tf.nn.relu(output)

    output = tf.nn.conv2d_transpose(output, gw2, output_shape=[size, 32, 32, l[2]], strides=[1, 2, 2, 1], padding='SAME')
    output = tf.reshape(output, [-1, 32, 32, l[2]])
    output = tf.layers.batch_normalization(output, training=True)
    output = tf.nn.relu(output)

    output = tf.nn.conv2d_transpose(output, gw3, output_shape=[size, 64, 64, l[3]], strides=[1, 2, 2, 1], padding='SAME')
    output = tf.reshape(output, [-1, 64, 64, l[3]])
    output = tf.tanh(output)
    return output

def generator3( z , reuse = False ):
    dl = [3, 32, 64, 128]
    l = [128, 64, 32, 16, 3]
    with tf.variable_scope(name_or_scope = "Gen3") as scope:
        dw1 = tf.get_variable(name = "dw1", shape = [5, 5, dl[0], dl[1]], initializer = init)
        dw2 = tf.get_variable(name = "dw2", shape = [5, 5, dl[1], dl[2]], initializer = init)
        dw3 = tf.get_variable(name = "dw3", shape = [5, 5, dl[2], dl[3]], initializer = init)

        gw1 = tf.get_variable(name = "w1", shape = [5, 5, l[1], l[0]], initializer = init)
        gw2 = tf.get_variable(name = "w2", shape = [5, 5, l[2], l[1]], initializer = init)
        gw3 = tf.get_variable(name = "w3", shape = [5, 5, l[3], l[2]], initializer = init)
        gw4 = tf.get_variable(name = "w4", shape = [5, 5, l[4], l[3]], initializer = init)


    output = tf.nn.conv2d(z, dw1, strides=[1, 2, 2, 1], padding='SAME')
    output = tf.reshape(output, [-1, 32, 32, dl[1]])
    output = tf.layers.batch_normalization(output, training=True)
    output = tf.nn.relu(output)

    output = tf.nn.conv2d(output, dw2, strides=[1, 2, 2, 1], padding='SAME')
    output = tf.reshape(output, [-1, 16, 16, dl[2]])
    output = tf.layers.batch_normalization(output, training=True)
    output = tf.nn.relu(output)

    output = tf.nn.conv2d(output, dw3, strides=[1, 2, 2, 1], padding='SAME')
    output = tf.reshape(output, [-1, 8, 8, dl[3]])
    output = tf.layers.batch_normalization(output, training=True)
    output = tf.nn.relu(output)



    size = tf.shape(z)[0]

    output = tf.nn.conv2d_transpose(output, gw1, output_shape=[size, 16, 16, l[1]], strides=[1, 2, 2, 1], padding='SAME')
    output = tf.reshape(output, [-1, 16, 16, l[1]])
    output = tf.layers.batch_normalization(output, training=True)
    output = tf.nn.relu(output)


    output = tf.nn.conv2d_transpose(output, gw2, output_shape=[size, 32, 32, l[2]], strides=[1, 2, 2, 1], padding='SAME')
    output = tf.reshape(output, [-1, 32, 32, l[2]])
    output = tf.layers.batch_normalization(output, training=True)
    output = tf.nn.relu(output)

    output = tf.nn.conv2d_transpose(output, gw3, output_shape=[size, 64, 64, l[3]], strides=[1, 2, 2, 1], padding='SAME')
    output = tf.reshape(output, [-1, 64, 64, l[3]])
    output = tf.layers.batch_normalization(output, training=True)
    output = tf.nn.relu(output)

    output = tf.nn.conv2d_transpose(output, gw4, output_shape=[size, 128, 128, l[4]], strides=[1, 2, 2, 1], padding='SAME')
    output = tf.reshape(output, [-1, 128, 128, l[4]])
    output = tf.tanh(output)
    return output




def discriminator1( x , reuse = False):
    l = [3, 16, 32, 64]
    x = tf.convert_to_tensor(x)
    with tf.variable_scope(name_or_scope="Dis1", reuse = reuse) as scope:
        dw1 = tf.get_variable(name = "w1", shape = [5, 5, l[0], l[1]], initializer = init)
        dw2 = tf.get_variable(name = "w2", shape = [5, 5, l[1], l[2]], initializer = init)
        dw3 = tf.get_variable(name = "w3", shape = [5, 5, l[2], l[3]], initializer = init)



    alpha = 0.1
    output = tf.nn.conv2d(x, dw1, strides=[1, 2, 2, 1], padding='SAME')
    output = tf.reshape(output, [-1, 16, 16, l[1]])
    output = tf.layers.batch_normalization(output, training=True)
    output = tf.nn.relu(output) - alpha*tf.nn.relu(-output)

    output = tf.nn.conv2d(output, dw2, strides=[1, 2, 2, 1], padding='SAME')
    output = tf.reshape(output, [-1, 8, 8, l[2]])
    output = tf.layers.batch_normalization(output, training=True)
    output = tf.nn.relu(output) - alpha*tf.nn.relu(-output)

    output = tf.nn.conv2d(output, dw3, strides=[1, 2, 2, 1], padding='SAME')
    output = tf.reshape(output, [-1, 4, 4, l[3]])
    output = tf.layers.batch_normalization(output, training=True)
    output = tf.nn.relu(output) - alpha*tf.nn.relu(-output)

    reshape = tf.reshape(output, [-1, 4*4*l[3]])
    output = tf.layers.dense(reshape, 1)

    return output

def discriminator2( x , reuse = False):
    l = [3, 16, 32, 64, 128]
    x = tf.convert_to_tensor(x)
    with tf.variable_scope(name_or_scope="Dis2", reuse = reuse) as scope:
        dw1 = tf.get_variable(name = "w1", shape = [5, 5, l[0], l[1]], initializer = init)
        dw2 = tf.get_variable(name = "w2", shape = [5, 5, l[1], l[2]], initializer = init)
        dw3 = tf.get_variable(name = "w3", shape = [5, 5, l[2], l[3]], initializer = init)
        dw4 = tf.get_variable(name = "w4", shape = [5, 5, l[3], l[2]], initializer = init)



    alpha = 0.1
    output = tf.nn.conv2d(x, dw1, strides=[1, 2, 2, 1], padding='SAME')
    output = tf.reshape(output, [-1, 32, 32, l[1]])
    output = tf.layers.batch_normalization(output, training=True)
    output = tf.nn.relu(output) - alpha*tf.nn.relu(-output)

    output = tf.nn.conv2d(output, dw2, strides=[1, 2, 2, 1], padding='SAME')
    output = tf.reshape(output, [-1, 16, 16, l[2]])
    output = tf.layers.batch_normalization(output, training=True)
    output = tf.nn.relu(output) - alpha*tf.nn.relu(-output)

    output = tf.nn.conv2d(output, dw3, strides=[1, 2, 2, 1], padding='SAME')
    output = tf.reshape(output, [-1, 8, 8, l[3]])
    output = tf.layers.batch_normalization(output, training=True)
    output = tf.nn.relu(output) - alpha*tf.nn.relu(-output)

    output = tf.nn.conv2d(output, dw4, strides=[1, 2, 2, 1], padding='SAME')
    output = tf.reshape(output, [-1, 4, 4, l[4]])
    output = tf.layers.batch_normalization(output, training=True)
    output = tf.nn.relu(output) - alpha*tf.nn.relu(-output)

    reshape = tf.reshape(output, [-1, 4*4*l[4]])
    output = tf.layers.dense(reshape, 1)

    return output

def discriminator3( x , reuse = False):
    l = [3, 16, 32, 64, 128, 256]
    x = tf.convert_to_tensor(x)
    with tf.variable_scope(name_or_scope="Dis3", reuse = reuse) as scope:
        dw1 = tf.get_variable(name = "w1", shape = [5, 5, l[0], l[1]], initializer = init)
        dw2 = tf.get_variable(name = "w2", shape = [5, 5, l[1], l[2]], initializer = init)
        dw3 = tf.get_variable(name = "w3", shape = [5, 5, l[2], l[3]], initializer = init)
        dw4 = tf.get_variable(name = "w4", shape = [5, 5, l[3], l[4]], initializer = init)
        dw5 = tf.get_variable(name = "w5", shape = [5, 5, l[4], l[5]], initializer = init)




    alpha = 0.1
    output = tf.nn.conv2d(x, dw1, strides=[1, 2, 2, 1], padding='SAME')
    output = tf.reshape(output, [-1, 64, 64, l[1]])
    output = tf.layers.batch_normalization(output, training=True)
    output = tf.nn.relu(output) - alpha*tf.nn.relu(-output)

    output = tf.nn.conv2d(output, dw2, strides=[1, 2, 2, 1], padding='SAME')
    output = tf.reshape(output, [-1, 32, 32, l[2]])
    output = tf.layers.batch_normalization(output, training=True)
    output = tf.nn.relu(output) - alpha*tf.nn.relu(-output)

    output = tf.nn.conv2d(output, dw3, strides=[1, 2, 2, 1], padding='SAME')
    output = tf.reshape(output, [-1, 16, 16, l[3]])
    output = tf.layers.batch_normalization(output, training=True)
    output = tf.nn.relu(output) - alpha*tf.nn.relu(-output)

    output = tf.nn.conv2d(output, dw4, strides=[1, 2, 2, 1], padding='SAME')
    output = tf.reshape(output, [-1, 8, 8, l[4]])
    output = tf.layers.batch_normalization(output, training=True)
    output = tf.nn.relu(output) - alpha*tf.nn.relu(-output)

    output = tf.nn.conv2d(output, dw5, strides=[1, 2, 2, 1], padding='SAME')
    output = tf.reshape(output, [-1, 4, 4, l[5]])
    output = tf.layers.batch_normalization(output, training=True)
    output = tf.nn.relu(output) - alpha*tf.nn.relu(-output)


    reshape = tf.reshape(output, [-1, 4*4*l[5]])
    output = tf.layers.dense(reshape, 1)

    return output




def random_noise(batch_size):
    return np.random.normal(size=[batch_size , random_size])




g = tf.Graph()

with g.as_default():
    X_32 = tf.placeholder(tf.float32, [None, 32, 32, 3])
    Z = tf.placeholder(tf.float32, [None, random_size])

    fake_x_32 = generator1(Z)
    result_of_fake_32 = discriminator1(fake_x_32, False)
    result_of_real_32 = discriminator1(X_32 , True)

    d_loss_real_32 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=result_of_real_32, labels=tf.ones_like(result_of_real_32)))
    d_loss_fake_32 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=result_of_fake_32, labels=tf.zeros_like(result_of_fake_32)))
    d_loss_32 = d_loss_real_32 + d_loss_fake_32

    g_loss_32 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=result_of_fake_32, labels=tf.ones_like(result_of_fake_32)))



    t_vars = tf.trainable_variables()
    g_vars_32 = [var for var in t_vars if "Gen1" in var.name]
    d_vars_32 = [var for var in t_vars if "Dis1" in var.name]
    optimizer = tf.train.AdamOptimizer(learning_rate)
    #gvs = optimizer.compute_gradients(g_loss, var_list = g_vars)
    #capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    #g_train = optimizer.apply_gradients(capped_gvs)
    g_train_32 = optimizer.minimize(g_loss_32, var_list = g_vars_32)
    d_train_32 = optimizer.minimize(d_loss_32, var_list = d_vars_32)






    X_64 = tf.placeholder(tf.float32, [None, 64, 64, 3])

    fake_x_64 = generator2(fake_x_32)
    result_of_fake_64 = discriminator2(fake_x_64, False)
    result_of_real_64 = discriminator2(X_64 , True)

    d_loss_real_64 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=result_of_real_64, labels=tf.ones_like(result_of_real_64)))
    d_loss_fake_64 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=result_of_fake_64, labels=tf.zeros_like(result_of_fake_64)))
    d_loss_64 = d_loss_real_64 + d_loss_fake_64

    g_loss_64 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=result_of_fake_64, labels=tf.ones_like(result_of_fake_64)))



    t_vars = tf.trainable_variables()
    g_vars_64 = [var for var in t_vars if "Gen2" in var.name]
    d_vars_64 = [var for var in t_vars if "Dis2" in var.name]
    optimizer = tf.train.AdamOptimizer(learning_rate)
    #gvs = optimizer.compute_gradients(g_loss, var_list = g_vars)
    #capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    #g_train = optimizer.apply_gradients(capped_gvs)
    g_train_64 = optimizer.minimize(g_loss_64, var_list = g_vars_64)
    d_train_64 = optimizer.minimize(d_loss_64, var_list = d_vars_64)





    X_128 = tf.placeholder(tf.float32, [None, 128, 128, 3])

    fake_x_128 = generator3(fake_x_64)
    result_of_fake_128 = discriminator3(fake_x_128, False)
    result_of_real_128 = discriminator3(X_128, True)

    d_loss_real_128 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=result_of_real_128, labels=tf.ones_like(result_of_real_128)))
    d_loss_fake_128 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=result_of_fake_128, labels=tf.zeros_like(result_of_fake_128)))
    d_loss_128 = d_loss_real_128 + d_loss_fake_128

    g_loss_128 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=result_of_fake_128, labels=tf.ones_like(result_of_fake_128)))



    t_vars = tf.trainable_variables()
    g_vars_128 = [var for var in t_vars if "Gen3" in var.name]
    d_vars_128 = [var for var in t_vars if "Dis3" in var.name]
    optimizer = tf.train.AdamOptimizer(learning_rate)
    #gvs = optimizer.compute_gradients(g_loss, var_list = g_vars)
    #capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    #g_train = optimizer.apply_gradients(capped_gvs)
    g_train_128 = optimizer.minimize(g_loss_128, var_list = g_vars_128)
    d_train_128 = optimizer.minimize(d_loss_128, var_list = d_vars_128)





with tf.Session(graph = g) as sess:
    saver = tf.train.Saver()
    #saver.restore(sess,"./dcgan"+".ckpt")
    sess.run(tf.global_variables_initializer())
    total_batches = int(train_x_32.shape[0] / batch_size)
    """
    for epoch in range(total_epochs):
        epoch += last_epoch
        noise = []
        for batch in range(total_batches):
            batch_x_32 = np.subtract(np.multiply(train_x_32[batch * batch_size : (batch+1) * batch_size],2),1)

            noise.append(random_noise(batch_size))

            sess.run(d_train_32, feed_dict = {X_32: batch_x_32 , Z : noise[-1]})
            sess.run(g_train_32, feed_dict = {Z : noise[-1]})
            gl_32, dl_32 = sess.run([g_loss_32, d_loss_32], feed_dict = {X_32 : batch_x_32 , Z : noise[-1]})

        for batch in range(total_batches):
            batch_x_64 = np.subtract(np.multiply(train_x_64[batch * batch_size : (batch+1) * batch_size],2),1)

            sess.run(d_train_64, feed_dict = {X_64: batch_x_64 , Z : noise[batch]})
            sess.run(g_train_64, feed_dict = {Z : noise[batch]})
            gl_64, dl_64 = sess.run([g_loss_64, d_loss_64], feed_dict = {X_64 : batch_x_64 , Z : noise[batch]})

        for batch in range(total_batches):
            batch_x_128 = np.subtract(np.multiply(train_x_128[batch * batch_size : (batch+1) * batch_size],2),1)

            sess.run(d_train_128, feed_dict = {X_128: batch_x_128 , Z : noise[batch]})
            sess.run(g_train_128, feed_dict = {Z : noise[batch]})
            gl_128, dl_128 = sess.run([g_loss_128, d_loss_128], feed_dict = {X_128 : batch_x_128 , Z : noise[batch]})


        samples = 20
        if epoch % 5 == 0:
            sample_noise = random_noise(samples)
            generated = sess.run(fake_x_32 , feed_dict = { Z : sample_noise})
            for i in range(samples):
                print(np.max(generated[i]))
                print(np.min(generated[i]))
            for i in range(samples):
                mpimg.imsave('epoch'+str(epoch)+'_32_'+str(i)+'.png', (generated[i]+1)/2.0, format='png')

            generated = sess.run(fake_x_64 , feed_dict = { Z : sample_noise})
            for i in range(samples):
                mpimg.imsave('epoch'+str(epoch)+'_64_'+str(i)+'.png', (generated[i]+1)/2.0, format='png')

            generated = sess.run(fake_x_128 , feed_dict = { Z : sample_noise})
            for i in range(samples):
                mpimg.imsave('epoch'+str(epoch)+'_128_'+str(i)+'.png', (generated[i]+1)/2.0, format='png')



        if epoch == 0:
            for i in range(10):
                i += 1000
                mpimg.imsave('orig'+str(i)+'.png',train_x_128[i], format='png')

        if epoch % 1 == 0:
            print("======= Epoch : ", epoch , " =======")
            print("generator: " , gl_32, gl_64, gl_128 )
            print("discriminator: " , dl_32, dl_64, dl_128 )
        """
    total_step = 150
    for epoch in range(total_epochs):
        for step in range(total_step):
            for batch in range(total_batches):
                batch_x_32 = np.subtract(np.multiply(train_x_32[batch * batch_size : (batch+1) * batch_size],2),1)

                noise = random_noise(batch_size)

                sess.run(d_train_32, feed_dict = {X_32: batch_x_32 , Z : noise})
                sess.run(g_train_32, feed_dict = {Z : noise})
                gl_32, dl_32 = sess.run([g_loss_32, d_loss_32], feed_dict = {X_32 : batch_x_32 , Z : noise})

            samples = 20
            if step % 10 == 0:
                sample_noise = random_noise(samples)
                generated = sess.run(fake_x_32 , feed_dict = { Z : sample_noise})
                for i in range(samples):
                    mpimg.imsave('step'+str(epoch)+'_'+str(step)+'_32_'+str(i)+'.png', (generated[i]+1)/2.0, format='png')


            if step % 5 == 0:
                print("======= 32 step : ", step , " =======")
                print("generator: " , gl_32)
                print("discriminator: " , dl_32)


        for step in range(total_step):
            for batch in range(total_batches):
                batch_x_64 = np.subtract(np.multiply(train_x_64[batch * batch_size : (batch+1) * batch_size],2),1)

                noise = random_noise(batch_size)

                sess.run(d_train_64, feed_dict = {X_64: batch_x_64 , Z : noise})
                sess.run(g_train_64, feed_dict = {Z : noise})
                gl_64, dl_64 = sess.run([g_loss_64, d_loss_64], feed_dict = {X_64 : batch_x_64 , Z : noise})

            samples = 20
            if step % 10 == 0:
                sample_noise = random_noise(samples)
                generated = sess.run(fake_x_64 , feed_dict = { Z : sample_noise})
                for i in range(samples):
                    mpimg.imsave('step'+str(epoch)+'_'+str(step)+'_64_'+str(i)+'.png', (generated[i]+1)/2.0, format='png')


            if step % 5 == 0:
                print("======= 64 step : ", step , " =======")
                print("generator: " , gl_64,)
                print("discriminator: " , dl_64)


        for step in range(total_step):
            for batch in range(total_batches):
                batch_x_128 = np.subtract(np.multiply(train_x_128[batch * batch_size : (batch+1) * batch_size],2),1)

                noise = random_noise(batch_size)

                sess.run(d_train_128, feed_dict = {X_128: batch_x_128 , Z : noise})
                sess.run(g_train_128, feed_dict = {Z : noise})
                gl_128, dl_128 = sess.run([g_loss_128, d_loss_128], feed_dict = {X_128 : batch_x_128 , Z : noise})

            samples = 20
            if step % 10 == 0:
                sample_noise = random_noise(samples)
                generated = sess.run(fake_x_128 , feed_dict = { Z : sample_noise})
                for i in range(samples):
                    mpimg.imsave('step'+str(epoch)+'_'+str(step)+'_128_'+str(i)+'.png', (generated[i]+1)/2.0, format='png')


            if step % 5 == 0:
                print("======= 128 step : ", step , " =======")
                print("generator: " , gl_128 )
                print("discriminator: " , dl_128 )


        """
        samples = 20
        if epoch % 5 == 0:
            sample_noise = random_noise(samples)
            generated = sess.run(fake_x_32 , feed_dict = { Z : sample_noise})
            for i in range(samples):
                print(np.max(generated[i]))
                print(np.min(generated[i]))
            for i in range(samples):
                mpimg.imsave('epoch'+str(epoch)+'_32_'+str(i)+'.png', (generated[i]+1)/2.0, format='png')

            generated = sess.run(fake_x_64 , feed_dict = { Z : sample_noise})
            for i in range(samples):
                mpimg.imsave('epoch'+str(epoch)+'_64_'+str(i)+'.png', (generated[i]+1)/2.0, format='png')

            generated = sess.run(fake_x_128 , feed_dict = { Z : sample_noise})
            for i in range(samples):
                mpimg.imsave('epoch'+str(epoch)+'_128_'+str(i)+'.png', (generated[i]+1)/2.0, format='png')
        """



        if epoch % 1 == 0:
            print("======= Epoch : ", epoch , " =======")
            print("generator: " , gl_32, gl_64, gl_128)
            print("discriminator: " , dl_32, dl_64, dl_128)



        save_path = saver.save(sess, "./dcgan"+".ckpt")
