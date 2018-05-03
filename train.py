import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import os
import re

#MAX_SIZE = 7 #log(256==2^8)-1
MAX_SIZE = 6
DATA_NUM = 12780

def resize_up(image, w, h):
    H = image.shape[0]
    W = image.shape[1]
    print(H,W)
    h_scale = h / int(H)
    w_scale = w / int(W)
    img = np.zeros((h, w, 3))
    for i in range(H):
        for j in range(W):
            for k in range(3):
                img[int(i*h_scale):int((i+1)*h_scale),int(j*w_scale):int((j+1)*w_scale),k] = image[i,j,k]
    return img

def resize_down(image, w, h):
    H = image.shape[0]
    W = image.shape[1]
    img = np.zeros((h, w, 3))
    for i in range(h):
        for j in range(w):
            for k in range(3):
                img[i,j,k] = np.average(image[int(i*H/h):int((i+1)*H/h), int(j*W/w):int((j+1)*W/w), k])
    return img


def upscale2d(x, factor=2):
    if factor == 1: return x
    s = x.shape
    x = tf.reshape(x, [-1, s[1], 1, s[2], 1, s[3]])
    x = tf.tile(x, [1, 1, factor, 1, factor, 1])
    x = tf.reshape(x, [-1, s[1] * factor, s[2] * factor, s[3]])
    return x

def downscale2d(x, factor=2):
    if factor == 1: return x
    ksize = [1, factor, factor, 1]
    return tf.nn.avg_pool(x, ksize=ksize, strides=ksize, padding='VALID') 



train_x = {}
for i in range(MAX_SIZE):
    train_x[pow(2,i+2)] = []
print(train_x)


# read key
key = []
for i in range(DATA_NUM):
    key.append(0)
with open('./0/key.txt','r') as f:
    raw_key = f.readlines()

for i in range(len(raw_key)-1):
    s = int(re.findall('\d+', raw_key[i])[0])
    e = int(re.findall('\d+', raw_key[i])[1])
    for j in range(s,e+1):
        key[j] = 1

# read data
for f in range(1):
    for i in range(MAX_SIZE):
        w = pow(2,i+2)
        h = int(576/pow(2,8-i))
        filepath = './'+str(f)+'/'+str(w)+'x'+str(h)+'/'
        print(filepath)
        for filename in os.listdir(filepath):
            if key[int(os.path.splitext(filename)[0])] == 0:
                continue
            image = mpimg.imread(filepath+filename)
            train_x[w].append(np.zeros((w, w, 3)))
            train_x[w][-1][int((w-h)/2):int((w-h)/2)+h] = image


for i in range(MAX_SIZE):
    s = pow(2,i+2)
    train_x[s][:] = [x * 2 - 1 for x in train_x[s]]
    print(i, len(train_x[pow(2,i+2)]))

last_epoch = 0
batch_size = 16

init = tf.random_normal_initializer(mean=0.0, stddev = 0.01)

def lrelu(x, alpha=0.2):
    return tf.maximum(x*alpha, x)

def pixel_norm(x, epsilon=1e-8):
    with tf.variable_scope('PixelNorm'):
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + epsilon)

def apply_bias(x, name):
    b = tf.get_variable('bias'+name, shape=[x.shape[3]], initializer=tf.initializers.zeros())
    b = tf.cast(b, x.dtype)
    if len(x.shape) == 2:
        return x + b
    else:
        return tf.nn.bias_add(x, b)

#==================================================================================

def generator4x4(z, reuse = False):
    with tf.variable_scope(name_or_scope = "Gen4x4") as scope:
        output = tf.layers.dense(z, 512*4*4)
        output = tf.reshape(output, shape=[-1, 4, 4, 512])
        output = pixel_norm(lrelu(apply_bias(output,"0")))
        output = tf.layers.conv2d(output, 512, 3, strides=1, padding='same', name='conv')
        output = pixel_norm(lrelu(apply_bias(output,"1")))
        toRGB = tf.layers.conv2d(output, 3, 1, strides=1, padding='same', name='toRGB')
        toRGB = apply_bias(toRGB,"2")
        return output, toRGB

def generator8x8(x, reuse = False):
    with tf.variable_scope(name_or_scope = "Gen8x8") as scope:
        output = tf.layers.conv2d_transpose(x, 512, 1, strides=2)
        output = pixel_norm(lrelu(apply_bias(output,"0")))
        output = tf.layers.conv2d(output, 512, 3, strides=1, padding='same', name='conv')
        output = pixel_norm(lrelu(apply_bias(output,"1")))
        toRGB = tf.layers.conv2d(output, 3, 1, strides=1, padding='same', name='toRGB')
        toRGB = apply_bias(toRGB,"2")
        return output, toRGB

def generator16x16(x, reuse = False):
    with tf.variable_scope(name_or_scope = "Gen16x16") as scope:
        output = tf.layers.conv2d_transpose(x, 512, 1, strides=2)
        output = pixel_norm(lrelu(apply_bias(output,"0")))
        output = tf.layers.conv2d(output, 512, 3, strides=1, padding='same', name='conv')
        output = pixel_norm(lrelu(apply_bias(output,"1")))
        toRGB = tf.layers.conv2d(output, 3, 1, strides=1, padding='same', name='toRGB')
        toRGB = apply_bias(toRGB,"2")
        return output, toRGB

def generator32x32(x, reuse = False):
    with tf.variable_scope(name_or_scope = "Gen32x32") as scope:
        output = tf.layers.conv2d_transpose(x, 512, 1, strides=2)
        output = pixel_norm(lrelu(apply_bias(output,"0")))
        output = tf.layers.conv2d(output, 512, 3, strides=1, padding='same', name='conv')
        output = pixel_norm(lrelu(apply_bias(output,"1")))
        toRGB = tf.layers.conv2d(output, 3, 1, strides=1, padding='same', name='toRGB')
        toRGB = apply_bias(toRGB,"2")
        return output, toRGB

def generator64x64(x, reuse = False):
    with tf.variable_scope(name_or_scope = "Gen64x64") as scope:
        output = tf.layers.conv2d_transpose(x, 512, 1, strides=2)
        output = pixel_norm(lrelu(apply_bias(output,"0")))
        output = tf.layers.conv2d(output, 256, 3, strides=1, padding='same', name='conv')
        output = pixel_norm(lrelu(apply_bias(output,"1")))
        toRGB = tf.layers.conv2d(output, 3, 1, strides=1, padding='same', name='toRGB')
        toRGB = apply_bias(toRGB,"2")
        return output, toRGB

def generator128x128(x, reuse = False):
    with tf.variable_scope(name_or_scope = "Gen128x128") as scope:
        output = tf.layers.conv2d_transpose(x, 256, 1, strides=2)
        output = pixel_norm(lrelu(apply_bias(output,"0")))
        output = tf.layers.conv2d(output, 128, 3, strides=1, padding='same', name='conv')
        output = pixel_norm(lrelu(apply_bias(output,"1")))
        toRGB = tf.layers.conv2d(output, 3, 1, strides=1, padding='same', name='toRGB')
        toRGB = apply_bias(toRGB,"2")
        return output, toRGB

#==================================================================================

def discriminator4x4(x, fromRGB, reuse = False):
    with tf.variable_scope(name_or_scope = "Dis4x4", reuse=reuse) as scope:
        if fromRGB:
            x = tf.layers.conv2d(x, 512, 1, strides=1, padding='same', name='fromRGB_4x4')
            x = lrelu(apply_bias(x,"0"))
        output = tf.layers.conv2d(x, 512, 3, strides=1, padding='same', name='conv')
        output = lrelu(apply_bias(output,"1"))
        output = tf.layers.dense(output, 256, name='dense0')
        output = lrelu(apply_bias(output,"2"))
        output = tf.layers.dense(output, 1, name='dense1')
        output = lrelu(apply_bias(output,"3"))
        return output

def discriminator8x8(x, fromRGB, reuse = False):
    with tf.variable_scope(name_or_scope = "Dis8x8", reuse=reuse) as scope:
        if fromRGB:
            x = tf.layers.conv2d(x, 512, 1, strides=1, padding='same', name='fromRGB_8x8')
            x = lrelu(apply_bias(x,"0"))
        output = tf.layers.conv2d(x, 512, 3, strides=1, padding='same', name='conv0')
        output = lrelu(apply_bias(output,"1"))
        output = tf.layers.conv2d(output, 512, 3, strides=1, padding='same', name='conv1')
        output = lrelu(apply_bias(output,"2"))
        output = downscale2d(output)
        return output

def discriminator16x16(x, fromRGB, reuse = False):
    with tf.variable_scope(name_or_scope = "Dis16x16", reuse=reuse) as scope:
        if fromRGB:
            x = tf.layers.conv2d(x, 512, 1, strides=1, padding='same', name='fromRGB_16x16')
            x = lrelu(apply_bias(x,"0"))
        output = tf.layers.conv2d(x, 512, 3, strides=1, padding='same', name='conv0')
        output = lrelu(apply_bias(output,"1"))
        output = tf.layers.conv2d(output, 512, 3, strides=1, padding='same', name='conv1')
        output = lrelu(apply_bias(output,"2"))
        output = downscale2d(output)
        return output

def discriminator32x32(x, fromRGB, reuse = False):
    with tf.variable_scope(name_or_scope = "Dis32x32", reuse=reuse) as scope:
        if fromRGB:
            x = tf.layers.conv2d(x, 512, 1, strides=1, padding='same', name='fromRGB_32x32')
            x = lrelu(apply_bias(x,"0"))
        output = tf.layers.conv2d(x, 512, 3, strides=1, padding='same', name='conv0')
        output = lrelu(apply_bias(output,"1"))
        output = tf.layers.conv2d(output, 512, 3, strides=1, padding='same', name='conv1')
        output = lrelu(apply_bias(output,"2"))
        output = downscale2d(output)
        return output

def discriminator64x64(x, fromRGB, reuse = False):
    with tf.variable_scope(name_or_scope = "Dis64x64", reuse=reuse) as scope:
        if fromRGB:
            x = tf.layers.conv2d(x, 256, 1, strides=1, padding='same', name='fromRGB_64x64')
            x = lrelu(apply_bias(x,"0"))
        output = tf.layers.conv2d(x, 256, 3, strides=1, padding='same', name='conv0')
        output = lrelu(apply_bias(output,"1"))
        output = tf.layers.conv2d(output, 512, 3, strides=1, padding='same', name='conv1')
        output = lrelu(apply_bias(output,"2"))
        output = downscale2d(output)
        return output

def discriminator128x128(x, fromRGB, reuse = False):
    with tf.variable_scope(name_or_scope = "Dis128x128", reuse=reuse) as scope:
        if fromRGB:
            x = tf.layers.conv2d(x, 128, 1, strides=1, padding='same', name='fromRGB_128x128')
            x = lrelu(apply_bias(x,"0"))
        output = tf.layers.conv2d(x, 128, 3, strides=1, padding='same', name='conv0')
        output = lrelu(apply_bias(output,"1"))
        output = tf.layers.conv2d(output, 256, 3, strides=1, padding='same', name='conv1')
        output = lrelu(apply_bias(output,"2"))
        output = downscale2d(output)
        return output




#==================================================================================

def random_noise(batch_size):
    rd = np.random.uniform(size=[batch_size, 512])
    return rd / np.linalg.norm(rd, axis=1)[:,None]

#==================================================================================

def D_loss(result_fake, result_real):
    #loss = tf.reduce_mean(tf.log(result_real) + tf.log(1 - result_fake))
    #loss = -tf.reduce_mean(result_real) + tf.reduce_mean(result_fake)
    #"""
    real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=result_real, 
        labels=tf.ones_like(result_real)
    ))
    fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=result_fake, 
        labels=tf.zeros_like(result_fake)
    ))
    loss = real_loss + fake_loss
    #"""
    return loss

def G_loss(result_fake):
    # loss = tf.log(result_fake)
    #loss = -tf.reduce_mean(result_fake) 
    #"""
    loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=result_fake, 
            labels=tf.ones_like(result_fake)
    )
    loss = tf.reduce_mean(loss)
    #"""
    return loss
    
#==================================================================================
#==================================================================================

g = tf.Graph()
with g.as_default():
    Z = tf.placeholder(tf.float32, [None, 512])
    alpha = tf.placeholder(tf.float32, [])
    curr_alpha = tf.placeholder(tf.float32, [])
    s = (tf.sign(alpha - curr_alpha) + 1) / 2
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001 , beta1=0 , beta2=0.99 , epsilon=1e-8)
    
    Dis_dict = {4:discriminator4x4,
                8:discriminator8x8,
                16:discriminator16x16,
                32:discriminator32x32,
                64:discriminator64x64,
                128:discriminator128x128
    }
    Gen_dict = {4:generator4x4,
                8:generator8x8,
                16:generator16x16,
                32:generator32x32,
                64:generator64x64,
                128:generator128x128
    }

    def Dis(x, size, reuse):
        x = Dis_dict[size](x, fromRGB=True, reuse=reuse)
        size /= 2
        while size >= 4:
            x = Dis_dict[size](x, fromRGB=False, reuse=True)
            size /= 2
        return x
    
    def Gen(x, size, cur_size=4):
        while cur_size <= size:
            x, x_rgb = Gen_dict[cur_size](x)
            cur_size *= 2
        return x, x_rgb


    #=====================================================================
    
    real_4 = tf.placeholder(tf.float32, [None, 4, 4, 3])
    fake_4, fake_4_rgb = Gen(Z, 4)
    result_fake_4 = Dis(fake_4_rgb, 4, False)
    result_real_4 = Dis(real_4, 4, True)
    
    d_loss_4 = D_loss(result_fake_4, result_real_4)
    g_loss_4 = G_loss(result_fake_4)

    t_vars = tf.trainable_variables()
    g_vars_4 = [var for var in t_vars if "Gen4x4" in var.name]
    d_vars_4 = [var for var in t_vars if "Dis4x4" in var.name]
    
    g_train_4 = optimizer.minimize(g_loss_4, var_list = g_vars_4)
    d_train_4 = optimizer.minimize(d_loss_4, var_list = d_vars_4)


    #=====================================================================
    
    real_8 = tf.placeholder(tf.float32, [None, 8, 8, 3])
    fake_8, fake_8_rgb = Gen(fake_4, 8, 8)
    fake_4_to_8 = upscale2d(fake_4_rgb)
 
    fake_8_chosen = (s * fake_8_rgb) + ((1-s) * fake_4_to_8)
    
    result_fake_8_rgb = Dis(fake_8_chosen, 8, False)
    result_fake_4_to_8 = Dis(downscale2d(fake_8_chosen), 4, True)

    result_fake_8 = s * result_fake_8_rgb + (1-s) * result_fake_4_to_8

    result_real_8 = Dis(real_8, 8, True)

    d_loss_8 = D_loss(result_fake_8, result_real_8)
    g_loss_8 = G_loss(result_fake_8)

    t_vars = tf.trainable_variables()
    g_vars_8 = [var for var in t_vars if "Gen8x8" in var.name]
    d_vars_8 = [var for var in t_vars if "Dis8x8" in var.name]
    
    g_train_8 = optimizer.minimize(g_loss_8, var_list = g_vars_8)
    d_train_8 = optimizer.minimize(d_loss_8, var_list = d_vars_8)


    #=====================================================================
    
    real_16 = tf.placeholder(tf.float32, [None, 16, 16, 3])
    fake_16, fake_16_rgb = Gen(fake_8, 16, 16)
    fake_8_to_16 = upscale2d(fake_8_rgb)
 
    fake_16_chosen = (s * fake_16_rgb) + ((1-s) * fake_8_to_16)
    
    result_fake_16_rgb = Dis(fake_16_chosen, 16, False)
    result_fake_8_to_16 = Dis(downscale2d(fake_16_chosen), 8, True)

    result_fake_16 = s * result_fake_16_rgb + (1-s) * result_fake_8_to_16

    result_real_16 = Dis(real_16, 16, True)

    d_loss_16 = D_loss(result_fake_16, result_real_16)
    g_loss_16 = G_loss(result_fake_16)

    t_vars = tf.trainable_variables()
    g_vars_16 = [var for var in t_vars if "Gen16x16" in var.name]
    d_vars_16 = [var for var in t_vars if "Dis16x16" in var.name]
    
    g_train_16 = optimizer.minimize(g_loss_16, var_list = g_vars_16)
    d_train_16 = optimizer.minimize(d_loss_16, var_list = d_vars_16)


    #=====================================================================
    
    real_32 = tf.placeholder(tf.float32, [None, 32, 32, 3])
    fake_32, fake_32_rgb = Gen(fake_16, 32, 32)
    fake_16_to_32 = upscale2d(fake_16_rgb)
 
    fake_32_chosen = (s * fake_32_rgb) + ((1-s) * fake_16_to_32)
    
    result_fake_32_rgb = Dis(fake_32_chosen, 32, False)
    result_fake_16_to_32 = Dis(downscale2d(fake_32_chosen), 16, True)

    result_fake_32 = s * result_fake_32_rgb + (1-s) * result_fake_16_to_32

    result_real_32 = Dis(real_32, 32, True)

    d_loss_32 = D_loss(result_fake_32, result_real_32)
    g_loss_32 = G_loss(result_fake_32)

    t_vars = tf.trainable_variables()
    g_vars_32 = [var for var in t_vars if "Gen32x32" in var.name]
    d_vars_32 = [var for var in t_vars if "Dis32x32" in var.name]
    
    g_train_32 = optimizer.minimize(g_loss_32, var_list = g_vars_32)
    d_train_32 = optimizer.minimize(d_loss_32, var_list = d_vars_32)


    #=====================================================================
    
    real_64 = tf.placeholder(tf.float32, [None, 64, 64, 3])
    fake_64, fake_64_rgb = Gen(fake_32, 64, 64)
    fake_32_to_64 = upscale2d(fake_32_rgb)
 
    fake_64_chosen = (s * fake_64_rgb) + ((1-s) * fake_32_to_64)
    
    result_fake_64_rgb = Dis(fake_64_chosen, 64, False)
    result_fake_32_to_64 = Dis(downscale2d(fake_64_chosen), 32, True)

    result_fake_64 = s * result_fake_64_rgb + (1-s) * result_fake_32_to_64

    result_real_64 = Dis(real_64, 64, True)

    d_loss_64 = D_loss(result_fake_64, result_real_64)
    g_loss_64 = G_loss(result_fake_64)

    t_vars = tf.trainable_variables()
    g_vars_64 = [var for var in t_vars if "Gen64x64" in var.name]
    d_vars_64 = [var for var in t_vars if "Dis64x64" in var.name]
    
    g_train_64 = optimizer.minimize(g_loss_64, var_list = g_vars_64)
    d_train_64 = optimizer.minimize(d_loss_64, var_list = d_vars_64)


    #=====================================================================
    
    real_128 = tf.placeholder(tf.float32, [None, 128, 128, 3])
    fake_128, fake_128_rgb = Gen(fake_64, 128, 128)
    fake_64_to_128 = upscale2d(fake_64_rgb)
 
    fake_128_chosen = (s * fake_128_rgb) + ((1-s) * fake_64_to_128)
    
    result_fake_128_rgb = Dis(fake_128_chosen, 128, False)
    result_fake_64_to_128 = Dis(downscale2d(fake_128_chosen), 64, True)

    result_fake_128 = s * result_fake_128_rgb + (1-s) * result_fake_64_to_128

    result_real_128 = Dis(real_128, 128, True)

    d_loss_128 = D_loss(result_fake_128, result_real_128)
    g_loss_128 = G_loss(result_fake_128)

    t_vars = tf.trainable_variables()
    g_vars_128 = [var for var in t_vars if "Gen128x128" in var.name]
    d_vars_128 = [var for var in t_vars if "Dis128x128" in var.name]
    
    g_train_128 = optimizer.minimize(g_loss_128, var_list = g_vars_128)
    d_train_128 = optimizer.minimize(d_loss_128, var_list = d_vars_128)





    #=====================================================================
    g_train_dict = {
            4:g_train_4,
            8:g_train_8,
            16:g_train_16,
            32:g_train_32,
            64:g_train_64,
            128:g_train_128
    }
    d_train_dict = {
            4:d_train_4,
            8:d_train_8,
            16:d_train_16,
            32:d_train_32,
            64:d_train_64,
            128:d_train_128
    }
    g_loss_dict = {
            4:g_loss_4,
            8:g_loss_8,
            16:g_loss_16,
            32:g_loss_32,
            64:g_loss_64,
            128:g_loss_128
    }
    d_loss_dict = {
            4:d_loss_4,
            8:d_loss_8,
            16:d_loss_16,
            32:d_loss_32,
            64:d_loss_64,
            128:d_loss_128
    }
    real_dict = {
            4:real_4,
            8:real_8,
            16:real_16,
            32:real_32,
            64:real_64,
            128:real_128
    }
    fake_rgb_dict = {
            4:fake_4_rgb,
            8:fake_8_rgb,
            16:fake_16_rgb,
            32:fake_32_rgb,
            64:fake_64_rgb,
            128:fake_128_rgb
    }







#==================================================================================
#==================================================================================


with tf.Session(graph = g) as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    
    img_size = 4
    flag = False
    al = 1.0

    epoch_per_size = [100, 200, 400, 800, 1600, 3200]
    for i in range(1,len(epoch_per_size)):
        epoch_per_size[i] += epoch_per_size[i-1]

    total_batches = int(len(train_x[img_size]) / batch_size)
    for epoch in range(epoch_per_size[-1]):
        for batch in range(total_batches):
            batch_x = train_x[img_size][batch*batch_size : (batch+1)*batch_size]
            noise = random_noise(batch_size)

            sess.run(g_train_dict[img_size], feed_dict={
                Z:noise, 
                alpha:al, 
                curr_alpha:np.random.uniform()
            })
            sess.run(d_train_dict[img_size], feed_dict={
                Z:noise, 
                alpha:al, 
                curr_alpha:np.random.uniform(), 
                real_dict[img_size]:batch_x
            })

            gl,dl = sess.run([g_loss_dict[img_size],d_loss_dict[img_size]],
                    feed_dict = {
                        Z:noise, 
                         alpha:al, 
                        curr_alpha:np.random.uniform(), 
                        real_dict[img_size]:batch_x
            })
        
        print("======= Epoch : ", epoch, " =======")
        print("generator",img_size,": ", gl)
        print("discriminator",img_size,": ", dl)
        print("alpha",img_size,": ",al)
    
        samples = 20
        if epoch % 5 == 0:
            sample_noise = random_noise(samples)
            generated = sess.run(fake_rgb_dict[img_size], feed_dict={
                Z:sample_noise,
                alpha:al, 
                curr_alpha:np.random.uniform()
            })
            for i in range(samples):
                mpimg.imsave('./epoch/epoch'+str(epoch)+'_'+str(img_size)+'_'+str(i)+'.png', ((generated[i]+1)/2).clip(0,1), format='png')
                
        
        if epoch in epoch_per_size:
            img_size *= 2
            flag = True
            al = 0

        if flag:
            al += 0.01
            if al >= 1.0:
                al = 1.0
                flag = False

        save_path = saver.save(sess, "./ckpt/epoch"+str(epoch)+".ckpt")






 
