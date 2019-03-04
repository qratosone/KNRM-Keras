import tensorflow as tf
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import *
from keras.initializers import Constant, RandomNormal, RandomUniform
from keras.activations import softmax

text1_maxlen=10
text2_maxlen=1000
vocab_size=342326
embed_size=300
embed_path="./embed_glove_d300"
embed=None
hyper_param_dict={
        "kernel_num": 21,
        "sigma": 0.1,
	    "exact_sigma": 0.001,
	    "dropout_rate": 0.0
    }

def show_layer_info(layer_name, layer_out):
    print('[layer]: %s\t[shape]: %s \n' % (layer_name,str(layer_out.get_shape().as_list())))



def rank_hinge_loss(kwargs=None):
    margin = 1.
    if isinstance(kwargs, dict) and 'margin' in kwargs:
        margin = kwargs['margin']

    def _margin_loss(y_true, y_pred):
        # output_shape = K.int_shape(y_pred)
        y_pos = Lambda(lambda a: a[::2, :], output_shape= (1,))(y_pred)
        y_neg = Lambda(lambda a: a[1::2, :], output_shape= (1,))(y_pred)
        loss = K.maximum(0., margin + y_neg - y_pos)
        return K.mean(loss)
    return _margin_loss

def read_embedding(filename):
    embed = {}
    for line in open(filename):
        line = line.strip().split()
        embed[int(line[0])] = list(map(float, line[1:]))
    print('[%s]\n\tEmbedding size: %d' % (filename, len(embed)), end='\n')
    return embed
def convert_embed_2_numpy(embed_dict, max_size=0, embed=None):
    feat_size = len(embed_dict[list(embed_dict.keys())[0]])
    if embed is None:
        embed = np.zeros((max_size, feat_size), dtype=np.float32)

    if len(embed_dict) > len(embed):
        raise Exception("vocab_size %d is larger than embed_size %d, change the vocab_size in the config!"
                        % (len(embed_dict), len(embed)))

    for k in embed_dict:
        embed[k] = np.array(embed_dict[k])
    print('Generate numpy embed:', str(embed.shape), end='\n')
    return embed

def get_embed():
    global embed
    embed_dict = read_embedding(filename=embed_path)
    _PAD_ = vocab_size - 1
    embed_dict[_PAD_] = np.zeros((embed_size, ), dtype=np.float32)
    embed_arr = np.float32(np.random.uniform(-0.2, 0.2, [vocab_size, embed_size]))
    embed = convert_embed_2_numpy(embed_dict, embed = embed_arr)
    print("Embed loaded.")

def build():
    def Kernel_layer(mu,sigma):
        def kernel(x):
            return K.tf.exp(-0.5 * (x - mu) * (x - mu) / sigma / sigma)
        return Activation(kernel)
    print("building model")
    query = Input(name='query', shape=(text1_maxlen,))#输入查询
    show_layer_info('Input', query)
    doc = Input(name='doc', shape=(text2_maxlen,))#输入文档
    show_layer_info('Input', doc)
    embedding = Embedding(vocab_size, embed_size, weights=[embed], trainable=True)
    q_embed = embedding(query)#查询的embedding
    show_layer_info('Embedding', q_embed)
    d_embed = embedding(doc)#文档的embedding
    show_layer_info('Embedding', d_embed)
    mm = Dot(axes=[2, 2], normalize=True)([q_embed, d_embed]) #交互矩阵
    show_layer_info('Dot', mm)

    KM = [] #Kernel Pooling
    for i in range(hyper_param_dict['kernel_num']):
        # 均匀建立一批Kernel
        mu = 1. / (hyper_param_dict['kernel_num'] - 1) + (2. * i) / (hyper_param_dict['kernel_num'] - 1) - 1.0
        sigma = hyper_param_dict['sigma']
        if mu > 1.0: #固定mu调整sigma？
            sigma = hyper_param_dict['exact_sigma']
            mu = 1.0
        mm_exp = Kernel_layer(mu, sigma)(mm)#用生成的mu和sigma建立一个kernel
        show_layer_info('Exponent of mm:', mm_exp)
        mm_doc_sum = Lambda(lambda x: K.tf.reduce_sum(x,2))(mm_exp)#求和（对每一列，一个查询词对每一个doc term的相似度求RBFK之后加和）
        #得到针对当前Kernel，当前查询词和所有文档词的和，经过RBFK
        show_layer_info('Sum of document', mm_doc_sum)
        mm_log = Activation(K.tf.log1p)(mm_doc_sum)#加上对数激活
        show_layer_info('Logarithm of sum', mm_log)
        mm_sum = Lambda(lambda x: K.tf.reduce_sum(x, 1))(mm_log) #求和（每一行，即所有查询词）
        #得到针对当前Kernel，所有查询词的求和
        show_layer_info('Sum of all exponent', mm_sum)
        KM.append(mm_sum)
    #此时KM里有每一个Kernel对当前所有查询词的求和
    Phi = Lambda(lambda x: K.tf.stack(x, 1))(KM)
    show_layer_info('Stack', Phi)
    out_ = Dense(1, kernel_initializer=RandomUniform(minval=-0.014, maxval=0.014), bias_initializer='zeros')(Phi)
    show_layer_info('Dense', out_)

    model = Model(inputs=[query, doc], outputs=[out_])
    return model

if __name__=="__main__":
    get_embed()
    assert embed is not None
    model=build()
    model.compile(loss=rank_hinge_loss({ "margin": 1.0 }),optimizer="adam")
    print("model loaded.")
    model.load_weights("./knrm.dssa.weights.500")
    print("weights loaded.")


