
# coding: utf-8

# In[1]:


import tensorflow as tf
import pandas as pd
import numpy as np
import sys
import sklearn as sk
from sklearn import preprocessing as pre
import argparse

# In[2]:


def checkArgs(args=None):
    parser = argparse.ArgumentParser(description='LSTM')
    parser.add_argument('--lr',help = 'Learning Rate', default = 0.001,type=float)
    parser.add_argument('--batch_size',help = 'Mini-Batch Size', default= 250,type=int)
    parser.add_argument('--save_dir',help = 'Save location of the model', required=True)
    parser.add_argument('--dropout_prob',help = 'dropout probability',default=0.4,type=float)
    parser.add_argument('--decode_method',help = '[GREEDY/BEAM]', default='GREEDY')
    parser.add_argument('--beam_width',help = 'Beam width for Beam Search Decoder', default=5,type=int)
    parser.add_argument('--attention',help = 'Attention', action='store_true',default=False)
    args = parser.parse_args(args)
    return args

args=checkArgs(sys.argv[1:])
print "Starting With : ",args


#HYPER PARAMS
GO='<go>'
STOP='<stop>'
PAD='<pad>'

LR=args.lr
BATCH=args.batch_size
SAVE=args.save_dir
ATTEND=args.attention
DROPOUT=args.dropout_prob
if args.decode_method=='BEAM':
    BEAM=True
else:
    BEAM=False

BEAM_WIDTH=args.beam_width

EPOCHS=1


# In[3]:


#PREPROCESSING STARTS HERE
def breaker(listofsentences,start=True):
    '''returns a list of list of strings'''
    if listofsentences==None:
        return None
    lst=[]
    for i in listofsentences:
        t=i.split()
        if start:
            t=[GO]+t+[PAD]
        else:
            t=[GO]+t+[STOP]
        lst.append(t)
    return lst


# In[4]:


def read(data='train'):
    COMBINED='./WeatherGov/'+data+'/'+data+'.combined'
    SUMMARY='./WeatherGov/'+data+'/summaries.txt'
    op_sec=None
    with open(COMBINED) as f:
        content = f.readlines()
    ip_sec = [x.strip() for x in content]
    if data!='test':
        with open(SUMMARY) as f:
            content = f.readlines()
        op_sec = [x.strip() for x in content]
        
    return breaker(ip_sec,True),breaker(op_sec,False)


# In[5]:


def listofwords(data):
    '''takes a list of sentences nd returns vocab'''
    a=[]
    for i in data:
        for j in i:
            if j not in a:
                a.append(j)
    return a


# In[12]:


training_ip_sec,training_op_sec=read('train')
valid_ip_sec,valid_op_sec=read('dev')
test_ip_sec,_=read('test')


# In[13]:


ip_vocab=listofwords(training_ip_sec)
op_vocab=listofwords(training_op_sec)


# In[14]:


ip_vocab_size,op_vocab_size=len(ip_vocab),len(op_vocab)


# In[15]:


training_ip_sec_len=[len(i) for i in training_ip_sec]
training_op_sec_len=[len(i) for i in training_op_sec]
valid_ip_sec_len=[len(i) for i in valid_ip_sec]
valid_op_sec_len=[len(i) for i in valid_op_sec]
test_ip_sec_len=[len(i) for i in test_ip_sec]


# In[16]:


pre_ip=pre.LabelEncoder()
pre_ip.fit(ip_vocab)

onehoter=np.identity(len(pre_ip.classes_))

#word to int
training_ip_sect=[pre_ip.transform(i) for i in training_ip_sec]
valid_ip_sect=[pre_ip.transform(i) for i in valid_ip_sec]
test_ip_sect=[pre_ip.transform(i) for i in test_ip_sec]

#adding pads
maxi=0
for i in training_ip_sect:
    maxi=max(maxi,len(i))
training_ip_sect0=[np.pad(i,(0,maxi-len(i)),'constant',constant_values=pre_ip.transform([PAD])) for i in training_ip_sect]
valid_ip_sect0=[np.pad(i,(0,maxi-len(i)),'constant',constant_values=pre_ip.transform([PAD])) for i in valid_ip_sect]
test_ip_sect0=[np.pad(i,(0,maxi-len(i)),'constant',constant_values=pre_ip.transform([PAD])) for i in test_ip_sect]

#making them onehot
training_ip_sectt=[[onehoter[i] for i in j] for j in training_ip_sect0]
valid_ip_sectt=[[onehoter[i] for i in j] for j in valid_ip_sect0]
test_ip_sectt=[[onehoter[i] for i in j] for j in test_ip_sect0]


#del training_ip_sec,valid_ip_sec,test_ip_sec
#del training_ip_sect,valid_ip_sect,test_ip_sect
#del training_ip_sect0,valid_ip_sect0,test_ip_sect0

#OP sec
pre_op=pre.LabelEncoder()
pre_op.fit(op_vocab)

onehoter2=np.identity(len(pre_op.classes_))

#word to int
training_op_sect=[pre_op.transform(i) for i in training_op_sec]
valid_op_sect=[pre_op.transform(i) for i in valid_op_sec]

#adding pad
maxi=0
for i in training_op_sect:
    maxi=max(maxi,len(i))
training_op_sect0=[np.pad(i,(0,maxi-len(i)),'constant',constant_values=pre_op.transform([STOP])) for i in training_op_sect]
valid_op_sect0=[np.pad(i,(0,maxi-len(i)),'constant',constant_values=pre_op.transform([STOP])) for i in valid_op_sect]


#int to onehot
training_op_sectt=[[onehoter2[i] for i in j] for j in training_op_sect0]
valid_op_sectt=[[onehoter2[i] for i in j] for j in valid_op_sect0]

training_op_sectt_nopad=[[onehoter2[i] for i in j] for j in training_op_sect]
valid_op_sectt_nopad=[[onehoter2[i] for i in j] for j in valid_op_sect]


del training_op_sec,valid_op_sec
del training_op_sect,valid_op_sect
#del training_op_sect0,valid_op_sect0


# In[11]:


maxi=0
for i in training_op_sect0:
    maxi=max(maxi,len(i))
#PREPROCESSING ENDS HERE


# In[12]:


#Graph building starts here
embedding_size=256
lstm_units=512
tf.reset_default_graph()


# In[13]:


#place holders
source_seq = tf.placeholder(shape=(None, None), dtype=tf.int32)
target_seq = tf.placeholder(shape=(None, None), dtype=tf.int32)
source_seq_len = tf.placeholder(shape=(None,), dtype=tf.int32)
target_seq_len = tf.placeholder(shape=(None,), dtype=tf.int32)
no_start_target_seq = tf.placeholder(shape=(None, None), dtype=tf.int32)
trainer = tf.placeholder(shape=(None),dtype=tf.bool)
batch_size = tf.placeholder(shape=(None),dtype=tf.int32)
real_target_seq_len= tf.placeholder(shape=(None,), dtype=tf.int32)


# In[14]:


#input and output embeddings
embedding_matrix_encode = tf.get_variable(
    name="embedding_matrix_en",
    shape=[ip_vocab_size, embedding_size],
    dtype=tf.float32)
embedding_matrix_decode = tf.get_variable(
    name="embedding_matrix_de",
    shape=[op_vocab_size, embedding_size],
    dtype=tf.float32)
source_seq_embedded = tf.nn.embedding_lookup(embedding_matrix_encode, source_seq) 
decoder_input_embedded = tf.nn.embedding_lookup(embedding_matrix_decode, target_seq) 


# In[15]:


#Encoder
(encoder_outputs_fw,encoder_outputs_bw) ,(encoder_state_fw,encoder_state_bw) = tf.nn.bidirectional_dynamic_rnn(
    tf.contrib.rnn.LSTMCell(lstm_units),tf.contrib.rnn.LSTMCell(lstm_units),
    source_seq_embedded,
    sequence_length=source_seq_len,
    dtype=tf.float32)


# In[16]:


#Attention guys
attention_states=tf.concat((encoder_outputs_fw,encoder_outputs_bw),1)
attention_mechanism = tf.contrib.seq2seq.LuongAttention(lstm_units*2, attention_states,)


# In[17]:


#Concat and Dropout
encoder_final_state_c = tf.layers.dropout(inputs=tf.concat(
    (encoder_state_fw.c, encoder_state_bw.c), 1),rate=DROPOUT,training=trainer)

encoder_final_state_h = tf.layers.dropout(inputs=tf.concat(
    (encoder_state_fw.h, encoder_state_bw.h), 1),rate=DROPOUT,training=trainer)

encoder_final_state = tf.contrib.rnn.LSTMStateTuple(
    c=encoder_final_state_c,
    h=encoder_final_state_h
)


# In[18]:


#Projection layer and decoder cell
output_layer = tf.layers.Dense(op_vocab_size)

decoder_cell=tf.contrib.rnn.LSTMCell(lstm_units*2)


# In[19]:


decoder_initial_state=encoder_final_state


# In[20]:


# modify decoder cell if attention
if ATTEND:
    decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
        decoder_cell, attention_mechanism,
        attention_layer_size=lstm_units*2,alignment_history=True)


# In[21]:


# modify decoder initial state if attention
if ATTEND:
    decoder_initial_state = decoder_cell.zero_state(BATCH, tf.float32).clone(cell_state=decoder_initial_state)


# In[22]:


#Training helper and decoder
helper = tf.contrib.seq2seq.TrainingHelper(decoder_input_embedded,target_seq_len)
decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, initial_state=decoder_initial_state,output_layer=output_layer)#,output_layer=projection_layer)
outputs, state, seq_len = tf.contrib.seq2seq.dynamic_decode(decoder)
logits = outputs.rnn_output


# In[23]:


#Inference helper(greedy) and decoder
helper2 = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding_matrix_decode,tf.fill([batch_size],
                                                    np.int32(pre_op.transform([GO])[0])),
                                                   np.int32(pre_op.transform([STOP])[0]))


decoder2 = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper2, decoder_initial_state,output_layer=output_layer)#,output_layer=projection_layer)

outputs, state, seq_len = tf.contrib.seq2seq.dynamic_decode(decoder2,maximum_iterations=maxi+10)

translations_logits = outputs.rnn_output
trs=outputs.sample_id


# In[24]:


#beam search and decoder
if BEAM:
    decoder_initial_state_beam = tf.contrib.seq2seq.tile_batch(
        decoder_initial_state, multiplier=BEAM_WIDTH)

    # Define a beam-search decoder
    decoder3 = tf.contrib.seq2seq.BeamSearchDecoder(
            cell=decoder_cell,
            embedding=embedding_matrix_decode,
            start_tokens=tf.fill([batch_size],np.int32(pre_op.transform([GO])[0])),
            end_token=np.int32(pre_op.transform([STOP])[0]),
            initial_state=decoder_initial_state_beam,
            beam_width=BEAM_WIDTH,
            output_layer=output_layer,
            length_penalty_weight=0.0)
    outputs, state, seq_len = tf.contrib.seq2seq.dynamic_decode(decoder3,maximum_iterations=maxi+10)


    trs_beam=outputs.predicted_ids


# In[25]:


#loss and optimizer
cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=no_start_target_seq,logits=logits)

target_weights = tf.sequence_mask(real_target_seq_len, target_seq_len[0], dtype=logits.dtype)

loss=tf.reduce_sum(cross_entropy*target_weights)
train = tf.train.AdamOptimizer(LR).minimize(loss)


# In[26]:


#dont touch
maxtlen=max(training_op_sec_len)
maxvlen=max(valid_op_sec_len)
t_newlen=[maxtlen-1 for i in range(len(training_op_sec_len))]
v_newlen=[maxtlen-1 for i in range(len(valid_op_sec_len))]


# In[27]:


sess=tf.InteractiveSession()
tf.global_variables_initializer().run()


# In[ ]:


#training starts here
training_losses=[]
valid_losses=[]
for j in range(EPOCHS):
    training_loss=0
    for i in range(len(training_ip_sectt)/BATCH):
        start=i*BATCH
        stop=(i+1)*BATCH
        
        _,lost=sess.run([train,loss],feed_dict={source_seq:training_ip_sect0[start:stop],
                                                target_seq:training_op_sect0[start:stop],
                                              source_seq_len:training_ip_sec_len[start:stop],
                                                target_seq_len:t_newlen[start:stop],
                                                real_target_seq_len:training_op_sec_len[start:stop],
                                                no_start_target_seq:np.array(training_op_sect0[start:stop])[:,1:],
                                                batch_size:BATCH,
                                                trainer:True
                                                })
        
        training_loss+=lost
        
    #calculate t_loss
    training_losses.append(training_loss/len(training_ip_sectt))
    
    #calculate v_loss
    validation_loss=0
    for k in range(len(valid_ip_sectt)/BATCH):
        start=k*BATCH
        stop=(k+1)*BATCH

        lost=sess.run(loss,feed_dict={source_seq:valid_ip_sect0[start:stop],
                                                target_seq:valid_op_sect0[start:stop],
                                              source_seq_len:valid_ip_sec_len[start:stop],
                                                target_seq_len:v_newlen[start:stop],
                                                    real_target_seq_len:valid_op_sec_len[start:stop],
                                                    no_start_target_seq:np.array(valid_op_sect0[start:stop])[:,1:],
                                                    batch_size:BATCH,
                                                    trainer:False})
        validation_loss += lost
        
    valid_losses.append(validation_loss/len(valid_ip_sectt))
    print "Epoch:%d training loss%.4f: valid loss:%.4f"% (j,training_losses[-1],valid_losses[-1])


# In[ ]:


#saving loss values
filer=open('./losses.txt','w+')
for tl,vl in zip(training_losses,valid_losses):
    filer.write("%f,%f\n" % (tl,vl))
filer.close()


# In[ ]:


#visualize attention starts here


# In[ ]:


# In[ ]:



#generate test data summaries
data=test_ip_sect0
data_len=test_ip_sec_len
gen_sum=[]
for i in range(len(data)/BATCH):
    start=i*BATCH
    stop=(i+1)*BATCH
    if BEAM:
        load_trs=trs_beam
    else:
        load_trs=trs
    y=sess.run(load_trs,feed_dict={source_seq:data[start:stop],
                                               source_seq_len:data_len[start:stop],
                                              batch_size:BATCH,
                                                    trainer:False
                                                })
    if BEAM:
        y=y[:,:,0]
    for t in y:
        gen_sum.append(t)

start=len(data)-BATCH
stop=len(data)
if BEAM:
    load_trs=trs_beam
else:
    load_trs=trs
y=sess.run(load_trs,feed_dict={source_seq:data[start:stop],
                                            source_seq_len:data_len[start:stop],
                                            batch_size:BATCH,
                                                trainer:False
                                            })
if BEAM:
    y=y[:,:,0]

y=y[-(len(data)-len(gen_sum)):]
for t in y:
    gen_sum.append(t)

# In[ ]:


#processing summaries
summs=[]
for i in gen_sum:
    summ=''
    for j in i:
        
        if j!=205:
            summ = summ+' '+pre_op.inverse_transform(j)
    summs.append(summ[1:])


# In[ ]:


#save summaries
filer=open('./TestSummaries.txt','w+')
for item in summs:
    filer.write("%s\n" % item)
filer.close()


# In[ ]:


#save model
saver = tf.train.Saver()
saver.save(sess, SAVE+'/model',global_step=1000)


# In[ ]:


#restore model
#saver = tf.train.import_meta_graph('./model-1000.meta')
#saver.restore(sess,tf.train.latest_checkpoint('./'))


# In[18]:

