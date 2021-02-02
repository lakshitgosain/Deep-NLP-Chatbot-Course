#Building Chatbot


#Data Preprocessing
import numpy as np
import tensorflow as tf
import re
import time
from collections import Counter
#importing the Dataset
lines=open('movie_lines.txt',encoding='utf-8',errors='ignore').read().split('\n')
conversations=open('movie_conversations.txt',encoding='utf-8',errors='ignore').read().split('\n')

#We will now create a dictionary that maps the Key Id with each sentence
#We will need to create a dictionary to keep track of the conversation
id2line={}

for line in lines:
    _line=line.split(' +++$+++ ')
    #We need to make sure that the line has 5 elements.
    if len(_line)==5:
        id2line[_line[0]]=_line[4]

#Creating a list of all the conversations
conversations_ids=[]
for conversation in conversations[:-1]:
    _conversation=conversation.split(' +++$+++ ')[-1]
    _conversation=_conversation[1:-1]#We removed the square brackets
    _conversation=_conversation.replace("'","")
    _conversation=_conversation.replace(" ","")
    conversations_ids.append(_conversation.split(","))
    
#We will get separately the quenstions and answere i.e. the inputs and the targets.
#last step before we start the cleaning process
#We want to get separate Huge list for question and answer each of same size.
#The list should be correctly assigned.
#in the conversation list , the 1st key is the question and the 2nd one is the answer
questions=[]
answers=[]

for conversations in conversations_ids:
    for conv in range(len(conversations)-1):
    
        questions.append(id2line[conversations[conv]])
        answers.append(id2line[conversations[conv+1]])

#Doing the first cleaning step

def clean_text(text):
    text=text.lower()
    text=re.sub(r"i'm","i am",text)
    text=re.sub(r"he's","he is",text)
    text=re.sub(r"she's","he is",text)
    #text=re.sub(r"don't","do not",text)
    text=re.sub(r"that's","that is",text)
    text=re.sub(r"what's","what is",text)
    text=re.sub(r"where's","where is",text)
    text=re.sub(r"\'ll"," will ",text)
    text=re.sub(r"\'ve"," have",text)
    text=re.sub(r"\'re"," are",text)
    text = re.sub(r"\'d", " would", text)
    text=re.sub(r"won't","will not",text)
    text=re.sub(r"can't","cannot",text)
    
    text=re.sub(r"[-()\*#/@;:<>{}+=~|.?,]","",text)
    return text

#Cleaning the Questions
clean_questions=[]
for text in questions:
    _cleand_text=clean_text(text)
    clean_questions.append(_cleand_text)

#Cleaning the Answers
clean_answers=[]
for text in answers:
    _cleand_text=clean_text(text)
    clean_answers.append(_cleand_text)

#Remove the not so frequent of the corpus
#Create a dictionary that maps each word with the number of occurences

word2count={}
#having the process of creating a dictionary for questions
for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word]=1
        else:
            word2count[word]+=1
            
            
#having the process of creating a dictionary for answers

for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word]=1
        else:
            word2count[word]+=1
            

#Creating 2 dictionaries that has the question words and the answer words to a unique number
threshold= 20
#map question words to a unique integer
questionswords2int={}
word_number=0
for word,count in word2count.items():
    if count>=threshold:
        questionswords2int[word] = word_number
        word_number+=1
        
word_number=0
answerswords2int={}
for word,count in word2count.items():
    if count>=threshold:
        answerswords2int[word] = word_number
        word_number+=1
        
#Adding the last tokens in the two dictionaries
tokens=["<PAD>","<EOS>","<OUT>", "<SOS>"]
for token in tokens:
    questionswords2int[token] = len(questionswords2int) + 1  

for token in tokens:
    answerswords2int[token] = len(answerswords2int) + 1
    
#We need inverse mapping of a dictionary
answersint2word={w_i: w for w, w_i in answerswords2int.items() }

#Adding EOS to the end of every answer. it is needed at the end of layer of the Seq2seq model.
for i in range(len(clean_answers)):
    clean_answers[i]+= " <EOS>"

#translate all the ques and answers into integers
#and replace all the words that were filtered out bu <OUT>

#list of words translated into integers
questions_to_int=[]
for question in clean_questions:
    ints= []
    for word in question.split():
        if word not in questionswords2int:
            ints.append(questionswords2int['<OUT>'])
        else:
            ints.append(questionswords2int[word])
    questions_to_int.append(ints)
    
answers_to_int=[]
for answer in clean_answers:
    ints= []
    for word in answer.split():
        if word not in answerswords2int:
            ints.append(answerswords2int['<OUT>'])
        else:
            ints.append(answerswords2int[word])
    answers_to_int.append(ints)

#Sorting questions and answers by length of questions
#We want to sort them as this will speed up the training and reduce the loss. THis will reduce the amt. of padding during the training
sorted_clean_questions=[]
sorted_clean_answers=[]
#Max length of question is chosen to be 25
for length in range(1,25+1):
    #We need the index and the question. We will use enbumerate
    for i in enumerate(questions_to_int):
        if len(i[1])==length:
            sorted_clean_questions.append(questions_to_int[i[0]])
            


for length in range(1,25+1):#Max length of question is chosen to be 25
    #We need the index and the question. We will use enbumerate
    for i in enumerate(answers_to_int):
        if len(i[1])==length:
            sorted_clean_answers.append(answers_to_int[i[0]])
            
            

##Building Seq2Seq Model####


def model_inputs():
    #3 params- type of data(input),dimensions of the matrix of the input data(None,None)-defining 2-d matrix,just name of the input)
   inputs=tf.placeholder(tf.int32,[None,None],name='input')
   targets=tf.placeholder(tf.int32,[None,None],name='target')
   
   #one will hold learning rate, and the one to hold the dropout rate(keepprob).
   lr=tf.placeholder(tf.float32,[None,None],name='learning_rate')
   keep_prob=tf.placeholder(tf.float32,[None,None],name='keep_prob')
   
   return inputs,targets,lr,keep_prob
   
            
#Before we start creating the encoding and decoding layer, we need to preprocess the targets
#because the decoder will only accept the certain format of the target
#The targetsneed to have a special format
#The targets must be in batches(2 batches)
#e.g= we will feed the Neural Network of size 10 at a time
#Each of the answer must start with the SOS token
#We also need to put the SOS token at the beginning of each of these answers
#We will do 2 things - Create batches and put an SOS token
         

def preprocess_targets(targets, word2int, batch_size):
    left_side = tf.fill([batch_size,1],word2int["<SOS>"])
    right_side= tf.strided_slice(target,[0,0],[batch_size,-1], [1,1])
    preprocessed_targets= tf.concat([left_side,right_side],axis=1)
    return preprocessed_targets
    

#We are now creating the encoding and decoding layer of the Neural Network

def encoder_rnn_layer(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
#rnn_inputs-inputs of the models...learning rate etc..and not the question
#rnn_size-No. of input tensors 
#num_layers- 
#keep_prob-dropout regularization
#sequence_length- list of the length of each question in the batch

    lstm=tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout=tf.contrib.rnn.DropoutWrapper(lstm,input_keep_prob=keep_prob)
    encoder_cell=tf.contrib.rnn.MultiRNNCell([lstm_dropout]*num_layers)#number of layers in LSTM
    _,encoder_state=tf.nn.bidirectional_dynamic_rnn(cell_fw= encoder_cell,cell_bw=encoder_cell,#We are using a bidirectional RNN 
                                                    sequence_length=sequence_length,
                                                    inputs=rnn_inputs,
                                                    dtype=tf.float32
                                                    )
    return encoder_state

#we will now create a decoder
#Decode the training Set

def decode_training_set(encoder_state,decoder_cell,decoder_embedded_input,sequence_length,decoding_scope,output_function,keep_prob,batch_size):
    #Embedding is just the mapping from a word to a vectors of real numbers eachone encoding uniquely the qord associated to it.
    #Decoding_scope- advanced DS that will wrap the tf variables, it will be like variable_scope
    
    attention_states= tf.zeros([batch_size,1,decoder_cell.output_size])
    attention_keys,attention_values,attention_score_function,attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states,attention_option='bahdanau',num_units=decoder_cell.output_size)
    
    #attention_keys=keys to be compared with target_states
    #attention_values= the values that are used to consdtruct the context vector
    #attention_score- sismilarity b/w the keys and the target states
    #attention_construct- used to build the attention states
    
    
    training_decoder_function=tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                            attention_keys,
                                                                            attention_values,
                                                                            attention_score_function,
                                                                            attention_construct_function,
                                                                            name= 'attn_dec_train')
        
    decoder_output,_,_=tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                              training_decoder_function,
                                                              decoder_embedded_input,
                                                              sequence_length,
                                                              decoding_scope)
    decoder_output_dropout=tf.nn.dropout(decoder_output,keep_prob)
    return output_function(decoder_output_dropout)
        #We got the attention Decoder function.
        #We get the decoder 
        


#Decoding the Test/Validation Set
def decode_test_set(encoder_states, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words, sequence_length, decoding_scope):
    # we wil use attention_decoder_inference function
    #Num-words- length of the answers
    #maximum_length- the max length of the question in the batch
    
    attention_states=tf.zeros([batch_size,1,decoder_output.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states,attention_option='bahdanau',num_units=decoder_cell.output_size)
    test_decoder_function=tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                            encoder_state[0],
                                                                            attention_keys,
                                                                            attention_values,     
                                                                            attention_score_function,
                                                                            attention_construct_function,
                                                                            decoder_embeddings_matrix, 
                                                                            sos_id, 
                                                                            eos_id, 
                                                                            maximum_length, 
                                                                            num_words,
                                                                            name= 'attn_dec_test')
    
    
    test_pred,_,_=tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                              test_decoder_function,
                                                        
                                                              decoding_scope)
    #decoder_output_dropout=tf.nn.dropout(decoder_output,keep_prob)
    return test_predictions



#Creating the Decoder RNN
def decoder_rnn(decoder_embedded_inputs, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
    with tf.variable_scope('decoding') as decoding_scope:
        lstm= tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout=tf.contrib.rnn.DropoutWrapper(lstm,input_keep_prob=keep_prob)
        decoder_cell=tf.contrib.rnn.MultiRNNCell([lstm_dropout]*num_layers)
        weights=tf.truncated_normal_initializer(stddev=0.1)#Truncated Normal Distribution of weights
        biases= tf.zeros_initializer()
        output_function = lambda x: tf.contrib.layers.fully_connected(x, num_words, 
                                                                      'relu',
                                                                      None,
                                                                      scope=decoding_scope,
                                                                      weights_initializers=weights,
                                                                      biases_initializer=biases,
                                                                      )#We are making fully connected layer
        training_predictions=decode_training_set(encoder_state,
                                                 decoder_cell,
                                                 decoder_embedded_input,
                                                 sequence_length,
                                                 decoing_scope,
                                                 output_function,
                                                 keep_prob,
                                                 batch_size)
        decoding_scope=reuse_variables()
        test_predictions=decode_test_set(encoder_state,
                                         decoder_cell,
                                         decoder_embeddings_matrix,
                                         word2int['<SOS>'],
                                         word2int['<EOS>'],
                                         sequence_length-1,
                                         num_words,
                                         decoding_scope,
                                         output_function,
                                         keep_prob,
                                         batch_size
                                         )
    return training_predictions,test_predictions


def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questionswords2int):
    encoder_embedded_input=tf.contrib.layers.embed_sequence(inputs, #We are getting the encoder_states
                                                            answers_num_words+1,
                                                            encoder_embedding_size,
                                                             initializer=tf.random_normal_initializer(0,1))#no. of dimensions in the encoder
    
    encoder_state= encoder_rnn_layer(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
    preprocessed_targets= preprocess_targets(targets, questionswords2int, batch_size) #To backprapogate the loss
    decoder_embeddings_matrix= tf.Variable(tf.random_uniform([questions_num_words+1,decoder_embedding_size], 0, 1))
    #decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words + 1, decoder_embedding_size], 0, 1))
    decoder_embedding_input= tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    
    
    training_predictions, test_predictions= decoder_rnn(decoder_embedded_input, 
                                                        decoder_embeddings_matrix, 
                                                        encoder_state, 
                                                        questions_num_words, 
                                                        sequence_length, 
                                                        rnn_size, 
                                                        num_layers, 
                                                        questionswords2int, 
                                                        keep_prob, 
                                                        batch_size)

####Training the Hyperparameters####
epochs=50 #getting the batches into NN and then Fwd propagating and then backward prapogation of loss into the neural network
batch_size=64
rnn_size=512
num_layers=3
encoding_embedding_size= 512 #512 cols in embedding matrix
decoding_embedding_size=512
learning_rate= 0.01
learning_rate_decay= 0.9
min_learning_rate= 0.0001
keep_probability= 0.50 #geofry Hinton recommends a 50% for hidden layers and 20% for input layers


#Defining a session
#When we open a tensorflow session for training, we need to reset the graph first.
tf.reset_default_graph()
session= tf.InteractiveSession()


#Load the model Inputs. 
inputs, targets, lr, keep_prob= model_inputs()

#Setting the sequence lenght to a max length i.e. 25
sequence_length=tf.placeholder_with_default(25, None, 'sequence_length')

#Set the input shape of input tensors
input_shape= tf.shape(inputs)

#Geting the training and Test predictions
training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs,[-1]), targets, keep_prob, batch_size, sequence_length,len(answerswords2int), len(questionswords2int), encoding_embedding_size, decoding_embedding_size, rnn_size, num_layers, questionswords2int)

#Setting the loss error, the optimizer and the gradient clipping
#2 elements- loss_error, and optimizer with gradient clipping

with tf.name_scope('optimization'):
    loss_error= tf.contrib.seq2seq.sequence_loss(training_predictions, targets, tf.ones([input_shape(0),sequence_length]))
    
    #Get adam 
    #get gradient clipping
    #Apply gradient clipping to optimizer
    
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients= optimizer.compute_gradients(loss_error)
    
    clipped_gradients= [(tf.clip_by_value(gradient_tensor, -5., 5.),gradient_variable) for gradient_tensor,grad_variable in gradients if gradient_tensor is not None]
    
    optimizer_gradient_clipping= optimizer.apply_gradients(clipped_gradients)
    

#We will apply the padding.
#what are we going to apply the padding?
#-All the sentences in the batch must have the same length
#What are we going to do exactly
#- Question and answers should have the same length
#Question- ['Who','are','you'<PAD>,<PAD>,<PAD>]
#answer-[<SOS>,'I','am','a','bot'<EOS>,<PAD>]
 
    
def apply_padding(batch_of_sequences,word2int):
    max_sequence_length= max([len(sequence) for sequences in batch_of_sequences ])
    return [sequence + [word2int['<PAD>']]* (max_sequence_length-len(sequence)) for sequence in batch_of_sequences]

#spliting the data into batches of questions and answers

def split_into_batches(questions, answers, batch_size):
    for batch_index in range(0,(len(questions)//batch_size)):
        start_index=batch_index * batch_size
        
        questions_in_batch= questions[start_index:start_index+batch_size]
        answers_in_batch=answers[start_index: start_index+ batch_size]
        
        padded_questions_in_batch=np.array(apply_padding(questions_in_batch,questionswords2int))
        padded_answers_in_batch=np.array(apply_padding(answers_in_batch,answerswords2int))
        
        yield padded_questions_in_batch,padded_answers_in_batch
        
        
        
#Spliting the questions and answers into training and validation Data
    
training_validation_split= int(len(sorted_clean_questions)*0.15)
training_questions=sorted_clean_questions[training_validation_split:]
training_answers=sorted_clean_answers[training_validation_split:]
validation_questions=sorted_clean_questions[:training_validation_split]
validation_answers=sorted_clean_answers[:training_validation_split]    

#training_loss
batch_index_check_training_loss=100
batch_index_check_validation_loss= ((len(training_questions))//batch_size //2) -1

total_training_loss_error=0 #Training losses

list_validation_loss_error= []

early_stopping_check=0 #each time we dont reduce the validation loss, the early_stopping_check is going to be incremented by 1

early_stopping_stop=1000

checkpoint= 'chatbot_weights.ckpt'#file containing the weights

session.run(tf.global_variables_initializer())
for epoch in range(1,epochs+1):
    for batch_index , (padded_questions_in_batch,padded_answers_in_batch) in enumerate(split_into_batches(training_questions, training_answers,batch_size)):
        starting_time=time.time()
        _, batch_training_loss_error= session.run([optimizer_gradient_clipping, loss_error],{inputs: padded_questions_in_batch, targets:padded_answers_in_batch,
                                                  lr:learning_rate,
                                                  sequence_length:padded_answers_in_batch.shape[1],
                                                  keep_prob:keep_probability})
    total_training_loss_error= batch_training_loss_error
    ending_time=time.time()
    batch_time=ending_time-starting_time
    
    if batch_index % batch_index_check_training_loss==0:
        print('Epoch : {:>3}//{}, Batch : {>4}/{}, Training Loss Error : {:>6.3f}, Training Time on 100 batches {:d} seconds'.format(epoch,
              epochs,
              batch_index,
              len(training_questions)//batch_size,
              total_training_loss_error/batch_index_check_training_loss,
              int(batch_time * batch_index_check_training_loss)))
        
        total_training_loss_error=0#We just used for the 100 batches
        
    if batch_index % batch_index_check_validation_loss==0 and batch_index>0:
        total_validation_loss_erorr=0
        starting_time= time.time()
        
        for batch_index_validation , (padded_questions_in_batch,padded_answers_in_batch) in enumerate(split_into_batches(validation_questions, validation_answers,batch_size)):
       
            batch_validation_loss_error= session.run(loss_error,{inputs: padded_questions_in_batch, targets:padded_answers_in_batch,
                                                  lr:learning_rate,
                                                  sequence_length:padded_answers_in_batch.shape[1],
                                                  })
            total_validation_loss_error= batch_validation_loss_error
        
            batch_time=ending_time-starting_time
    
        
        ending_time=time.time()
        batch_time=ending_time-starting_time
        average_validation_loss_error= total_validation_loss_error / len(validation_questions)
        print('Validation_loss_error: {:>6.3f} , batch Validation Time {:d} seconds'.format(average_validation_loss_error,
              int(batch_time)))
        
        learning_rate*= learning_rate_decay
        if learning_rate<min_learning_rate:
            learning_rate=min_learning_rate
        
        list_of_validation_loss_errors.append(average_validation_loss_error)
        if average_validation_loss_error<= min(list_validation_loss_error):
            print('I speak better now!')
            early_stopping_check = 0
            saver= tf.train.Saver()
            saver.save(session, checkpoint)
            
        else:
            print('Sorry, I do not speak Better!. I need to practice more!')
            early_stopping_check +=1
            if early_stopping_check == early_stopping_stop:
                break
    if early_stopping_check == early_stopping_stop:
        print('I cannot speak better anymore. This is the best I can do!')
        break
print('Game over!')







    
    
    
    