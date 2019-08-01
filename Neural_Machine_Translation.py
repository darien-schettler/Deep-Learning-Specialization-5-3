from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import numpy as np

from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date
from nmt_utils import *
import matplotlib.pyplot as plt

"""

NEURAL MACHINE TRANSLATION (NMT)

You will build a Neural Machine Translation (NMT) model
The NMT will translate human readable dates ("25th of June, 2009") into machine readable dates ("2009-06-25")

You will do this using an attention model, one of the most sophisticated sequence to sequence models

Let's load all the packages you will need for this assignment (DONE ABOVE)

"""

'''

DATASET

We will train the model on a dataset of 10000 human readable dates and their equivalent machine readable dates

Let's run the following cells to load the dataset and print some examples

'''

m = 10000
dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)
print("\n\nBelow are some examples of human & machine readable date pairs\n")
for j, i in enumerate(dataset[:10]): print(str(j+1) + ": " + str(i))

'''

You've loaded:

1. dataset             : a list of tuples of (human readable date, machine readable date)

2. human_vocab         : a dictionary mapping all characters used in the human readable dates to an integer-valued index

3. machine_vocab       : a dictionary mapping all characters used in machine readable dates to an integer-valued index
                            --> These indices are not necessarily consistent with human_vocab
                        
4. inv_machine_vocab   : the inverse dictionary of machine_vocab, mapping from indices back to characters


Let's pre-process the data and map the raw text data into the index values
We will also use Tx = 30 and Ty = 10
        --> Assume Tx = 30 is the maximum length of the human readable date (if we get a longer input, we truncate it)
        --> Assume Ty = 10 is the maximum output length (since "YYYY-MM-DD" is 10 characters long)

'''

Tx = 30
Ty = 10
X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)

print("\n\nX.shape:", X.shape)
print("Y.shape:", Y.shape)
print("Xoh.shape:", Xoh.shape)
print("Yoh.shape:", Yoh.shape)

'''

You now have:

 X:    A processed version of the human readable dates in the training set
          --> Each character is replaced by an index mapped to the character via human_vocab
          --> Each date is further padded to  Tx  values with a special character (< pad >)
          --> X.shape = (m, Tx)

 Y:    A processed version of the machine readable dates in the training set
          --> Each character is replaced by the index it is mapped to in machine_vocab
          --> Y.shape = (m, Ty)

 Xoh:  One-hot version of X
          --> The "1" entry's index is mapped to the character thanks to human_vocab
          --> Xoh.shape = (m, Tx, len(human_vocab))

 Yoh:  One-hot version of Y
          --> The "1" entry's index is mapped to the character thanks to machine_vocab
          --> Yoh.shape = (m, Tx, len(machine_vocab))
                -- Here, len(machine_vocab) = 11 since there are 11 characters ('-' as well as 0-9)

'''

print("\nHere are examples of preprocessed training examples. Change the index in the code to see other examples")
print("========================================================================================================")

index = 0

print("--------------------------------------------------------------------------------------------------------\n")
print("Source date:", dataset[index][0])
print("Target date:", dataset[index][1])

print()
print("--------------------------------------------------------------------------------------------------------")
print()

print("Source after preprocessing (indices):\n", X[index])
print("\nTarget after preprocessing (indices):\n", Y[index])

print()
print("--------------------------------------------------------------------------------------------------------")
print()

print("Source after preprocessing (one-hot):\n", Xoh[index])
print("\nTarget after preprocessing (one-hot):\n", Yoh[index])

print("\n--------------------------------------------------------------------------------------------------------")
print("========================================================================================================\n")

'''

NEURAL MACHINE TRANSLATION WITH ATTENTION

If you had to translate a book's paragraph from French to English how would you approach it?
You would not read the whole paragraph, then close the book and attempt to then translate

It would be better to, read/re-read and focus on the parts of the French paragraph corresponding to the parts...
 ... of the English you are attempting to translate at a given time

The attention mechanism tells a Neural Machine Translation model where it should pay attention to at any step


ATTENTION MECHANISM

In this part, you will implement the attention mechanism presented in the lecture videos

------------------------------------------------------------
 Here are some properties of the model that you may notice:
------------------------------------------------------------

There are two separate LSTMs in this model
--> One is a Bi-directional LSTM and comes before the attention mechanism, we will call it pre-attention Bi-LSTM
--> One is an LSTM and comes after the attention mechanism, so we will call it the post-attention LSTM

The pre-attention Bi-LSTM goes through  Tx  time steps
The post-attention LSTM goes through  Ty  time steps

The post-attention LSTM passes  s⟨t⟩,c⟨t⟩  from one time step to the next

In the lecture videos, we were using only a basic RNN for the post-activation sequence model
    --> i.e. The state captured by the RNN output activations  s⟨t⟩
Since we are using an LSTM here, the LSTM has both the output activation  s⟨t⟩  and the hidden cell state  c⟨t⟩
However, in this model the post-activation LSTM at time  t   wont take the specific generated  y⟨t−1⟩  as input
    --> It only takes  s⟨t⟩  and  c⟨t⟩  as input
    --> We have designed the model this way, because...
        -- Unlike language generation where adjacent characters are highly correlated, there isn't as strong a ... 
            ... dependency between the previous character and the next character in a YYYY-MM-DD date

To represent the concatenation of the activations of both the forward-dir and backward-dir of the pre-attention Bi-LSTM:
        -->     We use:       a⟨t⟩ = [a→⟨t⟩; a←⟨t⟩]

We use a RepeatVector node to copy  s⟨t−1⟩'s value  Tx  times
We then use Concatenation to concatenate  s⟨t−1⟩  and  a⟨t⟩  to compute  e⟨t,t′)
This is then passed through a softmax to compute  α⟨t,t′⟩
            
             <<<< We'll explain how to use RepeatVector and Concatenation in Keras below >>>>

Lets implement this model
You will start by implementing two functions: one_step_attention() and model()

--------------------------------------------------------------------------------------------------------------------

1. one_step_attention(): 

--------------------------------------------------------------------------------------------------------------------
    --> At step  t, given all the hidden states of the Bi-LSTM ( [a<1>,...,a<Tx>] )
    --> At step  t, given the previous hidden state of the second LSTM ( s<t−1> )
    --> one_step_attention() will:
        -- Compute the attention weights ( [α<t,1>,α<t,2>,...,α<t,Tx>] )
        -- Output the context vector:

                                    context<t>= ∑{α<t,t′> * a<t′>}
            SUM IS --> from t′=0 to Tx
            
[[[ Note ]]] 
We are denoting the attention in this notebook  context⟨t⟩ 
    --> This is done to avoid confusion with the (post-attention) LSTM's internal memory cell variable (often c<t>)

--------------------------------------------------------------------------------------------------------------------

2) model():

--------------------------------------------------------------------------------------------------------------------

    --> Implements the entire model
    --> It first runs the input through a Bi-LSTM to get back  [a<1>,a<2>,...,a<Tx>]
    --> Then, it calls one_step_attention()  Ty  times (for loop)
        --> Each iteration it gives the computed context vector  c<t> to the second LSTM
        --> Each iteration it runs the output of the LSTM through a dense layer with softmax activation to generate ŷ<t>
        
--------------------------------------------------------------------------------------------------------------------

Exercise:   Implement one_step_attention()
            The function model() will call the layers in one_step_attention()  Ty  using a for-loop
                --> It is important that all  Ty  copies have the same weights
                    --> i.e., it should not re-initiaiize the weights every time
                    --> In other words, all  Ty  steps should have shared weights
                    --> Here's how you can implement layers with shareable weights in Keras:

                                1. Define the layer objects (as global variables for examples)
                                2. Call these objects when propagating the input.

We have defined the layers you need as global variables
Please run the following cells to create them

Please check the Keras documentation to make sure you understand what these layers are: 
        
        RepeatVector()
        Concatenate()
        Dense()
        Activation()
        Dot()

'''

# Defined shared layers as global variables
repeator = RepeatVector(Tx)
concatenator = Concatenate(axis=-1)
densor1 = Dense(10, activation = "tanh")
densor2 = Dense(1, activation = "relu")
activator = Activation(softmax, name='attention_weights') # We are using a custom softmax(axis = 1) loaded in this notebook
dotor = Dot(axes = 1)


def one_step_attention(a, s_prev):
    """
    Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
    "alphas" and the hidden states "a" of the Bi-LSTM.

    Arguments:
    a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
    s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)

    Returns:
    context -- context vector, input of the next (post-attetion) LSTM cell
    """

    # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states
    #  "a" (≈ 1 line)
    s_prev = repeator(s_prev)

    # Use concatenator to concatenate a and s_prev on the last axis (≈ 1 line)
    concat = concatenator([a, s_prev])

    # Use densor1 to propagate concat through a small fully-connected neural network to compute the "intermediate
    # energies" variable e. (≈1 lines)
    e = densor1(concat)

    # Use densor2 to propagate e through a small fully-connected neural network to compute the "energies" variable
    # energies. (≈1 lines)
    energies = densor2(e)

    # Use "activator" on "energies" to compute the attention weights "alphas" (≈ 1 line)
    alphas = activator(energies)

    # Use dotor together with "alphas" and "a" to compute the context vector to be given to the next (post-attention)
    # LSTM-cell
    context = dotor([alphas, a])

    return context


n_a = 32
n_s = 64
post_activation_LSTM_cell = LSTM(n_s, return_state = True)
output_layer = Dense(len(machine_vocab), activation=softmax)

'''

Now you can use these layers  Ty  times in a for loop to generate the outputs, their parameters will not be reinit.

You will have to carry out the following steps:

1. Propagate the input into a Bidirectional LSTM

2. Iterate for  t=0,…,Ty−1 :
    A) Call one_step_attention() on  [α<t,1>,α<t,2>,...,α<t,Tx>]  and  s<t−1>  mto get the context vector  context<t>
    
    B) Give  context<t>  to the post-attention LSTM cell
        --> Remember to pass in the previous hidden-state  s⟨t−1⟩  and cell-states  c⟨t−1⟩  of this LSTM using...
            -- initial_state = [previous hidden state, previous cell state]
        --> Get back the new hidden state  s<t>  and the new cell state  c<t>
    
    C) Apply a softmax layer to  s<t>, get the output
    
    D) Save the output by adding it to the list of outputs
    
3. Create your Keras model instance:
    -- It should have three inputs: "inputs",  s<0>  and  c<0>
    -- It should output the list of "outputs"

'''


def model(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
    """
    Arguments:
    Tx -- length of the input sequence
    Ty -- length of the output sequence
    n_a -- hidden state size of the Bi-LSTM
    n_s -- hidden state size of the post-attention LSTM
    human_vocab_size -- size of the python dictionary "human_vocab"
    machine_vocab_size -- size of the python dictionary "machine_vocab"

    Returns:
    model -- Keras model instance
    """

    # Define the inputs of your model with a shape (Tx,)
    # Define s0 and c0, initial hidden state for the decoder LSTM of shape (n_s,)
    X = Input(shape=(Tx, human_vocab_size))
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0

    # Initialize empty list of outputs
    outputs = []

    # Step 1: Define your pre-attention Bi-LSTM. Remember to use return_sequences=True. (≈ 1 line)
    a = Bidirectional(LSTM(n_a, return_sequences=True), input_shape=(m, Tx, n_a * 2))(X)

    # Step 2: Iterate for Ty steps
    for t in range(Ty):
        # Step 2.A: Perform one step of the attention mechanism to get back the context vector at step t (≈ 1 line)
        context = one_step_attention(a, s)

        # Step 2.B: Apply the post-attention LSTM cell to the "context" vector.
        # Don't forget to pass: initial_state = [hidden state, cell state] (≈ 1 line)
        s, _, c = post_activation_LSTM_cell(context, initial_state=[s, c])

        # Step 2.C: Apply Dense layer to the hidden state output of the post-attention LSTM (≈ 1 line)
        out = output_layer(s)

        # Step 2.D: Append "out" to the "outputs" list (≈ 1 line)
        outputs.append(out)

    # Step 3: Create model instance taking three inputs and returning the list of outputs. (≈ 1 line)
    model = Model([X, s0, c0], outputs=outputs)

    return model


model = model(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))
model.summary()

opt = Adam(lr = 0.005, beta_1 = 0.9, beta_2 = 0.999,decay = 0.01)
model.compile(loss = 'categorical_crossentropy', optimizer = opt,metrics = ['accuracy'])

'''

The last step is to define all your inputs and outputs to fit the model:

You already have X of shape  (m=10000,Tx=30)  containing the training examples
You need to create s0 and c0 to initialize your post_activation_LSTM_cell with 0s
Given the model() you coded, you need the "outputs" to be a list of 11 elements of shape (m, T_y)
    --> i.e. outputs[i][0], ..., outputs[i][Ty] represent the true labels corresponding to the  ith  training ex (X[i])
    --> More generally, outputs[i][j] is the true label of the  jth  character in the  ith  training example

'''

s0 = np.zeros((m, n_s))
c0 = np.zeros((m, n_s))
outputs = list(Yoh.swapaxes(0,1))

model.fit([Xoh, s0, c0], outputs, epochs=1, batch_size=100)

'''

We have run this model for longer, and saved the weights. Run the next cell to load our weights
---> By training a model for several minutes, you should be able to obtain a model of similar accuracy

'''

model.load_weights('models/model.h5')

'''

Let's see some examples...

'''

EXAMPLES = ['3 May 1979', '5 April 09', '21th of August 2016', 'Tue 10 Jul 2007', 'Saturday May 9 2018', 'March 3 2001',
            'March 3rd 2001', '1 March 2001']

for example in EXAMPLES:
    source = string_to_int(example, Tx, human_vocab)
    source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source))).swapaxes(0, 1)
    prediction = model.predict([source, s0, c0])
    prediction = np.argmax(prediction, axis=-1)
    output = [inv_machine_vocab[int(i)] for i in prediction]

    print("source:", example)
    print("output:", ''.join(output))

'''

ALL DONE!

Here's what you should remember from this notebook:

1. Machine translation models can be used to map from one sequence to another
    ---> They are useful not just for translating human languages (like French -> English) but other tasks (date recog.)
2. An attn mech. allows a network to focus on the most relevant parts of the input when producing a part of the output
3. A network using an attn mechanism can trans from inputs of length  Tx  to outputs of length  Ty, (Ty  == or !=  Tx)
4. You can visualize attn weights  α⟨t,t′⟩  to see what the network is paying attn to while generating each output

'''