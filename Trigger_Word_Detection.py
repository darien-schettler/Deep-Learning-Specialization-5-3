import numpy as np
from pydub import AudioSegment
import random
import sys
import io
import os
import glob
import IPython
from td_utils import *
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam

'''

What really is an audio recording? 
A microphone records little variations in air pressure over time
It is these little variations in air pressure that your ear also perceives as sound

You can think of an audio recording as a long list of numbers measuring the detected air pressure changes

We will use audio sampled at 44100 Hz (or 44100 Hertz)
    --> This means the microphone gives us 44100 numbers per second
    --> Thus, a 10 second audio clip is represented by 441000 numbers (  =  10×44100  )

It is quite difficult to figure out from this "raw" representation of audio whether the word "activate" was said
In order to help the sequence model more easily learn to detect triggerwords, we will compute a spectrogram of the audio

The spectrogram tells us how much of discrete different frequencies are present in an audio clip at a moment in time

A spectrogram is computed by sliding a window over the raw audio signal
It then calculates the most active frequencies in each window using a Fourier transform

'''

x = graph_spectrogram("audio_examples/example_train.wav")

'''

Spectrograms use colour to show the degree to which different freq are present in the audio at different points in time
    --> Green squares means a certain frequency is more active or more present in the audio clip (louder)
    --> Blue squares denote less active frequencies

The dimension of the output spectrogram depends upon:
    --> The hyperparameters of the spectrogram software
    --> The length of the input
    
In this program, we will be working with 10 second audio clips as the "standard length" for our training examples
The number of timesteps of the spectrogram will be 5511 (5511 discrete readings)

You'll see later that the spectrogram will be the input  x  into the network, and so  Tx=5511

'''

_, data = wavfile.read("audio_examples/example_train.wav")

print("Time steps in audio recording before spectrogram", data[:, 0].shape)
print("Time steps in input after spectrogram", x.shape)

Tx = 5511  # The number of time steps input to the model from the spectrogram
n_freq = 101  # Number of frequencies input to the model at each time step of the spectrogram

Ty = 1375  # The number of time steps in the output of our model
'''

--- Note --- 
Even with 10 seconds being our default training example, 10 seconds of time can be discretized differently (dif. values)
    --> You've seen 441000 (raw audio) and 5511 (spectrogram)
        --> In the former case, each step represents  10/441000≈0.000023  seconds
        --> In the second case, each step represents  10/5511≈0.0018  seconds.

For the 10sec of audio, the key values you will see in this program are:

441000      (raw audio)
5511 = Tx   (spectrogram output, and dimension of input to the neural network).
10000       (used by the pydub module to synthesize audio)
1375 = Ty   (the number of steps in the output of the GRU you'll build).

Note that each of these representations correspond to exactly 10 seconds of time
It's just that they are discretizing them to different degrees

All of these are hyperparameters and can be changed (except the 441000, which is a function of the microphone)

We have chosen values that are within the standard ranges uses for speech systems

Consider the  Ty=1375  number above
    --> This means that for the output of the model, we discretize the 10s into 1375 time-intervals ...
        ... (each one of length  10/1375≈0.0072 s) ...
        ... and try to predict for each of these intervals whether someone recently finished saying "activate."

Consider also the 10000 number above
    --> This corresponds to discretizing the 10sec clip into 10/10000 = 0.001 second itervals
        --> 0.001 seconds is also called 1 millisecond, or 1ms
        --> So when we say we are discretizing according to 1ms intervals, it means we are using 10,000 steps


1.3 - Generating a single training example

Because speech data is hard to acquire and label, you will synthesize your training data using the audio clips of:
    1. activates
    2. negatives
    3. backgrounds
    
It is quite slow to record lots of 10 second audio clips with random "activates" in it

Instead, it is easier to record lots of positives and negative words, and record background noise separately 
    --> (or download background noise from free online sources)

To synthesize a single training example, you will:

Pick a random 10 second background audio clip
Randomly insert 0-4 audio clips of "activate" into this 10sec clip
Randomly insert 0-2 audio clips of negative words into this 10sec clip
Because you had synthesized the word "activate" into the background clip, you know exactly when in the 10sec clip the "activate" makes its appearance

You'll see later that this makes it easier to generate the labels  y⟨t⟩  as well

You will use the pydub package to manipulate audio
    -- Pydub converts raw audio files into lists of Pydub data structures (it is not important to know the details here)
    -- Pydub uses 1ms as the discretization interval which is why a 10sec clip is always represented using 10,000 steps

'''

# Load audio segments using pydub
# activates, negatives, backgrounds = load_raw_audio()

# print("background len: " + str(len(backgrounds[0])))    # Should be 10,000, since it is a 10 sec clip
# print("activate[0] len: " + str(len(activates[0])))     # Maybe around 1000, since an "activate" audio clip is apx 1 sec
# print("activate[1] len: " + str(len(activates[1])))     # Different "activate" clips can have different lengths

'''

Overlaying positive/negative words on the background:

1. 
- Given a 10sec background clip and a short audio clip (positive or negative word) you need to be able to ... 
  ... "add" or "insert" the word's short audio clip onto the background
- To ensure audio segments inserted onto the background do not overlap, you will keep track of the times of ...
  ... previously inserted audio clips
- You will be inserting multiple clips of positive/negative words onto the background, and you don't want to ...
  ... insert an "activate" or a random word somewhere that overlaps with another clip you had previously added

2. 
- For clarity, when you insert a 1sec "activate" onto a 10sec clip of cafe noise, you end up with a ...
   ... 10sec clip that sounds like someone saying "activate" in a cafe, with "activate" superimposed over cafe noise
- You do not end up with an 11 sec clip (You'll see later how pydub allows you to do this)

Creating the labels at the same time you overlay:

1. 
Recall also that the labels  y⟨t⟩  represent whether or not someone has just finished saying "activate."
    --> Given a background clip, we can initialize  y⟨t⟩=0  for all  t , since the clip doesn't contain any "activates."
    --> When you insert or overlay an "activate" clip, you will also update labels for  y⟨t⟩
        -- 50 steps of the output now have target label 1
    --> You will train a GRU to detect when someone has finished saying "activate". 

For example, suppose the synthesized "activate" clip ends at the 5sec mark in the 10sec audio
    --> Recall that  Ty=1375 , so timestep  687 = int(1375*0.5) corresponds to the moment at 5sec into the audio
    --> So, you will set  y⟨688⟩ = 1
    --> Further, you would be quite satisfied if the GRU detects "activate" anywhere near (after) this moment
        -- Therefore we actually set 50 consecutive values of the label  y⟨t⟩  to 1
        -- Specifically, we have  y⟨689⟩ = ⋯ = y⟨737⟩ = 1

This is another reason for synthesizing the training data:
    --> It's relatively straightforward to generate these labels  y⟨t⟩  as described above
    --> In contrast, if you have 10sec of audio recorded on a mic, it's time consuming for a person to do it manually


To implement the training set synthesis process, you will use the following helper functions:
    -- All of these function will use a 1ms interval, so the 10sec of audio is alwsys discretized into 10,000 steps

                    1. get_random_time_segment(segment_ms) 
                            --> gets a random time segment in our background audio
                
                    2. is_overlapping(segment_time, existing_segments)
                            --> checks if a time segment overlaps with existing segments
                
                    3. insert_audio_clip(background, audio_clip, existing_times) 
                            --> inserts an audio segment at a random time in our background audio using other functions
                    4. insert_ones(y, segment_end_ms)
                            --> inserts 1's into our label vector y after the word "activate"

'''

'''

The function get_random_time_segment(segment_ms) returns a random time segment
    --> We can insert an audio clip of duration segment_ms at this random time segment
    
Read through the code to make sure you understand what it is doing

'''


def get_random_time_segment(segment_ms):
    """
    Gets a random time segment of duration segment_ms in a 10,000 ms audio clip.

    Arguments:
    segment_ms -- the duration of the audio clip in ms ("ms" stands for "milliseconds")

    Returns:
    segment_time -- a tuple of (segment_start, segment_end) in ms
    """

    segment_start = np.random.randint(low=0, high=10000 - segment_ms)  # Make sure segment doesn't run past 10sec backg.
    segment_end = segment_start + segment_ms - 1

    return (segment_start, segment_end)


'''

Next, suppose you have inserted audio clips at segments (1000,1800) and (3400,4500)
    i.e., the first segment starts at step 1000, and ends at step 1800
    
Now, if we're considering inserting a new audio clip at (3000,3600) does this overlap with previously inserted segments?
    --> In this case, (3000,3600) and (3400,4500) overlap, so we should decide against inserting a clip here

For the purpose of this function, define (100,200) and (200,250) to be overlapping, since they overlap at timestep 200
    --> However, (100,199) and (200,250) are non-overlapping.


Exercise: Implement is_overlapping(segment_time, existing_segments) 
    --> It will check if a new time segment overlaps with any of the previous segments
    --> You will need to carry out 2 steps:

            1. Create a "False" flag, that you will later set to "True" if you find that there is an overlap
            2. Loop over the previous_segments' start and end times
                --> Compare these times to the segment's start and end times
                --> If there is an overlap, set the flag defined in (1) as True. 
                
                You can use:
                        for ....:
                             if ... <= ... and ... >= ...:
                                ...


'''


def is_overlapping(segment_time, previous_segments):
    """
    Checks if the time of a segment overlaps with the times of existing segments.

    Arguments:
    segment_time -- a tuple of (segment_start, segment_end) for the new segment
    previous_segments -- a list of tuples of (segment_start, segment_end) for the existing segments

    Returns:
    True if the time segment overlaps with any of the existing segments, False otherwise
    """

    segment_start, segment_end = segment_time

    # Step 1: Initialize overlap as a "False" flag. (≈ 1 line)
    overlap = False

    # Step 2: loop over the previous_segments start and end times.
    # Compare start/end times and set the flag to True if there is an overlap (≈ 3 lines)
    for previous_start, previous_end in previous_segments:
        if segment_start <= (previous_end or previous_start) <= segment_end:
            overlap = True

    return overlap


'''

Now, lets use the previous helper functions to insert a new audio clip onto the 10sec background at a random time
    --> but we make sure that any newly inserted segments won't overlap with the previous segments


Exercise: Implement insert_audio_clip() to overlay an audio clip onto the background 10sec clip

You will need to carry out 4 steps:

1. Get a random time segment of the right duration in ms

2. Make sure that the time segment does not overlap with any of the previous time segments
    --> If it is overlapping, then go back to step 1 and pick a new time segment

3. Add the new time segment to the list of existing time segments to keep track of all the segments you've inserted

4. Overlay the audio clip over the background using pydub

'''


def insert_audio_clip(background, audio_clip, previous_segments):
    """
    Insert a new audio segment over the background noise at a random time step, ensuring that the
    audio segment does not overlap with existing segments.

    Arguments:
    background -- a 10 second background audio recording.
    audio_clip -- the audio clip to be inserted/overlaid.
    previous_segments -- times where audio segments have already been placed

    Returns:
    new_background -- the updated background audio
    """

    # Get the duration of the audio clip in ms
    segment_ms = len(audio_clip)

    # Step 1: Use one of the helper functions to pick a random time segment onto which to insert the new audio clip
    segment_time = get_random_time_segment(segment_ms)

    # Step 2: Check if the new segment_time overlaps with one of the previous_segments
    # If so, keep picking new segment_time at random until it doesn't overlap
    while is_overlapping(segment_time, previous_segments):
        segment_time = get_random_time_segment(segment_ms)

    # Step 3: Add the new segment_time to the list of previous_segments (≈ 1 line)
    previous_segments.append(segment_time)

    # Step 4: Superpose audio segment and background
    new_background = background.overlay(audio_clip, position=segment_time[0])

    return new_background, segment_time


# np.random.seed(5)
# audio_clip, segment_time = insert_audio_clip(backgrounds[0], activates[0], [(3790, 4400)])
# audio_clip.export("insert_test.wav", format="wav")
# print("Segment Time: ", segment_time)

'''

Finally, we implement code to update the labels  y⟨t⟩ , assuming you just inserted an "activate." 

In the code below, y is a (1,1375) dimensional vector, since  Ty=1375

If the "activate" ended at time step  t , then set  y⟨t+1⟩ = 1  as well as for up to 49 additional consecutive values
However, make sure you don't run off the end of the array and try to update y[0][1375]
    --> Since the valid indices are y[0][0] through y[0][1374] because  Ty=1375
    --> So if "activate" ends at step 1370, you would get only y[0][1371] = y[0][1372] = y[0][1373] = y[0][1374] = 1


Exercise: Implement insert_ones()
    
    Use the formula:
    
            segment_end_y = int(segment_end_ms * Ty / 10000.0)

'''


def insert_ones(y, segment_end_ms):
    """
    Update the label vector y. The labels of the 50 output steps strictly after the end of the segment
    should be set to 1. By strictly we mean that the label of segment_end_y should be 0 while, the
    50 following labels should be ones.


    Arguments:
    y -- numpy array of shape (1, Ty), the labels of the training example
    segment_end_ms -- the end time of the segment in ms

    Returns:
    y -- updated labels
    """

    # duration of the background (in terms of spectrogram time-steps)
    segment_end_y = int(segment_end_ms * Ty / 10000.0)

    # Add 1 to the correct index in the background label (y)

    for i in range(segment_end_y + 1, segment_end_y + 51):
        if i < Ty:
            y[0, i] = 1.0

    return y


# arr1 = insert_ones(np.zeros((1, Ty)), 9700)
# plt.plot(insert_ones(arr1, 4251)[0,:])
# plt.show()
# print("sanity checks:", arr1[0][1333], arr1[0][634], arr1[0][635])

'''

Finally, you can use insert_audio_clip and insert_ones to create a new training example


Exercise: Implement create_training_example(). You will need to carry out the following steps:

1. Initialize the label vector  y  as a numpy array of zeros and shape  (1,Ty)

2. Initialize the set of existing segments to an empty list

3. Randomly select 0 to 4 "activate" audio clips, and insert them onto the 10sec clip
    --> Also insert labels at the correct position in the label vector  y
    
4. Randomly select 0 to 2 negative audio clips, and insert them into the 10sec clip

'''


def create_training_example(background, activates, negatives):
    """
    Creates a training example with a given background, activates, and negatives.

    Arguments:
    background -- a 10 second background audio recording
    activates -- a list of audio segments of the word "activate"
    negatives -- a list of audio segments of random words that are not "activate"

    Returns:
    x -- the spectrogram of the training example
    y -- the label at each time step of the spectrogram
    """

    # Set the random seed
    np.random.seed(18)

    # Make background quieter
    background = background - 20

    # Step 1: Initialize y (label vector) of zeros (≈ 1 line)
    y = np.zeros((1, Ty))

    # Step 2: Initialize segment times as empty list (≈ 1 line)
    previous_segments = []

    # Select 0-4 random "activate" audio clips from the entire list of "activates" recordings
    number_of_activates = np.random.randint(0, 5)
    random_indices = np.random.randint(len(activates), size=number_of_activates)
    random_activates = [activates[i] for i in random_indices]

    # Step 3: Loop over randomly selected "activate" clips and insert in background
    for random_activate in random_activates:
        # Insert the audio clip on the background
        background, segment_time = insert_audio_clip(background, random_activate, previous_segments)
        # Retrieve segment_start and segment_end from segment_time
        segment_start, segment_end = segment_time
        # Insert labels in "y"
        y = insert_ones(y, segment_end)

    # Select 0-2 random negatives audio recordings from the entire list of "negatives" recordings
    number_of_negatives = np.random.randint(0, 3)
    random_indices = np.random.randint(len(negatives), size=number_of_negatives)
    random_negatives = [negatives[i] for i in random_indices]

    # Step 4: Loop over randomly selected negative clips and insert in background
    for random_negative in random_negatives:
        # Insert the audio clip on the background
        background, _ = insert_audio_clip(background, random_negative, previous_segments)

    # Standardize the volume of the audio clip
    background = match_target_amplitude(background, -20.0)

    # Export new training example
    file_handle = background.export("train" + ".wav", format="wav")
    print("File (train.wav) was saved in your directory.")

    # Get and plot spectrogram of the new recording (background with superposition of positive and negatives)
    x = graph_spectrogram("train.wav")

    return x, y


# x, y = create_training_example(backgrounds[1], activates, negatives)

# plt.plot(y[0])
# plt.show()

# ----------------------
# LOAD FULL TRAINING SET
# ----------------------

# Load preprocessed training examples
X = np.load("./XY_train/X.npy")
Y = np.load("./XY_train/Y.npy")

# ----------------------

# ----------------------
# LOAD FULL DEV SET
# ----------------------

# Load preprocessed dev set examples (REAL NOT SYNTHESIZED AUDIO)
X_dev = np.load("./XY_dev/X_dev.npy")
Y_dev = np.load("./XY_dev/Y_dev.npy")

# ----------------------

# ----------------------
# MAKE THE MODEL (KERAS)
# ----------------------
# Note the architecture can be viewed in the images folder as model.png

'''

One key step of this model is the 1D convolutional step (near the bottom of Figure 3). It inputs the 5511 step 
spectrogram, and outputs a 1375 step output, which is then further processed by multiple layers to get the final $T_y 
= 1375$ step output. This layer plays a role similar to the 2D convolutions you saw in Course 4, of extracting 
low-level features and then possibly generating an output of a smaller dimension. 

Computationally, the 1-D conv layer also helps speed up the model because now the GRU  has to process only 1375 
timesteps rather than 5511 timesteps. The two GRU layers read the sequence of inputs from left to right, 
then ultimately uses a dense+sigmoid layer to make a prediction for $y^{\langle t \rangle}$. Because $y$ is binary 
valued (0 or 1), we use a sigmoid output at the last layer to estimate the chance of the output being 1, 
corresponding to the user having just said "activate." 

Note that we use a uni-directional RNN rather than a bi-directional RNN. This is really important for trigger word 
detection, since we want to be able to detect the trigger word almost immediately after it is said. If we used a 
bi-directional RNN, we would have to wait for the whole 10sec of audio to be recorded before we could tell if 
"activate" was said in the first second of the audio clip 

'''

'''

Implementing the model can be done in four steps:

Step 1: CONV layer

    --> Use Conv1D() to implement this, with 196 filters, a filter size of 15 (kernel_size=15), and stride of 4

Step 2: First GRU layer

    --> To generate the GRU layer, use:
                -- X = GRU(units = 128, return_sequences = True)(X)
                -- Setting return_sequences = True ensures that all the GRU's hidden states are fed to the next layer
                        - Remember to follow this with Dropout and BatchNorm layers

Step 3: Second GRU layer
    
    --> This is similar to previous GRU layer (remember to use return_sequences = True), but has an extra dropout layer

Step 4: Create a time-distributed dense layer as follows:

    --> X = TimeDistributed(Dense(1, activation = "sigmoid"))(X)
            -- This creates a dense layer followed by a sigmoid
                - The parameters used for the dense layer are the same for every time step


Exercise: Implement model()

'''


# GRADED FUNCTION: model

def model(input_shape):
    """
    Function creating the model's graph in Keras.

    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)

    Returns:
    model -- Keras model instance
    """

    X_input = Input(shape=input_shape)

    # Step 1: CONV layer
    X = Conv1D(filters=196, kernel_size=15, strides=4)(X_input)  # CONV1D
    X = BatchNormalization()(X)  # Batch normalization
    X = Activation("relu")(X)  # ReLu activation
    X = Dropout(0.8)(X)  # dropout (use 0.8)

    # Step 2: First GRU Layer
    X = GRU(units=128, return_sequences=True)(X)  # GRU (use 128 units and return the sequences)
    X = Dropout(0.8)(X)  # dropout (use 0.8)
    X = BatchNormalization()(X)  # Batch normalization

    # Step 3: Second GRU Layer
    X = GRU(units=128, return_sequences=True)(X)
    X = Dropout(0.8)(X)  # dropout (use 0.8)
    X = BatchNormalization()(X)  # Batch normalization
    X = Dropout(0.8)(X)  # dropout (use 0.8)

    # Step 4: Time-distributed dense layer
    X = TimeDistributed(Dense(1, activation="sigmoid"))(X)  # time distributed  (sigmoid)

    model = Model(inputs=X_input, outputs=X)

    return model

    # ----------------------


model = model(input_shape=(Tx, n_freq))

model.summary()

# ----------------------
# FIT THE MODEL
# ----------------------

# Pre-partially trained model
model = load_model('./models/tr_model.h5')
