# Tibetan-Speech-recognition
This is web based speech recognition with use of machine learning algorithm(CNN) and achieve accuracy of 96%


Speech Emotion Recognition with Convolutional Neural Network
Recognizing Human Emotion from Audio Recording
Go to the profile of Reza Chu
Reza Chu
Jun 1

Image Credit: B-rina
Recognizing human emotion has always been a fascinating task for data scientists. Lately, I am working on an experimental Speech Emotion Recognition (SER) project to explore its potential. I selected the most starred SER repository from GitHub to be the backbone of my project.

Before we walk through the project, it is good to know the major bottleneck of Speech Emotion Recognition.

Major Obstacles:
Emotions are subjective, people would interpret it differently. It is hard to define the notion of emotions.
Annotating an audio recording is challenging. Should we label a single word, sentence or a whole conversation? How many emotions should we define to recognize?
Collecting data is complex. There are lots of audio data can be achieved from films or news. However, both of them are biased since news reporting has to be neutral and actors’ emotions are imitated. It is hard to look for neutral audio recording without any bias.
Labeling data require high human and time cost. Unlike drawing a bounding box on an image, it requires trained personnel to listen to the whole audio recording, analysis it and give an annotation. The annotation result has to be evaluated by multiple individuals due to its subjectivity.

Image Credit: depositphotos
Project Description:
Using Convolutional Neural Network to recognize emotion from the audio recording. And the repository owner does not provide any paper reference.

Data Description:
These are two datasets originally made use in the repository RAVDESS and SAVEE, and I only adopted RAVDESS in my model. In the RAVDESS, there are two types of data: speech and song.

Data Set: The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)

12 Actors & 12 Actresses recorded speech and song version respectively.
Actor no.18 does not have song version data.
Emotion Disgust, Neutral and Surprised are not included in the song version data.
Total Class:

Here is the emotion class distribution bar chart.


Feature Extraction:
When we do Speech Recognition tasks, MFCCs is the state-of-the-art feature since it was invented in the 1980s.

This shape determines what sound comes out. If we can determine the shape accurately, this should give us an accurate representation of the phoneme being produced. The shape of the vocal tract manifests itself in the envelope of the short time power spectrum, and the job of MFCCs is to accurately represent this envelope. — Noted from: MFCC tutorial

Waveform

Spectrogram
We would use MFCCs to be our input feature. If you want a thorough understanding of MFCCs, here is a great tutorial for you. Loading audio data and converting it to MFCCs format can be easily done by the Python package librosa.

Default Model Architecture:
The author developed the CNN model with Keras and constructed with 7 layers — 6 Conv1D layers followed by a Dense layer.

model = Sequential()
model.add(Conv1D(256, 5,padding='same', input_shape=(216,1))) #1
model.add(Activation('relu'))
model.add(Conv1D(128, 5,padding='same')) #2
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(128, 5,padding='same')) #3
model.add(Activation('relu'))
#model.add(Conv1D(128, 5,padding='same')) #4
#model.add(Activation('relu'))
#model.add(Conv1D(128, 5,padding='same')) #5
#model.add(Activation('relu'))
#model.add(Dropout(0.2))
model.add(Conv1D(128, 5,padding='same')) #6
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(10)) #7
model.add(Activation('softmax'))
opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)
The author commented layer #4 and #5 in the latest notebook (18 Sep 2018 Update) and the model weight file does not fit the network provided, thus, I cannot load the weight provide and replicate its result 72% Testing Accuracy.
The model only simply trained with batch_size=16 and 700 epochs without any learning rate schedule, etc.

# Compile Model
model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
# Fit Model
cnnhistory=model.fit(x_traincnn, y_train, batch_size=16, epochs=700, validation_data=(x_testcnn, y_test))
Its loss function is categorical_crossentropy and the evaluation metric is accuracy.

My Experiment
Exploratory Data Analysis:
In the RADVESS dataset, each actor has to perform 8 emotions by saying and singing two sentences and two times for each. As a result, each actor would induce 4 samples for each emotion except neutral, disgust and surprised since there is no singing data for these emotions. Each audio wave is around 4 second, the first and last second are most likely silenced.

The standard sentences are:

1. Kids are talking by the door.
2. Dogs are sitting by the door.
Observation:

After I selected 1 actor and 1 actress’s dataset and listened to all of them. I found out male and female are expressing their emotions in a different way. Here are some findings:

Male’s Angry is simply increased in volume.​
Male’s Happy and Sad significant features were laughing and crying tone in the silenced period in the audio.
Female’s Happy, Angry and Sad are increased in volume.​
Female’s Disgust would add vomiting sound inside.
Replicating Result:

The author excluded the class neutral, disgust and surprised to do a 10 class recognition for the RAVDESS dataset.

I tried to replicate his result with the model provided, I can achieve a result of


However, I found out there is a data leakage problem where the validation set used in the training phase is identical to the test set. So, I re-do the data splitting part by isolating two actors and two actresses data into the test set which make sure it is unseen in the training phase.

Actor no. 1–20 are used for Train / Valid sets with 8:2 splitting ratio.
Actor no. 21–24 are isolated for testing usage.
Train Set Shape: (1248, 216, 1)
Valid Set Shape: (312, 216, 1)
Test Set Shape: (320, 216, 1) — (Isolated)
I re-trained the model with the new data-splitting setting and here is the result:


Benchmark:
From the train valid loss graph, we can see the model cannot even converge well with 10 target classes. Thus, I decided to reduce the complexity of my model by recognizing male emotions only. I isolated the two actors to be the test set, and the rest would be the train/valid set with 8:2 Stratified Shuffle Split which ensures there is no class imbalance in the dataset. Afterward, I trained both male and female data separately to explore the benchmark.

Male Dataset

Train Set = 640 samples from actor 1- 10.
Valid Set = 160 samples from actor 1- 10.
Test Set = 160 samples from actor 11- 12.
Male Baseline


Female Dataset

Train Set = 608 samples from actress 1- 10.
Valid Set = 152 samples from actress 1- 10.
Test Set = 160 samples from actress 11- 12.
Female Baseline


As you can see, the confusion matrix of the male and female model is different.

- Male: Angryand Happy are the dominant predicted classes in the male model but they are unlikely to mix up.​

- Female: Sad and Happy are the dominant predicted classes in the female model and Angry and Happy are very likely to mix up.

Referring to the observation form the EDA section, I suspect the reason for female Angry and Happy are very likely to mix up is because their expression method is simply increasing the volume of the speech.

On top of it, I wonder what if I further simplify the model by reducing the target class to Positive, Neutral and Negative or even Positive and Negative only. So, I grouped the emotions into 2 class and 3 class.

2 Class:

Positive: happy, calm.
Negative: angry, fearful, sad.
3 Class:

Positive: happy.
Neutral: calm, neutral.
Negative: angry, fearful, sad.
(Added neutral to the 3 class to explore the result.)

Before I do the training experiment, I tune the model architecture with the male data by doing 5 class recognition.

# Set the target class number
target_class = 5
# Model 
model = Sequential()
model.add(Conv1D(256, 8, padding='same',input_shape=(X_train.shape[1],1))) #1
model.add(Activation('relu'))
model.add(Conv1D(256, 8, padding='same')) #2
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(128, 8, padding='same')) #3
model.add(Activation('relu')) 
model.add(Conv1D(128, 8, padding='same')) #4
model.add(Activation('relu'))
model.add(Conv1D(128, 8, padding='same')) #5
model.add(Activation('relu'))
model.add(Conv1D(128, 8, padding='same')) #6
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(64, 8, padding='same')) #7
model.add(Activation('relu'))
model.add(Conv1D(64, 8, padding='same')) #8
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(target_class)) #9
model.add(Activation('softmax'))
opt = keras.optimizers.SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)
I added 2 Conv1D layers, 1 MaxPooling1D layer and 2 BarchNormalization layers, moreover, I changed the dropout value to 0.25. Lastly, I changed the optimizer to SGD with 0.0001 learning rate.

lr_reduce = ReduceLROnPlateau(monitor=’val_loss’, factor=0.9, patience=20, min_lr=0.000001)
mcp_save = ModelCheckpoint(‘model/baseline_2class_np.h5’, save_best_only=True, monitor=’val_loss’, mode=’min’)
cnnhistory=model.fit(x_traincnn, y_train, batch_size=16, epochs=700, validation_data=(x_testcnn, y_test), callbacks=[mcp_save, lr_reduce])
For the model training, I adopted Reduce Learning On Plateau and save the best model with the min val_loss only. And here are the model performance of different target class setups.

New Model Performance
Male 5 Class

Female 5 Class

Male 2 Class

Male 3 Class

Augmentation
After I tuned the model architecture, optimizer and learning rate schedule, I found out the model still cannot converge in the training period. I assumed it is the data size problem since we have 800 samples for train valid set only. Thus, I decided to explore the audio augmentation methods. Let’s take a look at some augmentation method with code. I simply augmented all of the datasets once to double the train / valid set size.

Male 5 Class:
Dynamic Value Change

def dyn_change(data):
    """
    Random Value Change.
    """
    dyn_change = np.random.uniform(low=1.5,high=3)
    return (data * dyn_change)
Pitch Tuning

def pitch(data, sample_rate):
    """
    Pitch Tuning.
    """
    bins_per_octave = 12
    pitch_pm = 2
    pitch_change =  pitch_pm * 2*(np.random.uniform())   
    data = librosa.effects.pitch_shift(data.astype('float64'), 
                                      sample_rate, n_steps=pitch_change, 
                                      bins_per_octave=bins_per_octave)
Shifting

def shift(data):
    """
    Random Shifting.
    """
    s_range = int(np.random.uniform(low=-5, high = 5)*500)
    return np.roll(data, s_range)
White Noise Adding

def noise(data):
    """
    Adding White Noise.
    """
    # you can take any distribution from https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.random.html
    noise_amp = 0.005*np.random.uniform()*np.amax(data)
    data = data.astype('float64') + noise_amp * np.random.normal(size=data.shape[0])
    return data
We can see that the augmentation can jack up the Validation Accuracy a lot, 70+% in general. Especially that adding white noise can achieve 87.19% Validation Accuracy, however, the Testing Accuracy and Testing F1-score dropped more than 5% respectively. Then, I wonder if I mixing different augmentation methods would bring a good result.
Mixing Multiple Methods
Noise Adding + Shifting

Testing Augmentation on Male 2 Class Data
Male 2 Class:
Noise Adding + Shifting
For all sample

Noise Adding + Shifting 
For positive sample only since the 2 class set is imbalance (skewed toward negative).

Pitch Tuning + Noise Adding
For all sample

Pitch Tuning + Noise Adding
For positive sample only

Conclusion
In the end, I only have time to experiment with the male data set. I re-split the data with stratified shuffle split to make sure there is no data imbalance nor data leakage problem. I tuned the model by experimenting the male dataset since I want to simplified the model at the beginning. I also tested the by with different target label setups and augmentation method. I found out Noise Adding and Shifting for the imbalanced data could help in achieving a better result.

Key Take Away
Emotions are subjective and it is hard to notate them.
We should define the emotions that suitable for our own project objective.
Do not always trust the content from GitHub even it has lots of stars.
Be aware of the data splitting.
Exploratory Data Analysis always grant us good insight, and you have to be patient when you work on audio data!
Deciding the input for your model: a sentence, a recording or an utterance?
Lack of data is a crucial factor to achieve success in SER, however, it is complex and very expensive to build a good speech emotion dataset.
Simplified your model when you lack data.
Further Improvement
I only selected the first 3 seconds to be the input data since it would reduce the dimension, the original notebook used 2.5 sec only. I would like to use the full length of the audio to do the experiment.
Preprocess the data like cropping silence voice, normalize the length by zero padding, etc.
Experiment the Recurrent Neural Network approach on this topic.
About Me
GitHub: rezachu/emotion_recognition_cnn
Linkedin: Kai Cheong, Reza Chu
Remarks: Some papers related to this topic will be noted in the GitHub soon.

Machine LearningDeep LearningNLPSpeech RecognitionEmotion Recognition
Go to the profile of Reza Chu
Reza Chu
Medium member since Jan 2019
Data Scientist of Computer Vision & Natural Language Processing

Towards Data Science
Towards Data Science
Sharing concepts, ideas, and codes.

More from Towards Data Science
Jupyter is the new Excel
Go to the profile of Semi Koen
Semi Koen
Jun 7
More from Towards Data Science
Using reinforcement learning to trade Bitcoin for massive profit
Go to the profile of Adam King
Adam King
Jun 4
More from Towards Data Science
An Overview of Python’s Datatable package
Go to the profile of Parul Pandey
Parul Pandey
Jun 2
Responses
Tse Yang Macy
Write a response…
Tse Yang Macy
Conversation with Reza Chu.
Go to the profile of Kesavarao Bagadi
Kesavarao Bagadi
Jun 12
mcp_save = ModelCheckpoint(‘model/aug_noiseNshift_2class2_np.h5’, save_best_only=True, monitor=’val_loss’, mode=’min’)

can u clarify this point.. i got an error

Go to the profile of Reza Chu
Reza Chu
Jun 13
Can you provide the error code?

Towards Data Science
Sharing concepts, ideas, and codes.
4
3
Smart stories. New ideas. No ads. $5/month.
