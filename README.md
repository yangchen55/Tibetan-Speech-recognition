# Tibetan-Speech-recognition
This is web based speech recognition with use of machine learning algorithm(CNN) and achieve accuracy of 96%


Speech Recognition with Convolutional Neural Network


Before we walk through the project, it is good to know the major bottleneck of Speech  Recognition.

Major Obstacles:

Annotating an audio recording is challenging. Should we label a single word, sentence or a whole conversation?
Collecting data is complex. There are lots of audio data can be achieved from films or news. 
Image Credit: depositphotos
Project Description:




Feature Extraction:
When we do Speech Recognition tasks, MFCCs is the state-of-the-art feature since it was invented in the 1980s.

This shape determines what sound comes out. If we can determine the shape accurately, this should give us an accurate representation of the phoneme being produced. The shape of the vocal tract manifests itself in the envelope of the short time power spectrum, and the job of MFCCs is to accurately represent this envelope. — Noted from: MFCC tutorial



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


# Compile Model
model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])

# Fit Model
cnnhistory=model.fit(x_traincnn, y_train, batch_size=16, epochs=700, validation_data=(x_testcnn, y_test))
Its loss function is categorical_crossentropy and the evaluation metric is accuracy.

My Experiment
Exploratory Data Analysis:
in the dataset, each author speak about 8 alphabets and machine got stuck to classify in terms of nga and na as both of them are nasal and sounds very similar.

