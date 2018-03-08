import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Dropout,Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

##Reading csv file and saving in  lines array
lines=[]
with open('./data/driving_log.csv') as csvfile:
	reader=csv.reader(csvfile)
	for line in reader:
		lines.append(line)


#Now changing images file names relative to cloud storage
images=[] 
measurements=[]
i=0
for line in lines:
	if(i==0):
		i+=1
		continue
	for j in range(3): #Reading all three column images (center, left and right)
		source_path=line[j]
		filename=source_path.split('/')[-1]
		current_path='./data/IMG/'+filename
		image=cv2.imread(current_path, cv2.COLOR_BGR2RGB)
		images.append(image)
		if(j==0):
			measurement=float(line[3])			
		elif(j==1):
			measurement=float(line[3])+0.2 #Reducing the left camera bias
		else:
			measurement=float(line[3])-0.2 #Reducing the right camera bias		
		measurements.append(measurement)
		if(j==0):
 		if(measurement!=0.0): #Image augmentation
 			images.append(cv2.flip(image,1)) #Flipping the image vertically
 			measurements.append(measurement*-1.0) # Changing the steering direction

#Reading additional training images 
lines=[]
with open('./mydata/driving_log.csv') as csvfile:
	reader=csv.reader(csvfile)
	for line in reader:
		lines.append(line)
print(len(lines))


#Now changing images file names relative to cloud storage
i=0
for line in lines:
	if(i==0):
		i+=1
		continue
	for j in range(3):
		source_path=line[j]
		#print(source_path)
		filename=source_path.split('/')[-1]
		print(filename)
		current_path='./mydata/IMG/'+filename
		print(current_path)
		image=cv2.imread(current_path, cv2.COLOR_BGR2RGB)
		print(image.shape)
		images.append(image)
		if(j==0):
			measurement=float(line[3])			
		elif(j==1):
			measurement=float(line[3])+0.2
		else:
			measurement=float(line[3])-0.2		
		measurements.append(measurement)



X_train=np.array(images)  # Creating training array values
#print(X_train.shape)
Y_train=np.array(measurements) # Creating training labels



model = Sequential()
model.add(Lambda(lambda x:x/255 -0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(3,5,5,subsample=(1, 1),activation='relu'))
model.add(Convolution2D(24,5,5,subsample=(2, 2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2, 2),activation='relu'))
model.add(Convolution2D(48,3,3,subsample=(2, 2),activation='relu'))
model.add(Convolution2D(64,3,3,subsample=(2, 2),activation='relu'))
model.add(Flatten())
model.add(Dense(1166))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse',optimizer='adam')
model.fit(X_train,Y_train,validation_split=0.2,shuffle=True,nb_epoch=5)

model.save('./model.h5')
