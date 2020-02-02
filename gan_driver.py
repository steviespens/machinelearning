from gan import training
from jpg_to_csv import convert



#convert some jpg images to csv of pixels
myDir = 'faces'
numToConvert = 200
# convert(myDir, numToConvert)

#train GAN
epochs = 1
batch_size = 1
fileLocation = 'img_pixels.csv'
outputFolderName = 'training_progression'
saveModelAsName = 'model.h5'
training(fileLocation, outputFolderName, saveModelAsName, epochs, batch_size)
