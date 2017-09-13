import nibabel as nib
import os
import numpy
import scipy
from scipy import ndimage
import math
import time
from nilearn import plotting
from skimage.transform import downscale_local_mean as downscale
from nilearn import masking
from sklearn import svm
import random
from random import shuffle
from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2
import xlsxwriter


def main():
	"""
	This is where the testing is set up as well as storing the results into an excel file. You can control several parameters to see what the results
	of changing them are. 
	"""


	#downsampling range. Current testing is set up to see the effects of changing the downsampling factor on the accuracy.
	d_low = 4
	d_high = 6


	num_controls = 10
	num_patients = 10
	num_features = 50
	num_tests = 2
	resultFileName = 'cohogResults.xlsl'

	c = 0
	while resultFileName in os.listdir():
		resultFileName = 'cohogResults'+str(c)+'.xlsl'
		c += 1



	#open and create results excel file
	workbook = xlsxwriter.Workbook(resultFileName)
	worksheet = workbook.add_worksheet()


	count = 0
	test_num = 1
	step = 10

	#for each downsampling factor being tested
	for d in range(d_low, d_high+1):

		#setting up row titles
		worksheet.write(0 + step*count,0,"Downsampled")
		worksheet.write(1 + step*count,0,"Accuracy")
		worksheet.write(2 + step*count,0,"Number of Samples")
		worksheet.write(3 + step*count,0,"Number of Controls")
		worksheet.write(4 + step*count,0,"Number of Patients")
		worksheet.write(5 + step*count,0,"Time")
		worksheet.write(6 + step*count,0,"Num Features")
		worksheet.write(7 + step*count,0,"OVERALL ACCURACY")


		overall_downed_accuracy = 0


		for n in range(1, num_tests+1):
			print("DOWNSAMPLED %d TEST %d" % (d, n))

			#start timer and get results
			start = time.time()
			acc = cohog(d, num_controls, num_patients, num_features)
			t = time.time()-start

			#write results into excel file
			worksheet.write(0 + step*count,n,d)
			worksheet.write(1 + step*count,n,acc)
			worksheet.write(2 + step*count,n,num_controls + num_patients)
			worksheet.write(3 + step*count,n,num_controls)
			worksheet.write(4 + step*count,n,num_patients)
			worksheet.write(5 + step*count,n,t)
			worksheet.write(6 + step*count,n,num_features)

			#accumulate overall accuracy to calc avg accuracy later
			overall_downed_accuracy += acc

			result = "Downsampled: %d \nAccuracy: %f \nTime: %f \nNum Samples: %d \nNum Controls: %d \nNum Patients: %d \nNum Features: %d\n\n" % (d,acc,t,num_patients + num_controls,num_controls,num_patients,num_features)
			print(result)
				

		avg_downed_acc = overall_downed_accuracy/num_tests

		worksheet.write(7 + step*count,1,avg_downed_acc)
		count += 1


	workbook.close()




def bin_sobel_3D(s_mags):
	"""
	Given the magnitudes for the Soble gradient in the x,y and z directions, returns a number representing
	the appropriate bin. Currently returns bin 1-8 and 0 if the gradient in any direction == 0

	Refer to the diagram in bin_sobel_2d but now think of it as being a cube split into eight sections. 

	Parameters:
		s_mags: LIST containing the Sobel gradient magnitudes in the x,y and z directions.

	Returns:
		INT from 0-8 
	"""
	x = s_mags[0]
	y = s_mags[1]
	z = s_mags[2]

	if x > 0 and y > 0 and z > 0:
		return 1
	if x > 0 and y > 0 and z < 0:
		return 2
	if x > 0 and y < 0 and z > 0:
		return 3
	if x > 0 and y < 0 and z < 0:
		return 4
	if x < 0 and y > 0 and z > 0:
		return 5
	if x < 0 and y > 0 and z < 0:
		return 6
	if x < 0 and y < 0 and z > 0:
		return 7
	if x < 0 and y < 0 and z < 0:
		return 8
	else:
		return 0

def create_co_occurence_vector_3D(matrix):
	"""
	Given a matrix representing the bin value of each voxel, creates a co-occurence matrix and returns it
	in vectorized form.

	Parameters:
		matrix: NUMPY.NDARRAY matrix containing the bin value for each voxel
	Returns:
		NUMPY.ARRAY vector of size num_features
	"""
	# print("Making co-occurrence vector")
	start = time.time()
	# print("Sobel Matrix")
	# for i in range(len(matrix)):
	# 	print(matrix[i])

	#scans offets in a cube out from the current voxel. Neighbourhood size is the 
	#side length of that cube.
	neighbourhoodsize = 4
	num_offsets = neighbourhoodsize ** 3

	#9 because there's 9 direction bins
	co_matrix = numpy.zeros((num_offsets,9,9))
	xdim,ydim,zdim = len(matrix),len(matrix[0]),len(matrix[0][0])

	#for each voxel in the scan
	for x in range(xdim - neighbourhoodsize):
		for y in range(ydim - neighbourhoodsize):
			for z in range(zdim - neighbourhoodsize):
				scanning_voxel = int(matrix.item(x,y,z))

				#for each offset per voxel
				offset_count = 0
				for a in range(neighbourhoodsize):
					for b in range(neighbourhoodsize):
						for c in range(neighbourhoodsize):
							neighbour_voxel = int(matrix.item(x+a,y+b,z+c))
							prev_value = co_matrix.item((offset_count, scanning_voxel,neighbour_voxel))

							#increment the value by one every co-occurence, note the offset number and sobel values 
							#for both voxels are being used to index in the co-occurrence matrix
							co_matrix.itemset((offset_count, scanning_voxel, neighbour_voxel), prev_value + 1)

							offset_count += 1
							

	# print("Co-occurence matrix")
	# for i in range(len(co_matrix)):
	# 	print(co_matrix[i])

	#reshape matrix to one dimensional vector
	#9 because there's 9 direction bins
	co_vector = co_matrix.reshape(9 * 9 * num_offsets)

	# print("CoVector:")
	# print(co_vector)
	# print("TIME: " + str(time.time() - start))


	return co_vector

def create_co_occurence_vector_2d(matrix):
	"""
	Given a matrix representing the bin value of each voxel, creates a co-occurence matrix and returns it
	in vectorized form.

	Parameters:
		matrix: NUMPY.NDARRAY matrix containing the bin value for each voxel
	Returns:
		NUMPY.ARRAY vector of size num_features
	"""
	# print("Making co-occurrence vector")
	start = time.time()


	#scans offets in a cube out from the current voxel. Neighbourhood size is the 
	#side length of that cube.
	neighbourhoodsize = 8
	num_offsets = neighbourhoodsize ** 2

	#9 because there are 9 possible bins
	co_matrix = numpy.zeros((num_offsets,9,9))

	# xdim,ydim = len(matrix),len(matrix[0])
	xdim,ydim = matrix.shape[0], matrix.shape[1]

	#for each voxel in the scan
	for x in range(0, xdim - neighbourhoodsize):
		for y in range(0, ydim - neighbourhoodsize):

			scanning_voxel = int(matrix.item(x,y))

			#for each offset per voxel
			offset_count = 0
			for a in range(0, neighbourhoodsize):
				for b in range(0, neighbourhoodsize):

					neighbour_voxel = int(matrix.item(x+a,y+b))
					prev_value = co_matrix.item((offset_count, scanning_voxel,neighbour_voxel))

					#increment the value by one every co-occurence
					#note the offset number and pixel values are being used to index the co-occurence matrix
					co_matrix.itemset((offset_count, scanning_voxel, neighbour_voxel), prev_value + 1)

					offset_count += 1
							

	#reshape matrix to one dimensional vector
	#9*9 because there are 9 possible bins
	co_vector = co_matrix.reshape(9 * 9 * num_offsets)

	# print("TIME: " + str(time.time() - start))


	return co_vector



def apply_sobel_3D(img):

	"""
	Takes in the data array from a nii file and applies the Sobel gradient operator on it. Sobel gradient
	directions are binned from 0-8.
	Parameters:
		img: NUMPY.NDARRAY representing the greyscale voxel values

	Returns:
		NUMPY.NDARRAY of the binned Sobel direction for each voxel
		NUMPY.NDARRAY of the Sobel magnitude for each voxel
	"""


	# print("Applying Sobel")
	start = time.time()
	sobel_threshold = 2

	#sobel gradient matrices in x, y, z diirections
	dx = ndimage.sobel(img,0)
	dy = ndimage.sobel(img,1)
	dz = ndimage.sobel(img,2)

	shape = dx.shape

	sobel_magnitudes = numpy.ndarray(shape = shape)
	sobel_directions = numpy.ndarray(shape = shape)

	#for every voxel in the matrix
	for x in range(shape[0]):
		for y in range(shape[1]):
			for z in range(shape[2]):
				coord = tuple((x,y,z))
				mx = dx.item(coord)
				my = dy.item(coord)
				mz = dz.item(coord)

				#calculate sobel magnitude and save it
				mag = math.sqrt((mx**2) + (my**2) + (mz**2))

				sobel_magnitudes.itemset(coord,mag)

				#if the magnitude is above the threshold, calculate the bin and save 
				#it in the gradient directions matrix
				if mag > sobel_threshold:
					bin = bin_sobel_3D((mx,my,mz))
					sobel_directions.itemset(coord,bin)
				else:
					sobel_directions.itemset(coord,0)


	return sobel_directions, sobel_magnitudes



def apply_sobel_2d(img):

	"""
	Takes in the data array from a nii file and applies the Sobel gradient operator on it. Sobel gradient
	directions are binned from 0-8.
	Parameters:
		NUMPY.NDARRAY representing the greyscale voxel values

	Returns:
		NUMPY.NDARRAY of the binned Sobel direction for each voxel
		NUMPY.NDARRAY of the Sobel magnitude for each voxel
	"""



	start = time.time()
	sobel_threshold = 2

	#matrices of sobel gradients in x and y directions
	dx = ndimage.sobel(img,0)
	dy = ndimage.sobel(img,1)

	shape = dx.shape

	sobel_magnitudes = numpy.ndarray(shape = shape)
	sobel_directions = numpy.ndarray(shape = shape)

	#for every pixel
	for x in range(shape[0]):
		for y in range(shape[1]):

			coord = tuple((x,y))
			mx = dx.item(coord)
			my = dy.item(coord)

			#calculate gradient magnitude
			mag = math.sqrt((mx**2) + (my**2))
			sobel_magnitudes.itemset(coord,mag)

			#if mag is high enough, calculate gradient bin and save it
			if mag > sobel_threshold:

				bin = bin_sobel_2d(mx, my)
				sobel_directions.itemset(coord,bin)
			else:

				sobel_directions.itemset(coord,0)

				

	# print("TIME: " + str(time.time()-start))

	return sobel_directions, sobel_magnitudes


def bin_sobel_2d(mx, my):
	"""
	The magnitudes are binned by the direction that would result from the vector addition
	of the two image gradients, x and y

	Parameters:
		mx: INT Gradient magnitude in the x-direction
		my: INT Gradient magnitude in the y-direction

	Returns:
		*bin number for each increment of degrees
		1: 0-45
		2: 45-90
		3: 90-135
		4: 135-180
		5: 180-225
		6: 225-270
		7: 270-315
		8: 315-360
		0: One of the gradient values is 0, must edge of image


Imagine these angles are actually 45 degrees. Visual to help
understand binning logic.
	   y		
	\ 3|2 /
     \ | / 
     4\|/1
--------------x 
     5/|\8
     / | \
    / 6|7 \ 

	"""
	if mx > 0 and my > 0 and mx > my:
		return 1

	if mx > 0 and my > 0 and mx <= my:
		return 2

	if mx < 0 and my > 0 and abs(mx) < my:
		return 3

	if mx < 0 and my > 0 and abs(mx) >= my:
		return 4

	if mx < 0 and my < 0 and abs(mx) > abs(my):
		return 5

	if mx < 0 and my < 0 and abs(mx) <= abs(my):
		return 6

	if mx > 0 and my < 0 and mx < abs(my):
		return 7

	if mx > 0 and my < 0 and mx >= abs(my):
		return 8

	else:
		return 0



def random_train_test_indices(num_patients, num_controls, percent_test):
	"""
	Returns randomly selected sets of indices used to test and train the svm.

	Parameters:
		num_patients: INT number of patients
		num_controls: INT number of controls
		percent_test: FLOAT percent of samples to be used for testing (50% --> 0.5)

	Returns:
		SET patient training indexes
		SET patient testing indexes
		SET control training indexes
		SET control testing indexes
	"""

	p_train_idxs = [p for p in range(num_patients)]
	c_train_idxs = [c for c in range(num_controls)]	
	
	p_test_idxs = set() 
	c_test_idxs = set()

	while len(p_test_idxs) < int(num_patients * percent_test):
		shuffle(p_train_idxs)
		rand_idx = p_train_idxs.pop()
		p_test_idxs.add(rand_idx)

	while len(c_test_idxs) < int(num_controls * percent_test):
		shuffle(c_train_idxs)
		rand_idx = c_train_idxs.pop()
		c_test_idxs.add(rand_idx)

	p_train_idxs = set(p_train_idxs)
	c_train_idxs = set(c_train_idxs)

	return p_train_idxs, p_test_idxs, c_train_idxs, c_test_idxs

	


def train_and_test_svm(patients, controls, num_features):
	"""
	Uses a defined portion of the samples provided to train a SVM, the rest of the samples
	are used to test the SVM.

	Parameters:
		patients: 		LIST feature vectors representing the patients
		controls: 		LIST feature vectors representing the controls
		num_features: 	INT number of features to use in k-best features selection

	Returns:
		INT accuracy result from testing
	"""
	p_class = [1 for x in range(len(patients))]
	c_class = [0 for x in range(len(controls))]
	all_classes = p_class + c_class

	#all samples are used for the top K feature selection
	all_samples = patients + controls
	# k_best = SelectPercentile(score_func = chi2, percentile = 5)
	k_best = SelectKBest(score_func = chi2, k = num_features)
	k_best.fit(all_samples, all_classes)

	#samples will be divided for training and testing
	percent_train = 0.5

	num_patients = len(patients)
	num_controls = len(controls)

	#number of iterations with different training/testing groups
	num_tests = 200
	# print("\n\nBeginning SVM Testing/Training")
	# print(str(num_tests) + " tests")
	start = time.time()
	total_accuracy = 0

	for n in range(num_tests):
		p_train_idxs, p_test_idxs, \
		c_train_idxs, c_test_idxs = random_train_test_indices(num_patients, num_controls, percent_train)

		train_array = list()
		test_array = list()
		train_classes = list()
		test_classes = list()

		#create arrays from patients/controls using randomized indices
		for i in p_train_idxs:
			train_array.append(patients[i])
			train_classes.append(1)

		for i in p_test_idxs:
			test_array.append(patients[i])
			test_classes.append(1)

		for i in c_train_idxs:
			train_array.append(controls[i])
			train_classes.append(0)

		for i in c_test_idxs:
			test_array.append(controls[i])
			test_classes.append(0)


		#transform arrays to only include the selected features
		train_array = k_best.transform(train_array)
		test_array = k_best.transform(test_array)
		
		#train (fit) the svm with training array/classes and then predict results from the test array
		clf = svm.LinearSVC()
		clf.fit(train_array, train_classes)
		guesses = clf.predict(test_array)

		num_correct = 0

		#calculate test score
		for n in range(len(test_array)):
			actual = test_classes[n]
			guess = guesses[n]


			if actual == guess: 
				num_correct += 1



		accuracy = num_correct*100/len(test_array)
		# print("test accuracy: " + str(accuracy))
		total_accuracy += accuracy

	#average accuracy over all tests
	overall_accuracy = total_accuracy/num_tests

	# print("SVM TIME: " + str(time.time()-start) + "\n\n")

	return overall_accuracy



def apply_mask(img, mask):
	"""
	Returns 2D array representing the masked img
	Parameters:
		img: nii-like nibabel image representing the image
		mask: nii-like nibabel image representing the mask

	Returns:
		masked_img: NUMPY ARRAY 2d array representing the masked img 
	"""
	#get 3d data arrays
	img_data = img.get_data()
	mask_data = mask.get_data()

	mask = list()
	masked_img = list()
	slices = set()

	#add the slices of the 3d image where the mask exists 
	for x in range(mask_data.shape[0]):
		for y in range(mask_data.shape[1]):
			if numpy.sum(mask_data[x][y]) > 0:
				mask.append(mask_data[x][y])
				masked_img.append(img_data[x][y])

	#mask out 2d img by multiplying it by mask values (which are 0 or 1)
	for x in range(len(masked_img)):
		for y in range(len(masked_img[0])):
			masked_img[x][y] = masked_img[x][y] * mask[x][y]


	#convert python list to numpy array
	masked_img = numpy.array(masked_img)
	return masked_img

def load_masked_downed_nii_from_directory(dir_name, num_samples, df):
	"""
	Loads all nii files in a directory. Assumes there is a subdirectory names "masks" containing masks named
	"<imgFileName>Mask.nii". Once files are loaded they are masked and downsampled and returned as a list.

	Parameters:
		dir_name: 		STRING name of directory containing nii files and mask directory
		num_samples: 	INT number of samples to load and process from the directory
		df: 			INT downsampling factor used when loading the scans. Use 0 for no downsampling.

	Returns:
						LIST matrices representing the downsampled and masked images
	"""

	os.chdir(os.getcwd() + "\\" + dir_name)
	all_files = os.listdir()

	sampleFiles = list()
	samples = list()

	for f in all_files:
		#only include files with "nii" extension
		f_split = f.split(".")
		if len(f_split) < 2 or f_split[1] != "nii":
			continue
		else:
			sampleFiles.append(f)

	#remove random files until you have num_samples left
	if str(num_samples).lower() != "all":
		while len(sampleFiles) > num_samples:
			shuffle(sampleFiles)
			sampleFiles.pop()

	#load, mask, and downsample every image, then append it to the list of samples
	for f in sampleFiles:
		split_f = f.split(".")
		img = nib.load(f)
		mask = nib.load(os.getcwd() + "\\masks\\" + split_f[0] + "Mask.nii")

		masked_img = apply_mask(img, mask)

		if df != 0:
			downed = downscale(masked_img, (df,df))
			samples.append(downed)
		else:
			samples.append(masked_img)
	os.chdir("..")

	return samples







def cohog(downsample_factor, num_controls, num_patients, num_features):
	"""
	
	Loads and masks nii files from directories 'controls' and 'patients'. Assumes each directory has a 
	'masks' subdirectory. 

	Implements the COHOG method on scans:
	1. apply Sobel gradient operator to each scans
	2. calculate co-occurrence matrices from each scans sobel_direction matrix
	3. train and test svm
	4. return testing accurracy 

	Parameters:
		downsample_factor: 	INT the downsampling factor used when loading the scans. use 0 for no downsampling
		num_controls: 		INT the number of controls to use from the 'controls' directory
		num_patients: 		INT the number of patients to use from the 'patients' directory
		num_features: 		INT the number of most relevant features used to test and train the SVM.

	Returns:
							INT overall average testing accuracy from the SVM


	"""
	start = time.time()

	#load scans from 'controls' and 'patients' directories
	controls = load_masked_downed_nii_from_directory('controls',num_controls,downsample_factor)
	patients = load_masked_downed_nii_from_directory('patients',num_patients,downsample_factor)

	c_vectors, p_vectors = list(), list()


	count = 0

	#process each scan and save vectors in separate lists
	for c in controls:
		sobel_d, sobel_m = apply_sobel_2d(c)
		co_vector = create_co_occurence_vector_2d(sobel_d)
		c_vectors.append(co_vector)
		count += 1


	for p in patients:
		sobel_d, sobel_m = apply_sobel_2d(p)
		co_vector = create_co_occurence_vector_2d(sobel_d)
		p_vectors.append(co_vector)
		count += 1



	accuracy = train_and_test_svm(p_vectors, c_vectors, num_features)

	return accuracy

main()





	
















