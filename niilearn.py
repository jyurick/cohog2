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




def bin_sobel(s_mags):
	"""
	Given the magnitudes for the Soble gradient in the x,y and z directions, returns a number representing
	the appropriate bin. Currently returns bin 1-8 and 0 if the gradient in any direction == 0

	Parameters:
		TUPLE/LIST containing the Sobel gradient magnitudes in the x,y and z directions.

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

def create_co_occurence_vector(matrix):
	"""
	Given a matrix representing the bin value of each voxel, creates a co-occurence matrix and returns it
	in vectorized form.

	Parameters:
		NUMPY.NDARRAY matrix containing the bin value for each voxel
	Returns:
		NUMPY.ARRAY vector of size num_features
	"""
	print("Making co-occurrence vector")
	start = time.time()

	#scans offets in a cube out from the current voxel. Neighbourhood size is the 
	#side length of that cube.
	neighbourhoodsize = 4
	num_offsets = neighbourhoodsize ** 3


	co_matrix = numpy.zeros((9,9,num_offsets))
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
							prev_value = co_matrix.item((scanning_voxel,neighbour_voxel,offset_count))

							#increment the value by one every co-occurence
							co_matrix.itemset((scanning_voxel, neighbour_voxel, offset_count), prev_value + 1)
							offset_count += 1


	#reshape matrix to one dimensional vector
	co_vector = co_matrix.reshape(9 * 9 * num_offsets)
	print("TIME: " + str(time.time() - start))

	return co_vector


def apply_sobel(img):

	"""
	Takes in the data array from a nii file and applies the Sobel gradient operator on it. Sobel gradient
	directions are binned from 0-8.
	Parameters:
		NUMPY.NDARRAY representing the greyscale voxel values

	Returns:
		NUMPY.NDARRAY of the binned Sobel direction for each voxel
		NUMPY.NDARRAY of the Sobel magnitude for each voxel
	"""


	print("Applying Sobel")
	start = time.time()
	dx = ndimage.sobel(img,0)
	dy = ndimage.sobel(img,1)
	dz = ndimage.sobel(img,2)

	shape = dx.shape

	sobel_magnitudes = numpy.ndarray(shape = shape)
	sobel_directions = numpy.ndarray(shape = shape)

	for x in range(shape[0]):
		for y in range(shape[1]):
			for z in range(shape[2]):
				coord = tuple((x,y,z))
				mx = dx.item(coord)
				my = dy.item(coord)
				mz = dz.item(coord)

				mag = math.sqrt((mx**2) + (my**2) + (mz**2))

				sobel_magnitudes.itemset(coord,mag)

				
	max_mag = numpy.max(sobel_magnitudes[0])
	print("Average: " + str(numpy.average(sobel_magnitudes[0])))
	print("Max: " + str(numpy.max(sobel_magnitudes[0])))

	sobel_threshold = max_mag/10

	for x in range(shape[0]):
		for y in range(shape[1]):
			for z in range(shape[2]):
				coord = tuple((x,y,z))
				mx = dx.item(coord)
				my = dy.item(coord)
				mz = dz.item(coord)

				mag = sobel_magnitudes.item(coord)

				if mag >= sobel_threshold:
					bin = bin_sobel((mx, my, mz))
					sobel_directions.itemset(coord,bin)
				else:
					sobel_directions.itemset(coord,0)



	print("TIME: " + str(time.time()-start))

	return sobel_directions, sobel_magnitudes


def load_and_downsample(df):
	"""
	Assumes the program is run from a directory containing folders for controls and patients. 
	Loads nii files from those folders and downsamples them by the downsampling factor df.

	Parameters:
		INT df downsampling factor

	Returns:
		LIST downsampled controls
		LIST downsampled patients
	"""

	controls = list()
	patients = list()

	os.chdir(os.getcwd() + "\\controls")
	for f in os.listdir():
		print("downsample " + str(f))
		split_f = f.split(".")
		if len(split_f) < 2  or split_f[1] != "nii":
			continue

		img = nib.load(f)
		downed = downscale(img.get_data(), (df,df,df))
		# plotting.plot_img(downed_img)
		# plotting.plot_img(img)
		# plotting.show()
		
		controls.append(downed)		

	os.chdir("..")

	os.chdir(os.getcwd() + "\\patients")
	for f in os.listdir():
		print("downsample " + str(f))
		split_f = f.split(".")
		if len(split_f) < 2  or split_f[1] != "nii":
			continue

		img = nib.load(f)
		downed = downscale(img.get_data(), (df,df,df))
		# plotting.plot_img(downed_img)
		# plotting.plot_img(img)
		# plotting.show() 
		
		patients.append(downed)		

	os.chdir("..")

	return controls, patients


def random_train_test_indices(num_patients, num_controls, percent_test):
	"""
	Returns randomly selected sets of indices used to test and train the svm.

	Parameters:
		INT number of patients
		INT number of controls
		FLOAT percent of samples to be used for testing (50% --> 0.5)

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

	


def train_and_test_svm(patients, controls):
	"""
	Uses a defined portion of the samples provided to train a SVM, the rest of the samples
	are used to test the SVM.

	Parameters:
		LIST feature vectors representing the patients
		LIST feature vectors representing the controls

	Returns:
		INT accuracy result from testing
	"""

	percent_train = 0.5

	num_patients = len(patients)
	num_controls = len(controls)

	num_tests = 1000
	print("\n\nBeginning SVM Testing/Training")
	print(str(num_tests) + " tests")
	start = time.time()
	total_accuracy = 0

	for n in range(num_tests):
		p_train_idxs, p_test_idxs, \
		c_train_idxs, c_test_idxs = random_train_test_indices(num_patients, num_controls, percent_train)

		train_array = list()
		test_array = list()
		train_classes = list()
		test_classes = list()

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

		k_best = SelectKBest(k = 50)
		train_array = k_best.fit_transform(train_array, train_classes)

		clf = svm.SVC()
		clf.fit(train_array, train_classes)

		test_array = k_best.transform(test_array)
		guesses = clf.predict(test_array)

		num_correct = 0

		for n in range(len(test_array)):
			print(str(type(guesses)))
			actual = test_classes[n]
			guess = guesses[n]

			if actual == guess: 
				num_correct += 1


		accuracy = num_correct*100/len(test_array))
		total_accuracy += accuracy

	overall_accuracy = total_accuracy/num_tests

	print("SVM TIME: " + str(time.time()-start) + "\n\n")

	return overall_accuracy



if __name__ == "__main__":
	start = time.time()
	downsample_factor = 4
	controls, patients = load_and_downsample(downsample_factor)
	c_vectors, p_vectors = list(), list()

	count = 0
	for c in controls:
		print("COUNT: " + str(count))
		start = time.time()
		sobel_d, sobel_m = apply_sobel(c)
		co_vector = create_co_occurence_vector(sobel_d)
		c_vectors.append(co_vector)
		print("\nTOTAL TIME:" + str(time.time() - start) + "\n\n")
		count += 1

	for p in patients:
		print("COUNT: " + str(count))
		start = time.time()
		sobel_d, sobel_m = apply_sobel(p)
		co_vector = create_co_occurence_vector(sobel_d)
		p_vectors.append(co_vector)
		print("\nTOTAL TIME:" + str(time.time() - start) + "\n\n")
		count += 1


	accuracy = train_and_test_svm(p_vectors, c_vectors)
	print("ACCURACY: " + str(accuracy))
	print("TOTAL TIME: " + str(time.time() - start))

	
















