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




def bin_sobel_3D(s_mags):
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
	# print("Making co-occurrence vector")
	start = time.time()
	# print("Sobel Matrix")
	# for i in range(len(matrix)):
	# 	print(matrix[i])

	#scans offets in a cube out from the current voxel. Neighbourhood size is the 
	#side length of that cube.
	neighbourhoodsize = 4
	num_offsets = (neighbourhoodsize*2) ** 3


	co_matrix = numpy.zeros((num_offsets,9,9))
	xdim,ydim,zdim = len(matrix),len(matrix[0]),len(matrix[0][0])

	#for each voxel in the scan
	for x in range(neighbourhoodsize, xdim - neighbourhoodsize):
		for y in range(neighbourhoodsize, ydim - neighbourhoodsize):
			for z in range(neighbourhoodsize, zdim - neighbourhoodsize):
				scanning_voxel = int(matrix.item(x,y,z))

				#for each offset per voxel
				offset_count = 0
				for a in range(-neighbourhoodsize, neighbourhoodsize):
					for b in range(-neighbourhoodsize, neighbourhoodsize):
						for c in range(-neighbourhoodsize, neighbourhoodsize):
							neighbour_voxel = int(matrix.item(x+a,y+b,z+c))
							prev_value = co_matrix.item((offset_count, scanning_voxel,neighbour_voxel))

							#increment the value by one every co-occurence
							co_matrix.itemset((offset_count, scanning_voxel, neighbour_voxel), prev_value + 1)

							offset_count += 1
							



	# print("Co-occurence matrix")
	# for i in range(len(co_matrix)):
	# 	print(co_matrix[i])

	#reshape matrix to one dimensional vector
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
		NUMPY.NDARRAY matrix containing the bin value for each voxel
	Returns:
		NUMPY.ARRAY vector of size num_features
	"""
	# print("Making co-occurrence vector")
	start = time.time()


	#scans offets in a cube out from the current voxel. Neighbourhood size is the 
	#side length of that cube.
	neighbourhoodsize = 4
	num_offsets = (neighbourhoodsize*2) ** 2


	co_matrix = numpy.zeros((num_offsets,9,9))
	xdim,ydim = len(matrix),len(matrix[0])

	#for each voxel in the scan
	for x in range(neighbourhoodsize, xdim - neighbourhoodsize):
		for y in range(neighbourhoodsize, ydim - neighbourhoodsize):

			scanning_voxel = int(matrix.item(x,y))

			#for each offset per voxel
			offset_count = 0
			for a in range(-neighbourhoodsize, neighbourhoodsize):
				for b in range(-neighbourhoodsize, neighbourhoodsize):

					neighbour_voxel = int(matrix.item(x+a,y+b))
					prev_value = co_matrix.item((offset_count, scanning_voxel,neighbour_voxel))

					#increment the value by one every co-occurence
					co_matrix.itemset((offset_count, scanning_voxel, neighbour_voxel), prev_value + 1)

					offset_count += 1
							

	#reshape matrix to one dimensional vector
	co_vector = co_matrix.reshape(9 * 9 * num_offsets)

	# print("TIME: " + str(time.time() - start))


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


	# print("Applying Sobel")
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
	# print("Average: " + str(numpy.average(sobel_magnitudes[0])))
	# print("Max: " + str(numpy.max(sobel_magnitudes[0])))

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
					bin = bin_sobel_3D((mx, my, mz))
					sobel_directions.itemset(coord,bin)
				else:
					sobel_directions.itemset(coord,0)



	# print("TIME: " + str(time.time()-start))

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


	# print("Applying Sobel")
	start = time.time()
	dx = ndimage.sobel(img,0)
	dy = ndimage.sobel(img,1)

	shape = dx.shape

	sobel_magnitudes = numpy.ndarray(shape = shape)
	sobel_directions = numpy.ndarray(shape = shape)
	mag_total = 0
	num_mags = 0

	for x in range(shape[0]):
		for y in range(shape[1]):

			coord = tuple((x,y))
			mx = dx.item(coord)
			my = dy.item(coord)

			mag = math.sqrt((mx**2) + (my**2))
			sobel_magnitudes.itemset(coord,mag)
			mag_total += mag
			num_mags += 1

				
	max_mag = numpy.max(sobel_magnitudes[0])
	avg_mag = mag_total / num_mags
	# print("Average: " + str(numpy.average(sobel_magnitudes[0])))
	# print("Max: " + str(numpy.max(sobel_magnitudes[0])))

	# sobel_threshold = max_mag/10
	sobel_threshold = avg_mag* 0.05
	# sobel_threshold = 2
	num_accepted = 0
	num_rejected = 0

	for x in range(shape[0]):
		for y in range(shape[1]):

			coord = tuple((x,y))
			mx = dx.item(coord)
			my = dy.item(coord)

			mag = sobel_magnitudes.item(coord)

			if mag >= sobel_threshold:
				num_accepted += 1
				bin = bin_sobel_2d(mx, my)
				sobel_directions.itemset(coord,bin)
			else:
				num_rejected += 1
				sobel_directions.itemset(coord,0)
	print("Num rejected: %d" % (num_rejected))
	print("Num num_accepted: %d" % (num_accepted))



	# print("TIME: " + str(time.time()-start))

	return sobel_directions, sobel_magnitudes


def bin_sobel_2d(mx, my):
	if mx > 0 and my > 0 and mx > my:
		return 1

	if mx > 0 and my > 0 and mx < my:
		return 2

	if mx < 0 and my > 0 and abs(mx) < my:
		return 3

	if mx < 0 and my > 0 and abs(mx) > my:
		return 4

	if mx < 0 and my < 0 and abs(mx) > abs(my):
		return 5

	if mx < 0 and my < 0 and abs(mx) < abs(my):
		return 6

	if mx > 0 and my < 0 and mx < abs(my):
		return 7

	if mx > 0 and my < 0 and mx > abs(my):
		return 8

	else:
		return 0


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
	# print("Downsampling by factor of " + str(df))
	controls = list()
	patients = list()

	os.chdir(os.getcwd() + "\\controls")
	for f in os.listdir():
		# print("downsample " + str(f))
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
		# print("downsample " + str(f))
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

	


def train_and_test_svm(patients, controls, num_features):
	"""
	Uses a defined portion of the samples provided to train a SVM, the rest of the samples
	are used to test the SVM.

	Parameters:
		LIST feature vectors representing the patients
		LIST feature vectors representing the controls

	Returns:
		INT accuracy result from testing
	"""
	p_class = [1 for x in range(len(patients))]
	c_class = [0 for x in range(len(controls))]
	all_classes = p_class + c_class

	all_samples = patients + controls
	# k_best = SelectPercentile(score_func = chi2, percentile = 5)
	k_best = SelectKBest(score_func = chi2, k = num_features)
	k_best.fit(all_samples, all_classes)

	percent_train = 0.8

	num_patients = len(patients)
	num_controls = len(controls)

	num_tests = 200
	# print("\n\nBeginning SVM Testing/Training")
	# print(str(num_tests) + " tests")
	start = time.time()
	total_accuracy = 0

	for n in range(num_tests):
		p_train_idxs, p_test_idxs, \
		c_train_idxs, c_test_idxs = random_train_test_indices(num_patients, num_controls, percent_train)

		# print(p_train_idxs)
		# print(p_test_idxs)
		# print(c_train_idxs)
		# print(c_test_idxs)

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



		train_array = k_best.transform(train_array)
		test_array = k_best.transform(test_array)
		

		clf = svm.LinearSVC()
		clf.fit(train_array, train_classes)
		guesses = clf.predict(test_array)

		num_correct = 0

		for n in range(len(test_array)):
			actual = test_classes[n]
			guess = guesses[n]
			# print("actual: " + str(actual))
			# print("guess: " + str(guess))

			if actual == guess: 
				num_correct += 1
				# print("CORRECT!\n")


		accuracy = num_correct*100/len(test_array)
		# print("test accuracy: " + str(accuracy))
		total_accuracy += accuracy

	overall_accuracy = total_accuracy/num_tests

	# print("SVM TIME: " + str(time.time()-start) + "\n\n")

	return overall_accuracy

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
	# print("Downsampling by factor of " + str(df))
	controls = list()
	patients = list()

	os.chdir(os.getcwd() + "\\controls")
	for f in os.listdir():
		# print("downsample " + str(f))
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
		# print("downsample " + str(f))
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

def apply_mask(img, mask):
	img_data = img.get_data()
	mask_data = mask.get_data()

	mask = list()
	masked_img = list()
	slices = set()

	for x in range(mask_data.shape[0]):
		for y in range(mask_data.shape[1]):
			if numpy.sum(mask_data[x][y]) > 0:
				mask.append(mask_data[x][y])
				masked_img.append(img_data[x][y])


	for x in range(len(masked_img)):
		for y in range(len(masked_img[0])):
			masked_img[x][y] = masked_img[x][y] * mask[x][y]

	masked_img = numpy.array(masked_img)
	return masked_img

def load_masked_downed_nii_from_directory(dir_name, num_samples, df):
	os.chdir(os.getcwd() + "\\" + dir_name)
	all_files = os.listdir()

	sampleFiles = list()
	samples = list()
	for f in all_files:
		f_split = f.split(".")

		if len(f_split) < 2 or f_split[1] != "nii":
			continue
		else:
			sampleFiles.append(f)

	if str(num_samples).lower() != "all":
		while len(sampleFiles) > num_samples:
			shuffle(sampleFiles)
			sampleFiles.pop()

	for f in sampleFiles:
		split_f = f.split(".")
		img = nib.load(f)
		mask = nib.load(os.getcwd() + "\\masks\\" + split_f[0] + "Mask.nii")
		# print("Mask: " +  split_f[0] + "Mask.nii")
		masked_img = apply_mask(img, mask)


		downed = downscale(masked_img, (df,df))
		samples.append(downed)
	os.chdir("..")

	return samples







def cohog(downsample_factor, num_controls, num_patients, num_features):
	start = time.time()


	controls = load_masked_downed_nii_from_directory('controls',num_controls,downsample_factor)
	patients = load_masked_downed_nii_from_directory('patients',num_patients,downsample_factor)

	c_vectors, p_vectors = list(), list()


	count = 0
	for c in controls:
		# print("COUNT: " + str(count))
		start = time.time()
		sobel_d, sobel_m = apply_sobel_2d(c)
		co_vector = create_co_occurence_vector_2d(sobel_d)
		c_vectors.append(co_vector)
		# print("\nTOTAL TIME:" + str(time.time() - start) + "\n\n")
		count += 1

	for p in patients:
		# print("COUNT: " + str(count))
		start = time.time()
		sobel_d, sobel_m = apply_sobel_2d(p)
		co_vector = create_co_occurence_vector_2d(sobel_d)
		p_vectors.append(co_vector)
		# print("\nTOTAL TIME:" + str(time.time() - start) + "\n\n")
		count += 1



	accuracy = train_and_test_svm(p_vectors, c_vectors, num_features)
	num_samples = len(c_vectors) + len(p_vectors)
	return accuracy, len(c_vectors), len(p_vectors)

if __name__ == "__main__":


	d_low = 1
	d_high = 8
	num_p_range = [10,11,1]
	num_c_range = [2,4,1]
	num_features_range = [1,62,10]




	if 'cohogResults.xlsl' in os.listdir():
		os.remove('cohogResults.xlsl')

	final_msg = ""

	workbook = xlsxwriter.Workbook('cohogResults.xlsl')
	worksheet = workbook.add_worksheet()


	count = 1
	test_num = 1
	for num_patients in range(num_p_range[0],num_p_range[1],num_p_range[2]):
		num_controls = num_patients
		worksheet.write(0 + 7*count,0,"Downsampled")
		worksheet.write(1 + 7*count,0,"Accuracy")
		worksheet.write(2 + 7*count,0,"Number of Samples")
		worksheet.write(3 + 7*count,0,"Number of Controls")
		worksheet.write(4 + 7*count,0,"Number of Patients")
		worksheet.write(5 + 7*count,0,"Time")
		worksheet.write(6 + 7*count,0,"Num Features")

		col = 1

		for x in range(d_low,d_high+1):
			for num_features in range(num_features_range[0],num_features_range[1],num_features_range[2]):
				for n in range(10):
					print("TEST %d" % (test_num))
					start = time.time()
					acc, num_c, num_p = cohog(x, num_controls, num_patients, num_features)
					t = time.time()-start

					worksheet.write(0 + 7*count,col,x)
					worksheet.write(1 + 7*count,col,acc)
					worksheet.write(2 + 7*count,col,num_c + num_p)
					worksheet.write(3 + 7*count,col,num_c)
					worksheet.write(4 + 7*count,col,num_p)
					worksheet.write(5 + 7*count,col,t)
					worksheet.write(6 + 7*count,col,num_features)

					col += 1
					test_num += 1



					result = "Downsampled: %d \nAccuracy: %f \nTime: %f \nNum Samples: %d \nNum Controls: %d \nNum Patients: %d \nNum Features: %d\n\n" % (x,acc,t,num_p + num_c,num_c,num_p,num_features)
					print(result)
		count += 1

	workbook.close()

	print(final_msg)

	
















