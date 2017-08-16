""" 
Plotting

from nilearn import plotting

plotting.plot_img('C:\\Users\\jyuri\\Desktop\\ALS\\cohog2\\controls\\Control0.nii')
plotting.show()
"""

"""
3D and 4D niimgs: handling and visualizing

from nilearn import datasets
print('Datasets are stored in : %r' % datasets.get_data_dirs())

tmap_filenames = datasets.fetch_localizer_button_task()['tmaps']
print(tmap_filenames)

tmap_filename = tmap_filenames[0]

from nilearn import plotting
plotting.plot_stat_map(tmap_filename)

plotting.plot_stat_map(tmap_filename, threshold=3)
plotting.show()

"""

"""
Using the SVM from sklearn

from sklearn.svm import SVC
svc = SVC(kernel = 'linear', C = 1)

from sklearn import datasets
digits = datasets.load_digits()
data = digits.data
labels = digits.target

svc.fit(data[:-10], labels[:-10])

print(svc.predict(data[-10:]))
print(labels[-10:])
"""

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
	print("Making co-occurrence vector")
	start = time.time()

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
							# print("Scanning: " + str(scanning_voxel))
							# print("neighbour_voxel: " + str(neighbour_voxel))
							# print("offset_num: " + str(offset_count))
							prev_value = co_matrix.item((scanning_voxel,neighbour_voxel,offset_count))

							co_matrix.itemset((scanning_voxel, neighbour_voxel, offset_count), prev_value + 1)
							offset_count += 1



	co_vector = co_matrix.reshape(9 * 9 * num_offsets)
	print("TIME: " + str(time.time() - start))

	return co_vector

def apply_sobel(img):


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

	

	controls = list()
	patients = list()


	os.chdir(os.getcwd() + "\\controls")
	for f in os.listdir():
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
		split_f = f.split(".")
		if len(split_f) < 2  or split_f[1] != "nii":
			continue

		img = nib.load(f)
		downed = downscale(img.get_data(), (df,df,df))
		# plotting.plot_img(downed_img)
		# plotting.plot_img(img)
		# plotting.show() new
		
		patients.append(downed)		

	os.chdir("..")

	return controls, patients

def random_train_test_indices(num_patients, num_controls, percent_test):

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



	percent_train = 0.5

	num_patients = len(patients)
	num_controls = len(controls)

	num_tests = 1
	print("\n\nBeginning SVM Testing/Training")
	print(str(num_tests) + " tests")
	start = time.time()

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

			print("ACTUAL:" + str(actual))
			print("GUESS:" + str(guess) + "\n\n")

			if actual == guess: 
				num_correct += 1

		print("Accuracy: " + str(num_correct*100/len(test_array)))

		





	print("SVM TIME: " + str(time.time()-start) + "\n\n")



if __name__ == "__main__":
	downsample_factor = 5
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


	train_and_test_svm(p_vectors, c_vectors)

	# ptr, pte, ctr, cte = random_train_test_indices(10, 10)
	# print(ptr)
	# print(pte)
	# print(ctr)
	# print(cte)
	
















