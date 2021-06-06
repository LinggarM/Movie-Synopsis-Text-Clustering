import numpy as np
import pandas as pd
import random as rd
from random import randint
import matplotlib.pyplot as plt

#menghitung nilai fitness suatu populasi
def fitness_func(population, time, priority, threshold):
    fitness = np.empty(population.shape[0])
    for i in range(population.shape[0]):
        S1 = np.sum(population[i] * priority)
        S2 = np.sum(population[i] * time)
        if S2 <= threshold:
            fitness[i] = S1
        else :
            fitness[i] = 0 
    return fitness.astype(int)
	
#menghitung nilai fitness suatu individu
def fitness_func_one(individual, time, priority, threshold):
    fitness = 0
    waktu_dibutuhkan = 0
    for i in range(len(individual)):
        if (individual[i] == 1):
            fitness = fitness + priority[i]
            waktu_dibutuhkan = waktu_dibutuhkan + time[i]
    if (waktu_dibutuhkan > threshold):
        fitness = 0
    return fitness
			
#membentuk populasi offspring
def generate_offsprings_func(population, sigma):
    offsprings = np.empty((population.shape))
    k = np.random.randint(0,10)
    for i in range(population.shape[0]):
        offsprings[i] = population[i]
    for i in range(population.shape[0]):
        for j in range (k):
          if (offsprings[i][j] == 0):
            offsprings[i][j] = 1
          else:
            offsprings[i][j] = 0
    return offsprings.astype(int)
	
#seleksi parent & offspring
def selection(population, offsprings, solutions_per_pop, time, priority, threshold):
	num_opponent = int (solutions_per_pop/2)
	opponent = []
	population_opponent = np.concatenate((population, offsprings.astype(int)), axis=0)
	#print("population_opponent :",population_opponent)
	#print("fitness_func : ",fitness_func(population_opponent, time, priority, threshold))
	for i in range(num_opponent):
		acak = rd.randint(0,len(population_opponent)-1)
		opponent.append(population_opponent[acak])
	jumlah_kemenangan = []
	for i in range(len(population_opponent)):
		jumlah = 0
		#print("pertandingan ke",i)
		for j in range(len(opponent)):
			#print(fitness_func_one(population_opponent[i], time, priority, threshold)," jika > dari", fitness_func_one(opponent[j], time, priority, threshold))
			if fitness_func_one(population_opponent[i], time, priority, threshold) > fitness_func_one(opponent[j], time, priority, threshold):
				jumlah = jumlah + 1
			#print("ya, lebih besar")
		jumlah_kemenangan.append(jumlah)
	#print(jumlah_kemenangan)
	hasil_rank = np.argsort(jumlah_kemenangan)
	index_akhir = hasil_rank[10:]
	selection_final = []
	for i in index_akhir:
		selection_final.append(population_opponent[i])
	return np.array(selection_final)


def kalkulasi(jumlah_tugas, nama_tugas, skala_prioritas, waktu_pengerjaan, waktu) :
	#Initialization
	item_number = np.arange(1,int(jumlah_tugas)+1)

	numpy_nama_tugas = np.array([i for i in nama_tugas.split(',')])

	numpy_skala_prioritas = np.array([int(i) for i in skala_prioritas.split(',')])
	value = numpy_skala_prioritas

	numpy_waktu_pengerjaan = np.array([int(i) for i in waktu_pengerjaan.split(',')])
	weight = numpy_waktu_pengerjaan

	threshold = int(waktu)

	#Default Initialization
	solutions_per_pop = 10
	num_generations = 50
	alpha = 0.1
	population_size = (solutions_per_pop, item_number.shape[0])

	initial_population_x = np.random.randint(2, size = population_size)
	initial_population_x = initial_population_x.astype(int)
	initial_population_sigma = np.random.randint(range(0,10), solutions_per_pop)
	
	parameters, fitness_history= [], []
	population = initial_population_x
	sigma = initial_population_sigma
	for i in range(num_generations):
		print("Iterasi ke-",i," :")
		print(population)
		fitness = fitness_func(population, weight, value, threshold)
		print("Nilai fitness : ",fitness)
		fitness_history.append(fitness)
		offsprings = generate_offsprings_func(population, sigma)
		#print("offsprings : ",offsprings)
		#print("sigma :",sigma)
		for j in range(len(sigma)):
			hasil_sigma = sigma[j] * alpha
			sigma[j] = hasil_sigma
		population = selection(population, offsprings, solutions_per_pop, weight, value, threshold)
		print("\n")
	index_max = np.argmax(population)
	individu_hasil = item_number * population[index_max]

	#Print hasil solusi terbaik
	hasil_tugas = []
	hasil_prioritas = 0
	hasil_waktu_yang_dibutuhkan = 0
	for i in individu_hasil :
		if (i!=0) :
		  hasil_tugas.append(numpy_nama_tugas[int(i)-1])
		  hasil_prioritas = hasil_prioritas + value[int(i)-1]
		  hasil_waktu_yang_dibutuhkan = hasil_waktu_yang_dibutuhkan + weight[int(i)-1]

	hasil_string = ""
	for i in hasil_tugas :
	  hasil_string = hasil_string + i + ","

	total = []
	total.append(hasil_string)
	total.append(hasil_prioritas)
	total.append(hasil_waktu_yang_dibutuhkan)
	
	return total