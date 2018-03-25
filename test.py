#imports
import numpy
from random import *
from cProfile import Profile
import pstats
import csv

#profiler to see how the code performs
#prof = Profile()
#prof.enable()

#file
test_run_no = 7
f = "./data/had12.dat"
out_file = "./out/had12_roulette_"+str(test_run_no)+".csv"

#settings
pop_size=100
gen=100
px=0.7
pm=0.01
tour=5
elitism = True
n = int(numpy.genfromtxt(open(f, "r"), dtype="int", max_rows=1))

#read from file
ddd = numpy.genfromtxt(open(f, "r"), skip_header=2, dtype="int", max_rows=n)
fff = numpy.genfromtxt(open(f, "r"), skip_header=n+3, dtype="int", max_rows=n)

#helpers
def inv_dict_from_list(list):
  invd = dict()
  for i in range(n):
    invd[list[i]] = i
  return invd

def dict_from_list(list):
  dicti = dict()
  for i in range(n):
    dicti[i] = list[i]
  return dicti

flow = dict_from_list(fff)
dist = dict_from_list(ddd)

for i in range(n):
  flow[i] = dict_from_list(flow[i])
for i in range(n):
  dist[i] = dict_from_list(dist[i])

#quality check of individual chromosome
def evaluate_fitness(chrom):
  sum = 0
  invdict = inv_dict_from_list(chrom)
  #print(invdict)
  for i in range(n):
    for j in range(n):
      if(i != j):
        sum = sum + flow[i][j] * dist[invdict[i+1]][invdict[j+1]]
  return sum

#select best chromosome from current population
def select_fittest(pop):
  chrom = (0, evaluate_fitness(pop[0]))
  for i in range(0, pop_size):
    ff = evaluate_fitness(pop[i])
    if(ff < chrom[1]):
      chrom = (i, ff)
  return pop[chrom[0]]

#select worst for stats
def select_least(pop):
  chrom = (0, evaluate_fitness(pop[0]))
  for i in range(0, pop_size):
    ff = evaluate_fitness(pop[i])
    if(ff > chrom[1]):
      chrom = (i, ff)
  return pop[chrom[0]]


#Calculate S (sum all fitnesses for current population) for roulette
def calculate_s(pop):
  s = 0
  for chrom in pop:
    s = s + evaluate_fitness(chrom)
  return s

#Roulette selection
def roulette(pop, s):
  r = sample(range(s), k=1)
  c = 0
  cond = False
  sum = 0
  while(cond != True and c < pop_size):
    sum = sum + evaluate_fitness(pop[c])
    if(sum > r):
      cond = True
    else:
      c = c + 1
  return pop[c]

#Tournament
def tournament(pop):
  tourn = sample(range(pop_size), k=tour)
  fit = dict()
  fit_i = 0
  for n in tourn:
    fit[n] = evaluate_fitness(pop[n])
  sorted_x = sorted(fit.items())
  llll = []
  for k, v in sorted_x:
    llll.append(k)
  fit_i = llll[0]
  return pop[fit_i]


#first population
def generate_population(size):
  population = []
  for i in range(0, size):
    population.append(sample(range(1, n+1), k=n))
  return population

#crossover
def crossover(chrom1, chrom2):
  if(random() <= px):
    x_point = randint(1, n)
    child = []
    for i in range(0, x_point):
      child.append(chrom1[i])
    condi = False
    for v in chrom2:
      for vv in child:
        if(v == vv):
          condi = True
      if(condi):
        condi = False
      else:
        child.append(v)
        condi = False  
    return mutate(child)
  else: 
    return chrom1

def mutate(child):
  mut = random()
  if(mut <= pm):
      m1 = randint(0, n-1)
      m2 = randint(0, n-1)
      m11 = child[m2]
      m12 = child[m1]
      child[m1] = m11
      child[m2] = m12
  return child


#With Tournament
def survival_of_the_fittest_t(pop, fittest, counter):
  new_pop = []
  if(elitism == True):
    new_pop.append(fittest)
    new_pop.append(mutate(fittest)) #mutates or doesn't
  while(len(new_pop)<= pop_size):
    c1 = tournament(pop)
    c2 = tournament(pop)
    offspring = crossover(c1, c2)
    new_pop.append(offspring)
  fittest = select_fittest(new_pop)
  c = counter + 1
  return (new_pop, fittest, c)

#With Roulette
def survival_of_the_fittest_r(pop, fittest, counter):
  s = calculate_s(pop) #only for roulette
  new_pop = []
  if(elitism == True):
    new_pop.append(fittest)
    new_pop.append(mutate(fittest)) #mutates or doesn't
  while(len(new_pop)<= pop_size):
    c1 = roulette(pop, s)
    c2 = roulette(pop, s)
    offspring = crossover(c1, c2)
    new_pop.append(offspring)
  fittest = select_fittest(new_pop)
  c = counter + 1
  return (new_pop, fittest, c)

#start
#prepare csv
writer = csv.writer(open(out_file, 'w'), delimiter=',', quoting = csv.QUOTE_NONE)

writer.writerow(["Plik: " + f])

#Run program
while (tour <= 5):
  writer.writerow([])
  writer.writerow(["Parameters:"])
  writer.writerow(["pop_size="+ str(pop_size), "gen="+ str(gen), "Px="+ str(px), "Pm"+str(pm), "tour=" + str(tour), "elitism=True"])
  writer.writerow([])
  writer.writerow(["<nr_pokolenia", " najlepsza_ocena", " srednia_ocen", " najgorsza_ocena>"])
  runs = 1
  counter = 1
  best_results = []
  while(runs <= 10):
    population = generate_population(pop_size)
    fittest = select_fittest(population)
    writer.writerow([])
    writer.writerow([str(runs) + " uruchomienie"])
    print(str(runs) + " uruchomienie")
    while(counter <= gen):
      meany = calculate_s(population)/pop_size
      ssss = "<" + str(counter), str(evaluate_fitness(fittest)), str(int(meany)), str(evaluate_fitness(select_least(population))) + ">"
      print(ssss)
      writer.writerow(ssss)
      p = survival_of_the_fittest_r(population, fittest, counter)
      population = p[0]
      fittest = p[1]
      
      counter = counter + 1
    counter = 1
    runs = runs + 1
    res = select_fittest(population)
    res_res = evaluate_fitness(res)
    besty = "Best: " + "[" + " ".join(map(str, res)) + "]", str(res_res)
    best_results.append(besty)
    writer.writerow(besty)
    print('Result:', res)
    print('Fitness:', res_res)

  writer.writerow([])
  writer.writerow(["Najlepsze wyniki"])
  for ro in best_results:
    print(ro)
    writer.writerow(ro)

  runs = 1
  counter = 1
  best_results = []
  tour = tour + 5

#prof.disable()
#ps = pstats.Stats(prof).sort_stats('tottime')
#ps.print_stats()