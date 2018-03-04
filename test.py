#imports
import numpy
from random import *
from cProfile import Profile
import pstats

prof = Profile()
prof.enable()
#file
f = "had12.dat"

#settings
pop_size=100
gen=100
px=0.7
pm=0.01
tour=5
n = int(numpy.genfromtxt(open(f, "r"), dtype="int", max_rows=1))

#read from file
ddd = numpy.genfromtxt(open(f, "r"), skip_header=2, dtype="int", max_rows=12)
fff = numpy.genfromtxt(open(f, "r"), skip_header=15, dtype="int", max_rows=12)

#best solution
sol = [3, 10, 11, 2, 12, 5, 6, 7, 8, 1, 4, 9]

#check
#print(n)
#print(dist)
#print(flow)

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

#Calculate S (sum all fitnesses for current population)
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
  x_point = randint(1, n)
  mut = random()
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

  if(mut <= pm):
    m1 = randint(0, 11)
    m2 = randint(0, 11)
    m11 = child[m2]
    m12 = child[m1]
    child[m1] = m11
    child[m2] = m12
  return child

#GA
def survival_of_the_fittest(pop, fittest, counter):
  s = calculate_s(pop)
  new_pop = []
  new_pop.append(fittest)
  while(len(new_pop)<= 100):
    c1 = tournament(pop)
    c2 = tournament(pop)
    offspring = crossover(c1, c2)
    new_pop.append(offspring)
  fittest = select_fittest(new_pop)
  c = counter + 1
  return (new_pop, fittest, c)

#sxsx = numpy.array([8, 10, 12, 5, 1, 6, 11, 2, 3, 4, 7, 9])
#print(numpy.where(sxsx == 12)[0][0])
#print(sxsx)



#Start
population = generate_population(pop_size)
fittest = select_fittest(population)

counter = 1
while(counter < 100):
  p = survival_of_the_fittest(population, fittest, counter)
  population = p[0]
  fittest = p[1]
  counter = counter + 1

res = select_fittest(population)
print('Result:', res)
print('Fitness:', evaluate_fitness(res))


prof.disable()
ps = pstats.Stats(prof).sort_stats('tottime')
ps.print_stats()