import numpy as np
from itertools import product, combinations

def get_index_combinations(a,b):
	# if(a==b):
	# 	return [(a[i[0]],b[i[1]]) for i in combinations(np.arange(len(a)),2)]
	# else:
	return [(i[0],i[1]) for i in product(a,b)]

def get_index_combinations_a_equals_b(a,b):
	# if(a==b):
	return [(a[i[0]],b[i[1]]) for i in combinations(np.arange(len(a)),2)]
	# else:
	#     return [(i[0],i[1]) for i in product(a,b)]


'''
a = [1,2,3,4]
b = [5,6,7,8]
ss, sn, nn = calculate(a,a), calculate(a,b), calculate(b,b)
print len(ss),len(sn), len(nn)
print ss
print sn
print nn
'''
