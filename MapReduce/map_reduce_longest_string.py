'''
Scaling your system by introducing better and faster hardware is called “Vertical Scaling”

We’ll break our code into two steps: 
1) compute the len of all strings (mapper)
2) select the max value. (reducer)

'''
import map, reduce


list_of_strings = ['abc', 'python', 'dima'] * 100

# stage 1
mapper = len

def reducer(p, c):
    if p[1] > c[1]:
        return p
    return c

%%time
#step 1
mapped = map(mapper, list_of_strings)
mapped = zip(list_of_strings, mapped)
#step 2:
reduced = reduce(reducer, mapped)
print(reduced)


# stage 2
data_chunks = chunkify(list_of_strings, number_of_chunks=30)
#step 1:
reduced_all = []
for chunk in data_chunks:
    mapped_chunk = map(mapper, chunk)
    mapped_chunk = zip(chunk, mapped_chunk)
    
    reduced_chunk = reduce(reducer, mapped_chunk)
    reduced_all.append(reduced_chunk)
    
#step 2:
reduced = reduce(reducer, reduced_all)


# stage 3, use Pool.map
from multiprocessing import Pool

pool = Pool(8)

data_chunks = chunkify(large_list_of_strings, number_of_chunks=8)
#step 1:
mapped = pool.map(mapper, data_chunks)
#step 2:
reduced = reduce(reducer, mapped)
print(reduced)




