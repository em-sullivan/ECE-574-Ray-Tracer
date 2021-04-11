# Multi-Threading

The current method for multithreading is using OpenMP, since
it was relatively easy to use.

To change the amount of threads used, run:

'''
export OMP_NUM_THREADS=N
'''

Where N is the number of threads.

Update 4/11/2021

OpenMP implemented (It was just like two lines of code to render). It
so far makes rendering slightly slower. Cool.

