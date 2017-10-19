from  matplotlib import pyplot as plt
import numpy as np

"""
hello numpy
"""

print("random sample:")
x = np.random.random_sample((3,2)) # notice tuple
print(x)

###

print("multiply matrices")
a = np.array([3,3,3,3,3,3,3,3,3])
b = np.array([2,2,2,2,2,2,2,2,2])
c= a*b
print(c)

##

print("multiply matrices 2")
# array([[ 0,  1,  2,  3,  4],
#        [ 5,  6,  7,  8,  9],
#        [10, 11, 12, 13, 14]])
a = np.arange(15).reshape(3,5)
b = np.arange(15).reshape(3,5)
c= a*b
print(c)

###
print("dot product")
a = np.arange(10,110,10)
b = np.arange(5,55,5)
c = np.dot(a,b)
print(c)
###

x = []
for i in range(1,100):
	x.append( i**2);
plt.suptitle("Squares 1-100 suptitle")
plt.plot(x)
plt.title("Squares 1-100 title")
#plt.show() # works

###	

print("subtract number from matrix")
x = np.array([[0],
            [1],
            [1],
            [0]])
y = x - 10
print (y)

###

print("can't add matrices of different shape")
a = np.arange(1,13).reshape(3,4)
b = np.arange(1,13).reshape(4,3)
#c = a+b # error
#print(c)

###

print("mean")
a = np.array([2,2,2,2,4])
b = np.arange(1,13).reshape(3,4)
print(a)
print(b)
amean = np.mean(a)
bmean = np.mean(b)
print(amean)
print(bmean)

###

print("deriv x*(1-x)")
x = np.array([1,2,3,4])
y = x*(1-x)
print(y)

###

print("add matrices with different cols (if one array has shape n,1)")
a = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])
                
b = np.array([[0],
            [1],
            [1],
            [0]])
c = a+b
print(c)

###

print("add matrices with different cols 2")
a = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])
                
b = np.array([[0,1],
            [1,1],
            [1,1],
            [0,1]])

# c = a+b # ValueError: operands could not be broadcast together with shapes (4,3) (4,2)
#print(c)

###

# a = np.array([[1,1,1],
#             [1,1,1],
#             [1,1,1],
#             [1,1,1]])
print("multiply (4,3) matrix with (1,3)")
a = np.ones(shape=(4,3)) * 2
b = np.array([3,3,3])
c = np.dot(a,b)
print(a)
print(b)
print(c)

###

print("multiply (4,3) matrix with (3,4)")
a = np.ones(shape=(4,3)) * 2
b = np.ones(shape=(3,4)) * 3
c = np.dot(a,b)
print(a)
print(b)
print(c)

###

print("multiply (2,5) matrix with (3,4)")
a = np.ones(shape=(2,5)) * 2
b = np.ones(shape=(3,4)) * 3
#c = np.dot(a,b) #ValueError: shapes (2,5) and (3,4) not aligned: 5 (dim 1) != 3 (dim 0)
# print(a)
# print(b)
# print(c)

###

# important is A cols to match B rows
print("multiply (10,3) matrix with (3,25)")
a = np.ones(shape=(10,3)) * 2
b = np.ones(shape=(3,25)) * 3
c = np.dot(a,b)
print(a)
print(b)
print(c)