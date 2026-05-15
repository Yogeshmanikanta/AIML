import numpy as np

vector = np.array([1,2,3])

matrix = np.array([[1,2],[6,7]])

#print(vector)

#print(matrix)

A=np.array([[1,2,3],[4,5,6],[7,8,9]])
B=np.array([[7,8,9],[10,11,12],[13,14,15]])

# this one through error because of the shape of the array
#A=np.array([[1,2,3],[4,5,6],[7,8,9]])
#B=np.array([[7,8,9],[10,11,12]])

#print(A+B)

mul=np.dot(A,B)
print(mul)

#we can use the @ operator for the matrix multoplocation but it will also through error because of the shape of the array
print(A@B)

print(A.T)
#solution is to transpose the second matrix 
#mul1=np.dot(A,B.T)
#print(mul1)

