import numpy as np

# Conjugate Gradient Method
def congradmet(A, b, x0, n, tol): 


    r0 = b - np.matmul(A,x0)
    d0 = r0.copy() 

    for i in range(n):

        if np.linalg.norm(r0) < tol:
            print(f'Converged after {i} iterations.')
            break
        else:
            a = np.dot(r0.T,r0) / (np.dot(d0.T,np.matmul(A,d0)))
            x1 = x0 + a * d0
            r1 = r0 - a * np.matmul(A,d0)
            B = np.dot(r1.T, r1) / np.dot(r0.T, r0)
            d1 = r1 + B * d0
            r0 = r1
            d0 = d1
            x0 = x1
    return x0


#TEST on 2.29

A = np.array([[2, 2], [2, 5]]) 
b = np.array([6, 3]).reshape(-1, 1)
x0 =  np.zeros((2, 1)) 

solution = congradmet(A,b,x0, 2,0 )

# Final Solution

print(f"a.) Impliment the Conjugate Gradient algorithm ")
print(f' I created a function that takes in matrix A, solution vector b, an intial guess, a number n, and a tolerance level.')
print(f'\n')
print("b.) Use Example 2.29 to check your code")
print(f'\n')
print("Solution x:")
print(solution)
print(f'\n')
print(f'This solution matches the one in the textbook, which gives me confidence in my code.')
print(f'\n')

# Test Code on the Hilbert Matrix
# Define Hilbert Matrix and b vector

H = np.array([[1, 1/2, 1/3, 1/4], [1/2, 1/3, 1/4, 1/5], [1/3, 1/4, 1/5, 1/6], [1/4, 1/5, 1/6, 1/7]])
b = np.array([1,1,1,1]).reshape(-1,1)
x0 = np.zeros((4, 1)) 

solutionc = congradmet(H, b, x0, 4, 0)

print('For the case n =4 solve Hx=b, where H is the nxn Hilbert Matrix')
print("and b is the vector of all ones.")
print(f'\n')
print("Solution x:")
print(solutionc)
print(f'\n')

# Compute residual vector
residual = b - np.matmul(H, solutionc)

# Compute backward error using inf norm
backwarderror_inf = np.linalg.norm(residual, ord=np.inf)

# Print the backward error
print("Backward Error (Infinity Norm):")
print(backwarderror_inf)
print(f'\n')
print("We can find the forward error. To do so we must find the difference")
print("between the computed solution and the actual.")
print(f'\n')

# True solution 
x_true = np.linalg.solve(H, b)

# Forward error: infinity norm of the difference
forwarderror_inf = np.linalg.norm(solutionc - x_true, ord=np.inf)

# Print forward error
print("Forward Error (Infinity Norm):")
print(forwarderror_inf)

#Repeat for hilbert matrix with 8 columns.
n = 8
H = np.array([[1/(i+j-1) for j in range(1, n+1)] for i in range(1, n+1)])
#print(H)
b = np.ones((n, 1))
x0 = np.zeros((n, 1))

solutiond = congradmet(H, b, x0, n, 0)

print("\nFor the case n = 8, solve Hx = b, where H is the nxn Hilbert Matrix")
print("and b is the vector of all ones.\n")
print("Solution x:")
print(solutiond)

# find the residual vector 
residual = b - np.matmul(H, solutiond)

# Backward error using infinity norm
backwarderror_inf = np.linalg.norm(residual, ord=np.inf)
print("\nBackward Error (Infinity Norm):")
print(backwarderror_inf)

print('\nResults Explained:')
print("I am confident in my answers for parts b and c; however, I did expect")
print("my backward error to be smaller for this problem.")
print("\nThe likely cause of this could be the conditioning of the Hilbert matrix,")
print("it may amplify errors and leads to larger backward errors.")
