# Naive TLS via SVD
import time
#From Wikipedia page
import numpy as np
import scipy
from scipy import linalg as spla
#global flag
#flag=0

def tls(data,m): # default
	Y = data[m]
	X = data[0:m]
	n_pairs       = len(Y)#number of x,y data pairs
	x_width       = np.shape(X)[0]          # n is the width of X (X is m by n)
	Z       = data.transpose()             # Z is X augmented with Y.
	U, S, V = np.linalg.svd(Z,full_matrices=True)           # find the SVD of Z.
	error = S[len(S)-1] # ERROR= smallest singular value of data matrix
	V = np.transpose(V)
	#slice= V[1:3,1:3]
	VXY     = V[0:x_width,x_width:m+1]     # Take the block of V consisting of the first n rows and the n+1 to last column
	VYY     = V[x_width:m+1,x_width:m+1] # Take the bottom-right block of V.
	VYY_inv = np.linalg.inv(VYY) # Invert VYY
	B       = -np.dot(VXY,VYY_inv) # B = VXY * VYY_inv
	# VXY_YY =  V[0:m+1,x_width:m+1]
	# error = np.dot(Z,VXY_YY) # 
	# error = np.dot(error,VXY_YY.transpose())
	return B, error


#########################################################

def soft_threshold(M,mu_inv):
  U,S,V = np.linalg.svd(M,full_matrices=True)
  V=V.transpose()
  eps=0.0001
  for i in range(len(S)):
    ##print S[i], mu_inv
    if(S[i] - mu_inv> eps ):
      S[i] = S[i] - mu_inv
    else:  
      S[i] = 0
  S_new = np.zeros((np.shape(U)[0],np.shape(V)[0]))
  np.fill_diagonal(S_new,S)
  A_new = np.dot(np.dot(U,S_new) , V.transpose())
  return A_new

def soft_threshold_NN(Z, E_k, mu_k, lambda_k ): #X= A_bar
  mu_inv= 1.0/(1.0*mu_k)
  M=   np.add(np.subtract(Z,E_k) , mu_inv*lambda_k)
  return soft_threshold(M,mu_inv)

def new_Error(alpha, mu_k, lambda_k, Z, A):
  Error_new= (1/(1.0*(2*alpha + mu_k)))*(np.add (lambda_k , mu_k* np.subtract(Z,A)))
  #Project error onto linear constraints
 # #print"returning norm of", np.linalg.norm(Error_new)
  return Error_new

def NN_ALM(alpha, Z, n, m, speed, struc):
  lambda_k=E =A = np.zeros((n,m+1)) #Initiliaze lagrange multiplier and guesses for variables to 0 matrix
  mu_init=1.05
  mu_k=mu_init
  #Z=data.transpose()
  step_A=step_E=np.matrix(np.identity(n))
  k=0
  eps= 0.00001
  while ( np.linalg.norm(step_E) > eps and k<10000):
    A_new = soft_threshold_NN(Z,E,mu_k,lambda_k)
    step_A= np.subtract(A, A_new)
    A=A_new
    E_new = new_Error(alpha, mu_k, lambda_k,Z, A)
    step_E = np.subtract(E, E_new)
    E=E_new
    if(struc==1):
	    for i in range(np.shape(E)[0]):
	    	for j in range(i):
	    		E[i,j]=0 # Projection
    h = np.subtract (np.subtract(Z,A),E)
    lambda_k = lambda_k + mu_k*h
    if(speed==0):
    	mu_k= mu_k*mu_init
    elif(speed==1):
    	mu_k= mu_k+mu_init
    k=k+1
  return A,E,k
##########RW_NN###############

def RW_NN_ALM(alpha, Z, n, m, W1, W2,flag,struc):
	D_rec=lambda_k_1 = lambda_k_2 =E =A = D = np.zeros((n,m+1)) #Initiliaze lagrange multiplier and guesses for variables to 0 matrix
	W1_sq_inv = np.linalg.inv (np.dot(W1, W1))
	W2_sq = np.dot(W2, W2)
	mu_init=1.05
	mu_k=mu_init
	gamma=1
	k=0
	eps= 0.0001
	step_A=step_E=np.matrix(np.identity(n))
	E=np.zeros((n,m+1))
	while (k<700):# and np.linalg.norm(step_E) > eps and np.linalg.norm(step_A) > eps ):
		WAW = np.dot(np.dot(W1,A), W2)
		D_new = soft_threshold(np.subtract(WAW, lambda_k_2* (1/(1.0*mu_k)) ) , 1/(1.0*mu_k)  )
		step_D= np.subtract(D, D_new)
		D=D_new
		E_new = new_Error(alpha, mu_k, lambda_k_1,Z, A)
		
		# for i in range(np.shape(E)[0]):
		# 	for j in range(np.shape(E)[1]):
		# 		E_new[i][j]=soft_threshold(np.subtract(np.subtract(Z,A),1/(1.0*mu_k)*lambda_k_1), alpha/mu_k)[i][j]
		E_new=np.matrix(E_new)
		step_E = np.subtract(E, E_new)		
		E=E_new
		if(struc==1):
		    for i in range(np.shape(E)[0]):
		    	for j in range(i):
		    		E[i,j]=0 # Projection

		RHS = np.add ( (1/(1.0*mu_k))*np.add(lambda_k_1, np.dot( np.dot(W1,lambda_k_2 ),W2)), np.add (np.subtract(Z, E) , np.dot( np.dot(W1,D),W2)) )
		A_new= spla.solve_sylvester(W1_sq_inv, W2_sq, np.dot(W1_sq_inv, RHS))
		#T1 = np.dot(np.dot(W1, np.add (np.subtract(np.subtract(Z, E) , A) , 1/(1.0*mu_k)*lambda_k_1) ), W2 )
		#T2 = np.add (np.subtract(D,WAW) , 1/(1.0*mu_k)*lambda_k_2)
		#A_new = A + gamma*np.add(T1,T2)
		step_A = np.subtract(A, A_new)
		A=A_new
		lambda_k_1 =  np.add(lambda_k_1, mu_k*(np.subtract(np.subtract(Z, A), E)))
		lambda_k_2 = np.add(lambda_k_2, mu_k*(np.subtract(D, WAW)))
		mu_k= mu_k*mu_init
		k=k+1		
	# 	U,S,V=np.linalg.svd(A)
	# 	s=S[len(S)-1]
	# 	A=soft_threshold(A,s)
	
	if (flag==1):
		print "alpha",alpha,"iterations",k,"error A", np.linalg.norm(A), "error norm E", np.linalg.norm(E),"rank A", np.linalg.matrix_rank(A)
	WAW = np.dot(np.dot(W1,np.subtract(Z,E)), W2)
	return A,E,flag,WAW, np.subtract(Z,E)


def RW_NN( Z, n, m, W1, W2, delta,flag,eps,struc):
	A,E=np.matrix(np.identity(n)), np.matrix(np.identity(n))
	#alpha= 1/(1/4*(error)**2)
	#alpha= RW_NN_bin_search_answer(10, 0, 0.001, Z, n, m)
	#A,E = NN_ALM(alpha_NN, Z, n, m)
	itr=0
	for i in range (3):
		alpha, new_itr=RW_NN_bin_search_answer(5000, 0, eps, Z, n, m, W1, W2,0,struc,0)	
		itr=itr+new_itr
		#print i
		A,E,flag, WAW, D = RW_NN_ALM(alpha, Z, n, m, W1, W2,1,0)
		#print"My flag is" , flag
		U,S,V=np.linalg.svd(A)
		s=S[len(S)-1]
		A=soft_threshold(A,s)
		WAW = np.dot(np.dot(W1,A) , W2)
		U,S,V = np.linalg.svd(WAW, full_matrices= False) # See if this makes sense
		V=V.transpose()
		W1_inv= np.linalg.inv(W1)
		W2_inv= np.linalg.inv(W2)
		dimension= min (np.shape(U)[0], np.shape(V)[0])
		S_new = np.zeros((np.shape(U)[0],np.shape(U)[0]))
		np.fill_diagonal(S_new,S)
		Y_inner = np.dot(np.dot(U,S_new) , U.transpose())
		Y=np.dot (np.dot(W1_inv, Y_inner), W1_inv)
		Zee_inner = np.dot(np.dot(V,S_new) , V.transpose())
		Zee=np.dot (np.dot(W2_inv, Zee_inner), W2_inv)
		
		W1 = spla.sqrtm(Y + np.matrix(np.identity(np.shape(Y)[0]))*delta)
		#W1= W1.real
		W1= np.linalg.inv(W1)
		
		W2 = spla.sqrtm(Zee + np.matrix(np.identity(np.shape(Zee)[0]))*delta)
		#W2=W2.real
		W2= np.linalg.inv(W2)
		##print "WAW", WAW
		#A,E,flag, WAW = RW_NN_ALM(alpha, Z, n, m, W1, W2,flag)
		#raw_input("Where do we go now?")
		#alpha= RW_NN_bin_search_answer(10, 0, 0.001, Z, n, m, W1, W2,flga)
		# U,S,V=np.linalg.svd(A)
		# s=S[len(S)-1]
		# A=soft_threshold(A,s)
		#E=Z-A
		#print "Alpha value is", alpha
	return A,E, WAW, W1, W2, itr

#########binary search alpha##########
def NN_bin_search_answer(hi, lo, eps, Z, n, m,speed,struc, itr):
	mid= lo + (hi-lo)*0.5
	#print hi, lo, mid
	k=0
	A, E,k = NN_ALM(mid, Z, n, m,speed,struc)
	itr=itr+k
	rank= np.linalg.matrix_rank(A)
	if ( (mid-lo) < eps and rank <=n-1):
		return mid,itr
	else:
		if (rank <= n-1): # try lower alpha
			return NN_bin_search_answer(hi, mid, eps, Z, n, m,speed,struc,itr)
		else:
			return NN_bin_search_answer(mid, lo, eps, Z, n, m,speed,struc,itr)

def RW_NN_bin_search_answer(hi, lo, eps, Z, n, m, W1, W2,flag,struc,itr):
	mid= lo + (hi-lo)*0.5
	#print hi, lo, mid
	itr=itr+1
	A, E, flag, WAW , D= RW_NN_ALM(mid, Z, n, m, W1, W2,flag,struc)
	U,S,V= np.linalg.svd(D, full_matrices=True)
	s_min= S[len(S)-1]
	rank= np.linalg.matrix_rank(A)
	#print "s_min ", s_min, "eps", eps 
	if ( (mid-lo) < 1 and s_min <eps ):
		return mid,itr
	else:
		if (rank <= n-1): # try lower alpha
			return RW_NN_bin_search_answer(hi, mid, eps, Z, n, m, W1, W2,flag,struc,itr)
		else:
			return RW_NN_bin_search_answer(mid, lo, eps, Z, n, m, W1, W2,flag,struc,itr)



flag=0
delta=0.001
########################## Follwing is code for printing everything relevant###############

print " NN, n, error, (np.linalg.norm(E_NN)/error), t_nn, np.linalg.matrix_rank(A_NN), k_nn, RW_NN, n, error, (np.linalg.norm(E)/error), t_rw_nn,  np.linalg.matrix_rank(A), itr_rw"
eps=0.01
for i in range(2):
	n=m=i*5+50
	Covar= np.matrix(np.identity(n)) # 0 mean IID: we have n*1 y and multiple n*1 Xs. We take n-1 Xs.
	mean = [0 for i in range(n)]
	data= np.random.multivariate_normal(mean, Covar, m+1)
	Z=data.transpose()
	B, error= tls(data,m)
	W1= np.matrix(np.identity(n))
	W2= np.matrix(np.identity(m+1))
	t1=time.clock()
	A,E, WAW, W1, W2, itr_rw = RW_NN(Z, n, m, W1, W2, delta,flag,eps,0)
	t_rw_nn=time.clock()
	t_rw_nn= t_rw_nn-t1
	t2=time.clock()
	alpha, k_nn = NN_bin_search_answer(5000, 0, eps, Z, n, m,0,0,0)
	A_NN, E_NN,k_nn_new = NN_ALM(alpha, Z, n, m,0,0)
	k_nn = k_nn + k_nn_new
	t_nn=time.clock()
	t_nn= t_nn-t2
	print "NN", n, error, (np.linalg.norm(E_NN)/error)**2, t_nn, np.linalg.matrix_rank(A_NN), k_nn,\
		"RW_NN", n, error, (np.linalg.norm(E)/error)**2, t_rw_nn,  np.linalg.matrix_rank(A), itr_rw, alpha
######thresholding mehtod to find rank dficient matrices close to A##########################
#W1= np.matrix(np.identity(n))
#W2= np.matrix(np.identity(m+1))
#Covar= np.matrix(np.identity(n)) # 0 mean IID: we have n*1 y and multiple n*1 Xs. We take n-1 Xs.
# mean = [0 for i in range(n)]
# data= np.random.multivariate_normal(mean, Covar, m+1)
# Z=data.transpose()
# B, error= tls(data,m)
# ARW, ERW, WAW , W1new, W2new= RW_NN( Z, n, m, W1, W2, delta,0)
# print n, (np.linalg.norm(Z-ARW)), (np.linalg.norm(Z-ARW)/error)**2, np.linalg.matrix_rank(ARW)
# if(np.linalg.matrix_rank(ARW)==n):# shouldnt need it
# 	s=S[len(S)-1]
# 	A_lesser=soft_threshold(ARW,s)
# 	E_lesser=np.subtract(Z,A_lesser)
# 	print "Post-thresholding",n, np.linalg.norm(E_lesser), (np.linalg.norm(E_lesser)/error)**2, np.linalg.matrix_rank(A_lesser)

####### Error comparision for ALM###########
# print "n, error,(np.linalg.norm(E)), (np.linalg.norm(E)/error)**2, np.linalg.matrix_rank(A)"
# for i in range(5):
# 	n=m=i+5\
# 	Covar= np.matrix(np.identity(n)) # 0 mean IID: we have n*1 y and multiple n*1 Xs. We take n-1 Xs.
# 	mean = [0 for i in range(n)]
# 	data= np.random.multivariate_normal(mean, Covar, m+1)
# 	Z=data.transpose()
# 	B, error= tls(data,m)
# 	W1= np.matrix(np.identity(n))
# 	W2= np.matrix(np.identity(m+1))
# 	A,E, WAW, W1, W2 = RW_NN(Z, n, m, W1, W2, delta,flag)
# 	#A, E,k = NN_ALM(alpha, Z, n, m,0)
# 	print n, error,(np.linalg.norm(E)), (np.linalg.norm(E)/error), np.linalg.matrix_rank(A)



#############outout to update_rule_comp##############
# for i in range(100):
# 	n=m=i+5
# 	Covar= np.matrix(np.identity(n)) # 0 mean IID: we have n*1 y and multiple n*1 Xs. We take n-1 Xs.
# 	mean = [0 for i in range(n)]
# 	data= np.random.multivariate_normal(mean, Covar, m+1)
# 	Z=data.transpose()
# 	B, error= tls(data,m)
# 	alpha_NN_1 = NN_bin_search_answer(10000, 0, 0.001, Z, n, m,0)
# 	A1,E1,k_1 = NN_ALM(alpha_NN_1, Z, n, m,0)
# 	alpha_NN_2 = NN_bin_search_answer(10000, 0, 0.001, Z, n, m,0)
# 	A2,E2,k_2 = NN_ALM(alpha_NN_2, Z, n, m,1)
# 	print n, error,(np.linalg.norm(E1)/error)**2, k_1, (np.linalg.norm(E2)/error)**2, k_2


# ARW, ERW, WAW , W1new, W2new= RW_NN( Z, n, m, W1, W2, delta,0)
# #print np.linalg.norm(ERW), np.linalg.matrix_rank(ARW), np.linalg.matrix_rank(WAW)	
#########################################