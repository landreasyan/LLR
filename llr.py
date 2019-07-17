import networkx as nx
import numpy as np
import random
import math

class LLR():
    
    def __init__(self):
        self.mode = ''

     
    def fit(self, X, y, mu, v, perm_size = 50, var=None, Graph=None):
        p = len(X[0,:])
        n = len(X[:,0])
        self.var = var
        self.perm_size = perm_size
        self.Y = np.zeros(n)
        self.W = np.zeros((len(X[:, 0]), len(X[:, 0])))
        self.X = X
        D_compress = np.zeros((len(X[:, 0]), len(X[:, 0])))
        D_inv_compress = np.zeros((len(X[:, 0]), len(X[:, 0])))
        #G = np.random.normal(0, 1, (perm_size, len(X[:, 0])))
        E = np.zeros((len(X[:, 0]), perm_size))

        # to store data of the reduced X
        W_small = np.zeros((perm_size, perm_size))
        #A = np.zeros((perm_size, perm_size))
        D = np.zeros((perm_size, perm_size))
        D_inv = np.zeros((perm_size, perm_size))

        self.perm = random.sample(range(0, len(X[:, 0])), perm_size)
        #q, r, self.perm = linalg.qr(np.matmul(G,self.W), pivoting=True)
        
        #Check for number of rows in X and W to match
        
        if Graph is None: 
            # Compute the weight matrix of X
            for i in range(0, len(X[:, 0])):
                for j in range(0, len(X[:, 0])):
                    if i != j:
                        self.W[i,j] = math.exp((-1)*np.linalg.norm(X[i]-X[j])**2/self.var**2);
                        
            # Compute the square weight matrix with only the choosen columns of X
            
            for i in range(0, perm_size):
                for j in range(0, perm_size):
                    if i != j:
                        W_small[i,j] = math.exp((-1)*np.linalg.norm(X[self.perm[i]]-self.X[self.perm[j]])**2/self.var**2); 
                        self.W_small = W_small
            # Find the degree of the columns of the reduced X matrix  
            for i in range(0, perm_size):
                D[i,i] = np.sum(W_small[i,:]);
                if D[i,i] == 0:
                    raise ValueError("Some vertices in the graph are not connected.")
                D_inv[i,i] = 1/D[i,i];

        else: 
            self.W = Graph
            W_graph = nx.from_numpy_matrix(self.W, parallel_edges=False, create_using=nx.Graph)
            while(1):
                try:
                    W_small_graph = W_graph.subgraph(self.perm[0:perm_size])
                    W_small = nx.adjacency_matrix(W_small_graph)
                    W_small = W_small.todense()
                    self.W_small = W_small
                    print('here again')
                    # Find the degree of the columns of the reduced X matrix  
                    for i in range(0, perm_size):
                        D[i,i] = np.sum(W_small[i,:]);
                        if D[i,i] == 0:
                            raise ValueError()
                        D_inv[i,i] = 1/D[i,i];
                    break

                except: 
                    print("here")
                    self.perm = random.sample(range(0, len(X[:, 0])), perm_size)
                    pass

        # pick the columns of W with biggest eigenvalues
        for i in range(0, perm_size): 
            E[:,i] = self.W[:, (self.perm[i])];
        E = np.mat(E)

        # Compute the degrees of the choosen columns 
        for i in range(0, len(X[:, 0])):
            D_compress[i,i] = np.sum(E[i,:]);
            if D_compress[i,i] == 0:
                raise ValueError("Some vertices in the graph are not connected.")
            D_inv_compress[i,i] = 1/D_compress[i,i];

        # Compute the normalized wieght matrix of the choosen columns
        self.W_asym = np.matmul(D_inv_compress, E)

        W_A = np.block([self.W_asym, np.zeros((n, perm_size*p))]);
        for i in range(1, p+1):
            W_A_temp = np.block([np.zeros((n,i*perm_size)), self.W_asym, np.zeros((n, p*perm_size-i*perm_size))]);    
            W_A = np.block([[W_A], [W_A_temp]])
            
        '''fig = plt.figure()
        plt.imshow(W_A)
        plt.show()'''
        
        n = perm_size
        # Compute the Laplacian matrix of reduced X matrix
        L = np.subtract(np.identity(perm_size), np.matmul(D_inv, W_small))
        self.L = L
        # Keep only the choosed columns of X and their y values
        X_new = self.X[self.perm[0], :]
        y_new = y[self.perm[0]]
        for i in range(1, perm_size):
            X_new = np.block([[X_new], [self.X[self.perm[i], :]]])
            y_new = np.block([[y_new],[y[self.perm[i]]]])   

        # Constructing X_block matrix of smaller X
        self.X_block = np.identity(perm_size)
        for i in range(0, len(X[0, :])):
            self.X_block = np.block([self.X_block, np.diag(X_new[:,i])]);

        # Constructing M block matrix of smaller X
        M_alpha = np.block([mu*L, np.zeros((perm_size, perm_size*p))]);
        M_beta = np.block([np.zeros((perm_size,perm_size)), (v/p)*L, np.zeros((perm_size, perm_size*p-perm_size))]);

        for i in range(1, p):
            M_temp = np.block([np.zeros((perm_size,(i+1)*perm_size)), (v/p)*L, np.zeros((perm_size, perm_size*p+perm_size-(i+2)*perm_size))]);    
            M_beta = np.block([[M_beta], [M_temp]])

        M = np.block([[M_alpha], [M_beta]])
        M = np.mat(M);
        '''
        fig = plt.figure()
        plt.imshow(M)
        plt.show()
        '''

        #Compute the theta of smaller X 
        self.X_block = np.mat(self.X_block);
        to_invert = np.matmul(self.X_block.transpose(),self.X_block) + M
        u, s, vh = np.linalg.svd((to_invert), full_matrices=True)

        for i in range(0, len(s)):
            s[i] = 1/s[i]

        inverse = np.matmul(np.matmul(vh.transpose(),np.diag(s)),u.transpose())
        self.Theta = np.matmul(np.matmul(inverse,self.X_block.transpose()),y_new)

        #extrapolate theta to the whole dataset
        Theta = np.matmul(W_A, self.Theta); 
        self.weight = W_A

        # Compute the predicted y values
        n = len(X[:,:])
        for i in range(0, len(X[:,:])):   
            temp = 0
            for j in range (1, len(X[0,:])+1):
                temp = temp + Theta[i+j*n, 0]*X[i,j-1]
            self.Y[i] = Theta[i,0] + temp
            #self.Y[i] = Theta[i,0] + Theta[i+n, 0]*X[i,0]+Theta[i+2*n,0]*X[i,1]
        return self
    
    def predict(self, X_new):
        
        # Prediction vector
        Y = np.zeros(len(X_new[:,0]))
        
        # Asymmetric weight vector 
        W_predict = np.zeros((len(X_new[:,0]), self.perm_size))
        
        # Find the distance between the new X values and X values in the model 
        for i in range(0, len(X_new[:,0])):
            for j in range(0, self.perm_size):
                W_predict[i,j] = math.exp((-1)*np.linalg.norm(X_new[i]-self.X[self.perm[j]])**2/self.var**2); 
              
        # Normalize the weight matrix
        for i in range(0, len(W_predict[:, 0])):
            d = np.sum(W_predict[i,:]);
            if d == 0:
                raise ValueError("Some vertices in the graph are not connected.")
            W_predict[i,:] = W_predict[i,:]/d;
            
        size = self.perm_size
        p = len(self.X[0,:])
        # Construct a block diagonal weight matrix with the new assymentric weight matrix
        W_A = np.block([W_predict, np.zeros((len(W_predict[:,0]), size*p))]);
        for i in range(1, p+1):
            W_A_temp = np.block([np.zeros((len(W_predict[:,0]),i*size)), W_predict, np.zeros((len(W_predict[:,0]), p*size-i*size))]);    
            W_A = np.block([[W_A], [W_A_temp]])
        
        # Extrapolate the previously computed theta to the new dataset
        Theta = np.matmul(W_A, self.Theta)
        
        # Compute the prediction 
        for i in range(0, len(X_new[:,0])):
            
            Y[i] = Theta[i,0] 
            temp = 0
            for j in range (1, len(X_new[0,:])+1):
                temp += Theta[i+j*len(X_new[:,0]), 0]*X_new[i,j-1]
            Y[i] += temp
            #Y[i] = Theta[i,0] + Theta[i+len(X_new[:,0]), 0]*X_new[i,0]+Theta[i+2*len(X_new[:,0]),0]*X_new[i,1]
        return Y   
