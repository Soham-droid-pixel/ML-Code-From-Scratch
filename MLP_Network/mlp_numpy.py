import numpy as np
X=np.array([[0,0],[0,1],[1,0],[1,1]])
y=np.array([[0],[1],[1],[0]])
np.random.seed(42)
W1=np.random.randn(2,4)
b1=np.zeros((1,4))
W2=np.random.randn(4,1)
b2=np.zeros((1,1))

for epoch in range(10000):
    z1=np.dot(X,W1)+b1
    A1=np.maximum(0,z1)

    z2=np.dot(A1,W2)+b2
    A2=1/(1+np.exp(-z2))
    
    error=A2-y

    hidden_error=np.dot(error,W2.T)*(z1>0)
    W2-=0.1*np.dot(A1.T,error)
    b2-=0.1*np.sum(error,axis=0,keepdims=True)
    
    W1-=0.1*np.dot(X.T,hidden_error)
    b1-=0.1*np.sum(hidden_error,axis=0,keepdims=True)

print(f"Final predictions:\n",np.round(A2,4))
