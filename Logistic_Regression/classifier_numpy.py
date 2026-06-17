import numpy as np
def sigmoid(z):
    return 1/(1+np.exp(-z))
def forward_pass(x,w,b):
    z=np.dot(w,x)+b
    probability=sigmoid(z)
    return probability

if __name__=="__main__":
    x=np.array([2.5,1.2])
    w=np.array([0.8,-0.4])
    b=-0.5
    probability=forward_pass(x,w,b)
    print(f"Calculated probability :{probability:.4f}")
    if probability is not None and round(probability,4)==0.7350:
        print("Status:Passed")
    else:
        print("Status failed:check ur maths formula")
