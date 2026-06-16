import numpy as np

def train_linear_regression(x,y,epochs=100,learning_rate=0.01):
    w=0.0
    b=0.0
    n=len(x)
    for epoch in range(epochs):
        y_pred=w*x+b #Forward pass
        loss=np.mean((y_pred-y)**2) #Calculate MSE loss
        dw=(2/n)*np.sum(x*(y_pred-y)) #Calculate Gradients for Backpropogation
        db=(2/n)*np.sum(y_pred-y)
        w-=learning_rate*dw #Update Weights
        b-=learning_rate*db
        if(epoch%100==0):
            print(f"Epoch{epoch}|Loss{loss:4f}|w:{w:4f}|b:{b:4f}")
x=np.array([1.0,2.0,3.0,4.0])
y=np.array([2.0,4.0,6.0,8.0])
train_linear_regression(x,y)