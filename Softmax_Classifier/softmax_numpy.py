import numpy as np
def softmax(z):
    z_safe=z-np.max(z)
    exponents=np.exp(z_safe)
    probabilities=exponents/np.sum(exponents)
    return probabilities
if __name__=="__main__":
    logits=np.array([3.2,1.1,-0.5])
    probabilities=softmax(logits)
    print(f"Raw logits : {logits}")
    print(f"Probabilites:{np.round(probabilities,4)}")
    print(f"Total sum :{np.sum(probabilities):.1f}")
    expected_jersey=0.871
    if np.abs(probabilities[0]-expected_jersey)<0.01:
        print("Status:Passed")
    else:
        print("Status failed")