# Demonstration of the Perceptron and Linear Regressor on the basic logic functions
import numpy as np

class pcn:
    """ A basic Perceptron"""
    
    def __init__(self,inputs,targets):
        """ Constructor """
        # Set up network size
        if np.ndim(inputs)>1:
            self.nIn = np.shape(inputs)[1]
        else: 
            self.nIn = 1
    
        if np.ndim(targets)>1:
            self.nOut = np.shape(targets)[1]
        else:
            self.nOut = 1

        self.nData = np.shape(inputs)[0]
    
        # Initialise network
        self.weights = np.random.rand(self.nIn+1,self.nOut)*0.1-0.05

    def pcntrain(self,inputs,targets,eta,nIterations):
        """ Train the thing """ 
        # Add the inputs that match the bias node
        inputs = np.concatenate((inputs,-np.ones((self.nData,1))),axis=1)
        # Training
        change = range(self.nData)

        for n in range(nIterations):
            
            self.activations = self.pcnfwd(inputs);
            self.weights -= eta*np.dot(np.transpose(inputs),self.activations-targets)
        
            # Randomise order of inputs
            #np.random.shuffle(change)
            #inputs = inputs[change,:]
            #targets = targets[change,:]
            
        #return self.weights

    def pcnfwd(self,inputs):
        """ Run the network forward """
        # Compute activations
        activations =  np.dot(inputs,self.weights)

        # Threshold the activations
        return np.where(activations>0,1,0)


    def confmat(self,inputs,targets):
        """Confusion matrix"""

        # Add the inputs that match the bias node
        inputs = np.concatenate((inputs,-np.ones((self.nData,1))),axis=1)
        
        outputs = np.dot(inputs,self.weights)
    
        nClasses = np.shape(targets)[1]

        if nClasses==1:
            nClasses = 2
            outputs = np.where(outputs>0,1,0)
        else:
            # 1-of-N encoding
            outputs = np.argmax(outputs,1)
            targets = np.argmax(targets,1)

        cm = np.zeros((nClasses,nClasses))
        for i in range(nClasses):
            for j in range(nClasses):
                cm[i,j] = np.sum(np.where(outputs==i,1,0)*np.where(targets==j,1,0))

        print cm
        print np.trace(cm)/np.sum(cm)
        

inputs = np.array([[0,0],[0,1],[1,0],[1,1]])

# AND data
ANDtargets = np.array([[0],[0],[0],[1]])

# OR data
ORtargets = np.array([[0],[1],[1],[1]])

# XOR data
XORtargets = np.array([[0],[1],[1],[0]])


print("AND logic function")
pAND = pcn(inputs,ANDtargets)
pAND.pcntrain(inputs,ANDtargets,0.25,6)
pAND.confmat(inputs,ANDtargets)

print "OR logic function"
pOR = pcn(inputs,ORtargets)
pOR.pcntrain(inputs,ORtargets,0.25,6)
pOR.confmat(inputs,ANDtargets)

print "XOR logic function"
pXOR = pcn(inputs,XORtargets)
pXOR.pcntrain(inputs,XORtargets,0.25,6)
pXOR.confmat(inputs,ANDtargets)