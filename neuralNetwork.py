from numpy import exp, array, random, dot

class NeuralNetwork():
    def __init__(self, neurons):
        random.seed(1)
        self.synaptic_weights = 2 * random.random((neurons,1)) - 1    
    def __sigmoid(self, x):
        return 1 / (1+exp(-x))
    
    def __sigmoid_derivative(self,x):
        return x * (1-x)
    
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            output = self.think(training_set_inputs)
        
            error = training_set_outputs - output
            #print(error)
            #print(dot(training_set_inputs.T, error * self.__sigmoid_derivative(output)))
            #print(self.synaptic_weights)
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))
            self.synaptic_weights += adjustment
        
    def think(self, inputs):
        return self.__sigmoid(dot(inputs, self.synaptic_weights))

def dataGen(arrayLen, numArrays=7, testing=False):
    noArrays = numArrays
    binaryArray = []
    for i in range(noArrays):
        binaryArray.append(bin(i))
    
    Array = []
    for item in binaryArray:
        item = item[2:]
        tempArray = []
        for char in item:
            temp = int(char)
            tempArray.append(temp)
            
        Array.append(tempArray)
        
    for i, item in enumerate(Array):
        if len(item) < arrayLen:
            while len(item) < arrayLen:
                item.insert(0,0)
                
    random.shuffle(Array)
    if not testing:
        outputArray = []
        for item in Array:
            outputArray.append(item[0])
        returnTuple = (Array, outputArray)
    else:
        returnTuple = (Array[random.randint(0,len(Array))])
    return returnTuple
    
training_set_neurons = 10
training_set_inputs, training_set_outputs = dataGen(training_set_neurons, (2**training_set_neurons))
training_set_inputs, training_set_outputs = array(training_set_inputs), array([training_set_outputs]).T

random.seed(1)
neural_network = NeuralNetwork(training_set_neurons)
neural_network.train(training_set_inputs, training_set_outputs,100000)

testing_set_input = dataGen(training_set_neurons, 2**training_set_neurons, True)
print(testing_set_input)
print(neural_network.think(array(testing_set_input)))