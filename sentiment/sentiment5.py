import time
import sys
import numpy as np

# Encapsulate our neural network in a class
class SentimentNetwork:
    def __init__(self, reviews, labels, hidden_nodes = 10, learning_rate = 0.1):
        """Create a SentimenNetwork with the given settings
        Args:
            reviews(list) - List of reviews used for training
            labels(list) - List of POSITIVE/NEGATIVE labels associated with the given reviews
            hidden_nodes(int) - Number of nodes to create in the hidden layer
            learning_rate(float) - Learning rate to use while training

        """
        # Assign a seed to our random number generator to ensure we get
        # reproducable results during development
        np.random.seed(1)

        # process the reviews and their associated labels so that everything
        # is ready for training
        self.pre_process_data(reviews, labels)

        # Build the network to have the number of hidden nodes and the learning rate that
        # were passed into this initializer. Make the same number of input nodes as
        # there are vocabulary words and create a single output node.
        self.init_network(len(self.review_vocab),hidden_nodes, 1, learning_rate)

    def pre_process_data(self, reviews, labels):

        # populate review_vocab with all of the words in the given reviews
        review_vocab = set()
        for i in range(len(reviews)):
            review_vocab.update(reviews[i].split(' '))

        # Convert the vocabulary set to a list so we can access words via indices
        self.review_vocab = list(review_vocab)

        # populate label_vocab with all of the words in the given labels.
        label_vocab = set()
        for i in range(len(labels)):
            label_vocab.update(labels[i])

        # Convert the label vocabulary set to a list so we can access labels via indices
        self.label_vocab = list(label_vocab)

        # Store the sizes of the review and label vocabularies.
        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)

        # Create a dictionary of words in the vocabulary mapped to index positions
        self.word2index = {}
        for i,word in enumerate(self.review_vocab):
            self.word2index[word] = i

        # Create a dictionary of labels mapped to index positions
        self.label2index = {}
        for i,label in enumerate(self.label_vocab):
            self.label2index[label] = i


    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Store the number of nodes in input, hidden, and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Store the learning rate
        self.learning_rate = learning_rate

        self.layer_1 = np.zeroes((1,self.hidden_nodes))
        # Initialize weights

        # : initialize self.weights_0_1 as a matrix of zeros. These are the weights between
        #       the input layer and the hidden layer.
        self.weights_0_1 = np.zeros((self.input_nodes, self.hidden_nodes))

        # : initialize self.weights_1_2 as a matrix of random values.
        #       These are the weights between the hidden layer and the output layer.
        self.weights_1_2 = np.random.normal(0.0, self.input_nodes**-0.5,
                                            (self.hidden_nodes, self.output_nodes))

        # : Create the input layer, a two-dimensional matrix with shape
        #       1 x input_nodes, with all values initialized to zero
        self.layer_0 = np.zeros((1,input_nodes))


    def get_target_for_label(self,label):
        if label == 'POSITIVE':
            return 1
        else:
            return 0

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def sigmoid_output_2_derivative(self,output):
        return output * (1 - output)

    def train(self, training_reviews_raw, training_labels):

        # make sure out we have a matching number of reviews and labels
        #assert(len(training_reviews) == len(training_labels))

        # Keep track of correct predictions to display accuracy during training
        correct_so_far = 0

        # Remember when we started for printing time statistics
        start = time.time()

        training_reviews = list()
        for i in range(len(training_reviews_raw)):
            training_reviews[i] = list()
            for word in training_reviews_raw[i]:
                training_reviews[i].append(word2index[word])

        # loop through all the given reviews and run a forward and backward pass,
        # updating weights for every item
        for i in range(len(training_reviews)):

            # TODO: Get the next review and its correct label
            review = training_reviews[i]
            label  = training_labels[i]

            # forward pass through the network.
            #self.layer_0 = np.zeros((1,self.input_nodes))
            #self.update_input_layer(review)

            # (1,72810) × (72810,10)
            input_to_hidden = np.zeros((1, self.weights_0_1.shape[1]))
            for index in review:
              self.layer_1 += self.weights_0_1[index]
            # (1,10) × (10,1)


            #input_to_hidden = np.zeros((self.layer_0.shape[0], self.weights_0_1.shape[1]))
            #for j in range(len(self.layer_0[0])):
            #    n = self.layer_0[0][j]
            #    if n == 1:
            #            for w in range(len(self.weights_0_1[0])):
            #                input_to_hidden[0][w] += self.weights_0_1[0][w]

            hidden_to_output = np.dot(input_to_hidden, self.weights_1_2)
            output = self.sigmoid(hidden_to_output)



            # output error
            actual = self.get_target_for_label(label)
            error = actual - output
            output_error_term = error * self.sigmoid_output_2_derivative(output)

            # hidden error
            hidden_error = np.dot(self.weights_1_2, output_error_term)
            hidden_error_term = hidden_error
            # hidden_error_term.shape == (hidden,1)

            # update weights with error
            delta_weights_0_1 = np.dot(hidden_error_term, self.layer_0)
            delta_weights_1_2 = np.dot(output_error_term, hidden_to_output)

            self.weights_0_1 += self.learning_rate * delta_weights_0_1.T
            self.weights_1_2 += self.learning_rate * delta_weights_1_2.T

            if abs(error) < 0.5:
                correct_so_far += 1


            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0

            sys.stdout.write("\rProgress:" + str(100 * i/float(len(training_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct_so_far) + " #Trained:" + str(i+1) \
                             + " Training Accuracy:" + str(correct_so_far * 100 / float(i+1))[:4] + "%")
            if(i % 2500 == 0):
                print("")

    def test(self, testing_reviews, testing_labels):
        """
        Attempts to predict the labels for the given testing_reviews,
        and uses the test_labels to calculate the accuracy of those predictions.
        """

        # keep track of how many correct predictions we make
        correct = 0

        # we'll time how many predictions per second we make
        start = time.time()

        # Loop through each of the given reviews and call run to predict
        # its label.
        for i in range(len(testing_reviews)):
            pred = self.run(testing_reviews[i])
            if(pred == testing_labels[i]):
                correct += 1

            # For debug purposes, print out our prediction accuracy and speed
            # throughout the prediction process.

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0

            sys.stdout.write("\rProgress:" + str(100 * i/float(len(testing_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct) + " #Tested:" + str(i+1) \
                             + " Testing Accuracy:" + str(correct * 100 / float(i+1))[:4] + "%")

    def run(self, review):
        """
        Returns a POSITIVE or NEGATIVE prediction for the given review.
        """
        update_input_layer(review.lower())
        input_to_hidden = np.dot(self.layer_0, self.weights_0_1)
        hidden_to_output = np.dot(input_to_hidden, self.weights_1_2)
        output = self.sigmoid(hidden_to_output)

        if output >= 0.5:
            return 'POSITIVE'
        else:
            return 'NEGATIVE'

