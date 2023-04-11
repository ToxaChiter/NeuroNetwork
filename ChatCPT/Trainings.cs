namespace ChatCPT;

internal class Trainings
{
}




public class NeuralNetwork
{
    private int[] layerSizes;
    private double[,] weights;
    private double[] biases;
    private double[] activations;
    private double[] deltas;

    public NeuralNetwork(int[] layerSizes)
    {
        this.layerSizes = layerSizes;
        int numWeights = 0;
        for (int i = 0; i < layerSizes.Length - 1; i++)
        {
            numWeights += layerSizes[i] * layerSizes[i + 1];
        }
        weights = new double[layerSizes.Length - 1, numWeights / (layerSizes.Length - 1)];
        biases = new double[layerSizes.Length - 1];
        activations = new double[layerSizes.Length];
        deltas = new double[numWeights / (layerSizes.Length - 1)];
        Random random = new Random();
        for (int i = 0; i < weights.GetLength(0); i++)
        {
            for (int j = 0; j < weights.GetLength(1); j++)
            {
                weights[i, j] = random.NextDouble() * 2 - 1;
            }
            biases[i] = random.NextDouble() * 2 - 1;
        }
    }

    public void Train(double[][] inputs, double[][] outputs, int epochs, double learningRate)
    {
        for (int i = 0; i < epochs; i++)
        {
            for (int j = 0; j < inputs.Length; j++)
            {
                // Feed the inputs forward through the network
                FeedForward(inputs[j]);

                // Backpropagate the error through the network
                Backpropagate(outputs[j]);

                // Update the weights and biases using the calculated deltas
                UpdateWeightsAndBiases(learningRate);
            }
        }
    }

    private void FeedForward(double[] input)
    {
        // Copy the inputs into the first layer's activation values
        Array.Copy(input, activations, input.Length);
        int weightIndex = 0;
        for (int i = 0; i < layerSizes.Length - 1; i++)
        {
            // Compute the newActivations for the next layer
            double[] newActivations = new double[layerSizes[i + 1]];
            for (int j = 0; j < layerSizes[i + 1]; j++)
            {
                double sum = 0.0;
                for (int k = 0; k < layerSizes[i]; k++)
                {
                    // Compute the weighted sum of the previous layer's newActivations
                    // to get the inputs to the current neuron
                    sum += weights[i, j * layerSizes[i] + k] * activations[k];
                }
                // Apply the activation function (tanh) to the neuron's inputs
                newActivations[j] = Math.Tanh(sum + biases[i]);
            }
            // Copy the newActivations into the current layer's activation values
            Array.Copy(newActivations, activations, newActivations.Length);
        }
    }

    private void Backpropagate(double[] output)
    {
        for (int i = 0; i < layerSizes[layerSizes.Length - 1]; i++)
        {
            double activation = activations[layerSizes[layerSizes.Length - 1 - 1] + i];
            // Compute the error for the output layer neurons using the derivative of the activation function (tanh)
            deltas[layerSizes[^1] + i] = activation * (1 - activation) * (output[i] - activations[layerSizes[^1] + i]);
        }
        int weightIndex = deltas.Length - layerSizes[^1];
        for (int i = layerSizes.Length - 2; i >= 0; i--)
        {
            for (int j = 0; j < layerSizes[i]; j++)
            {
                double activation = activations[i * layerSizes[i + 1] + j];
                double sum = 0.0;
                for (int k = 0; k < layerSizes[i + 1]; k++)
                {
                    // Compute the weighted sum of the next layer's deltas
                    // to get the contribution of each neuron to the current neuron's delta
                    sum += deltas[(i + 1) * layerSizes[i + 1] + k] * weights[i, k * layerSizes[i] + j];
                }
                // Compute the error for the current layer neurons using the derivative of the activation function (tanh)
                deltas[weightIndex--] = activation * (1 - activation) * sum;
            }
        }
    }

    private void UpdateWeightsAndBiases(double learningRate)
    {
        int weightIndex = 0;
        for (int i = 0; i < layerSizes.Length - 1; i++)
        {
            for (int j = 0; j < layerSizes[i + 1]; j++)
            {
                // Update the biases using the calculated deltas
                biases[i] += learningRate * deltas[layerSizes[i + 1] + j];
                for (int k = 0; k < layerSizes[i]; k++)
                {
                    // Update the weights using the calculated deltas
                    weights[i, j * layerSizes[i] + k] += learningRate * activations[k] * deltas[weightIndex++];
                }
            }
        }
    }
}



internal class MyNeuroNetwork
{
    public int Layers { get; }
    public int Inputs { get; }
    public int[] Hiddens { get; }
    public int Output { get; }


    private int[] layerSizes;
    private List<double[,]> weights;
    private List<double[]> biases;
    private List<double[]> outputs;
    private List<double[]> weightedSums;
    private List<double[]> errors;
    private List<double[]> deltas;



    public MyNeuroNetwork(int layers, int inputs, int[] hiddens, int output)
    {
        Layers = layers;
        Inputs = inputs;
        Hiddens = hiddens;
        Output = output;

        layerSizes = new int[Layers];
        layerSizes[0] = layers;
        layerSizes[^1] = output;
        for (int i = 0; i < Hiddens.Length; i++)
        {
            layerSizes[i + 1] = Hiddens[i];
        }

        // Initialize weights and biases with random values
        var random = new Random();

        weights = new List<double[,]>(Layers - 1);
        biases = new List<double[]>(Layers - 1);


        for (int L = 0; L < Layers - 1; L++)
        {
            var weights = new double[layerSizes[L], layerSizes[L + 1]];
            var biases = new double[layerSizes[L + 1]];

            for (int i = 0; i < weights.GetLength(0); i++)
            {
                for (int j = 0; j < weights.GetLength(1); j++)
                {
                    weights[i, j] = random.NextDouble() - 0.5;
                }
                biases[i] = random.NextDouble() - 0.5;
            }

            this.weights.Add(weights);
            this.biases.Add(biases);
        }
    }

    public MyNeuroNetwork(MyNeuroNetwork neuroNetwork)
    {
        // TODO
    }

    private static double Sigmoid(double x)
    {
        return 1 / (1 + Math.Exp(-x));
    }

    private static double SigmoidDerv(double x)
    {
        double sigm = Sigmoid(x);
        return sigm * (1 - sigm);
    }

    private void AdvancedTrain(double[] input, double[] output)
    {
        FeedForward(input);
        Backpropagate(output);
        UpdateWeightsBiases();
    }



    private void FeedForward(double[] inputs)
    {
        // Copy the inputs into the first layer's activation values
        Array.Copy(inputs, outputs[0], inputs.Length);
        Array.Copy(inputs, weightedSums[0], inputs.Length);

        for (int L = 0; L < layerSizes.Length - 1; L++)
        {
            // Compute the outputs for the next layer

            var weights = this.weights[L];
            var prevActivations = this.outputs[L];
            var newActivations = this.outputs[L + 1];
            var biases = this.biases[L + 1];
            var weightedSums = this.weightedSums[L + 1];

            for (int j = 0; j < layerSizes[L + 1]; j++)
            {
                double sum = 0.0;
                for (int i = 0; i < layerSizes[L]; i++)
                {
                    // Compute the weighted sum of the previous layer's outputs
                    // to get the inputs to the current neuron
                    sum += weights[i, j] * prevActivations[i];
                }
                weightedSums[j] = sum - biases[j];
                // Apply the activation function (sigmoid) to the neuron's inputs
                newActivations[j] = Sigmoid(weightedSums[j]);
            }
        }
    }


    // Maybe mistakes
    private void Backpropagate(double[] output)
    {
        // Compute the error for the output layer neurons
        for (int i = 0; i < layerSizes[^1]; i++) 
        {
            deltas[^1][i] = outputs[^1][i] - output[i];
        }


        for (int L = 1; L < layerSizes.Length; L++)
        {
            var activations = this.outputs[L];
            var prevActivations = this.outputs[L - 1];
            var weightedSums = this.weightedSums[L];
            var weights = this.weights[L];
            var deltas = this.deltas[L];
            var errors = this.errors[L];

            double bottom = 1.0;
            for (int k = 0; k < layerSizes[L - 1]; k++)
            {
                //bottom += prevActivations[k] * prevActivations[k];
                bottom += Math.Pow(prevActivations[k], 2);
            }

            for (int j = 0; j < layerSizes[L]; j++)
            {
                // mistake?
                double sum = 0.0;
                for (int i = 0; i < layerSizes[L]; i++)
                {
                    sum += weights[i, j];
                }
                errors[j] = sum * activations[j] * SigmoidDerv(weightedSums[j]);


                double ln = Math.Log((activations[j] - errors[j]) / (1 - activations[j] + errors[j]));
                deltas[j] = (weightedSums[j] - ln) / bottom;
            }
        }
    }

    private void UpdateWeightsBiases()
    {
        for (int L = 0; L < layerSizes.Length - 1; L++)
        {
            var activations = this.outputs[L];
            var weights = this.weights[L];
            var deltas = this.deltas[L];
            var biases = this.biases[L];

            for (int i = 0; i < weights.GetLength(0); i++)
            {
                for (int j = 0; j < weights.GetLength(1); j++)
                {
                    weights[i, j] -= deltas[i] * activations[i];
                }

                biases[i] += deltas[i];
            }
        }
    }



    private void RegularTrain(double[] input, double[] output)
    {
        // TODO
    }




    public int Predict(double[] input)
    {
        double[] inputs = input;
        double[] outputs = input;

        for (int L = 0; L < Layers - 1; L++)
        {
            outputs = new double[layerSizes[L + 1]];
            var weights = this.weights[L];
            var biases = this.biases[L];

            for (int j = 0; j < weights.GetLength(1); j++)
            {
                var weightedSum = 0.0;
                for (int i = 0; i < weights.GetLength(0); i++)
                {
                    weightedSum += inputs[i] * weights[i, j];
                }
                outputs[j] = Sigmoid(weightedSum - biases[j]);
            }

            inputs = outputs;
        }

        return Array.IndexOf(outputs, outputs.Max());
    }























}