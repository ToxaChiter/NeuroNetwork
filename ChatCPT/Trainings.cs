namespace ChatCPT;

public enum Mode
{
    Regular,
    Advanced
}

//public class NeuralNetwork
//{
//    private int[] layerSizes;
//    private double[,] weights;
//    private double[] biases;
//    private double[] activations;
//    private double[] deltas;

//    public NeuralNetwork(int[] layerSizes)
//    {
//        this.layerSizes = layerSizes;
//        int numWeights = 0;
//        for (int i = 0; i < layerSizes.Length - 1; i++)
//        {
//            numWeights += layerSizes[i] * layerSizes[i + 1];
//        }
//        weights = new double[layerSizes.Length - 1, numWeights / (layerSizes.Length - 1)];
//        biases = new double[layerSizes.Length - 1];
//        activations = new double[layerSizes.Length];
//        deltas = new double[numWeights / (layerSizes.Length - 1)];
//        Random random = new Random();
//        for (int i = 0; i < weights.GetLength(0); i++)
//        {
//            for (int j = 0; j < weights.GetLength(1); j++)
//            {
//                weights[i, j] = random.NextDouble() * 2 - 1;
//            }
//            biases[i] = random.NextDouble() * 2 - 1;
//        }
//    }

//    public void Train(double[][] inputs, double[][] outputs, int epochs, double learningRate)
//    {
//        for (int i = 0; i < epochs; i++)
//        {
//            for (int j = 0; j < inputs.Length; j++)
//            {
//                // Feed the inputs forward through the network
//                FeedForward(inputs[j]);

//                // Backpropagate the error through the network
//                Backpropagate(outputs[j]);

//                // Update the weights and biases using the calculated deltas
//                UpdateWeightsAndBiases(learningRate);
//            }
//        }
//    }

//    private void FeedForward(double[] input)
//    {
//        // Copy the inputs into the first layer's activation values
//        Array.Copy(input, activations, input.Length);
//        int weightIndex = 0;
//        for (int i = 0; i < layerSizes.Length - 1; i++)
//        {
//            // Compute the newActivations for the next layer
//            double[] newActivations = new double[layerSizes[i + 1]];
//            for (int j = 0; j < layerSizes[i + 1]; j++)
//            {
//                double sum = 0.0;
//                for (int k = 0; k < layerSizes[i]; k++)
//                {
//                    // Compute the weighted sum of the previous layer's newActivations
//                    // to get the inputs to the current neuron
//                    sum += weights[i, j * layerSizes[i] + k] * activations[k];
//                }
//                // Apply the activation function (tanh) to the neuron's inputs
//                newActivations[j] = Math.Tanh(sum + biases[i]);
//            }
//            // Copy the newActivations into the current layer's activation values
//            Array.Copy(newActivations, activations, newActivations.Length);
//        }
//    }

//    private void Backpropagate(double[] output)
//    {
//        for (int i = 0; i < layerSizes[layerSizes.Length - 1]; i++)
//        {
//            double activation = activations[layerSizes[layerSizes.Length - 1 - 1] + i];
//            // Compute the error for the output layer neurons using the derivative of the activation function (tanh)
//            deltas[layerSizes[^1] + i] = activation * (1 - activation) * (output[i] - activations[layerSizes[^1] + i]);
//        }
//        int weightIndex = deltas.Length - layerSizes[^1];
//        for (int i = layerSizes.Length - 2; i >= 0; i--)
//        {
//            for (int j = 0; j < layerSizes[i]; j++)
//            {
//                double activation = activations[i * layerSizes[i + 1] + j];
//                double sum = 0.0;
//                for (int k = 0; k < layerSizes[i + 1]; k++)
//                {
//                    // Compute the weighted sum of the next layer's deltas
//                    // to get the contribution of each neuron to the current neuron's delta
//                    sum += deltas[(i + 1) * layerSizes[i + 1] + k] * weights[i, k * layerSizes[i] + j];
//                }
//                // Compute the error for the current layer neurons using the derivative of the activation function (tanh)
//                deltas[weightIndex--] = activation * (1 - activation) * sum;
//            }
//        }
//    }

//    private void UpdateWeightsAndBiases(double learningRate)
//    {
//        int weightIndex = 0;
//        for (int i = 0; i < layerSizes.Length - 1; i++)
//        {
//            for (int j = 0; j < layerSizes[i + 1]; j++)
//            {
//                // Update the biases using the calculated deltas
//                biases[i] += learningRate * deltas[layerSizes[i + 1] + j];
//                for (int k = 0; k < layerSizes[i]; k++)
//                {
//                    // Update the weights using the calculated deltas
//                    weights[i, j * layerSizes[i] + k] += learningRate * activations[k] * deltas[weightIndex++];
//                }
//            }
//        }
//    }
//}



internal class MyNeuroNetwork
{
    public int Layers { get; }
    public int Inputs { get; }
    public int[] Hiddens { get; }
    public int Outputs { get; }


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
        Outputs = output;

        layerSizes = new int[Layers];
        layerSizes[0] = inputs;
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

            for (int j = 0; j < weights.GetLength(1); j++)
            {
                for (int i = 0; i < weights.GetLength(0); i++)
                {
                    weights[i, j] = random.NextDouble() - 0.5;
                    if (double.IsNaN(weights[i, j]) || double.IsInfinity(weights[i, j])) throw new Exception();
                }
                biases[j] = random.NextDouble() - 0.5;
                if (double.IsNaN(biases[j]) || double.IsInfinity(biases[j])) throw new Exception();
            }

            this.weights.Add(weights);
            this.biases.Add(biases);
        }


        outputs = new List<double[]>(layerSizes.Length);
        weightedSums = new List<double[]>(layerSizes.Length);
        errors = new List<double[]>(layerSizes.Length);
        deltas = new List<double[]>(layerSizes.Length);

        for (int i = 0; i < layerSizes.Length; i++)
        {
            int n = layerSizes[i];

            outputs.Add(new double[n]);
            weightedSums.Add(new double[n]);
            errors.Add(new double[n]);
            deltas.Add(new double[n]);
        }
    }

    public MyNeuroNetwork(MyNeuroNetwork neuroNetwork)
    {
        Layers = neuroNetwork.Layers;
        Inputs = neuroNetwork.Inputs;
        Outputs = neuroNetwork.Outputs;
        Hiddens = neuroNetwork.Hiddens.Clone() as int[];

        layerSizes = neuroNetwork.layerSizes.Clone() as int[];

        weights = new List<double[,]>(neuroNetwork.weights.Count);
        for (int i = 0; i < neuroNetwork.weights.Count; i++)
        {
            weights.Add(neuroNetwork.weights[i].Clone() as double[,]);
        }

        biases = new List<double[]>(neuroNetwork.biases.Count);
        for (int i = 0; i < neuroNetwork.biases.Count; i++)
        {
            biases.Add(neuroNetwork.biases[i].Clone() as double[]);
        }

        outputs = new List<double[]>(layerSizes.Length);
        weightedSums = new List<double[]>(layerSizes.Length);
        errors = new List<double[]>(layerSizes.Length);
        deltas = new List<double[]>(layerSizes.Length);

        for (int i = 0; i < layerSizes.Length; i++)
        {
            int n = layerSizes[i];

            outputs.Add(new double[n]);
            weightedSums.Add(new double[n]);
            errors.Add(new double[n]);
            deltas.Add(new double[n]);
        }
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

    public void Train(List<(double[] Labels, double[] Image)> testCases, Mode mode, double learningRate)
    {
        var totalError = 0.0;
        switch (mode)
        {
            case Mode.Regular:
                //Parallel.For(0, testCases.Count, i =>
                //{
                for (int i = 0; i < testCases.Count; i++)
                    totalError += RegularTrain(testCases[i].Image, testCases[i].Labels, learningRate);
                //});
                Console.WriteLine($"Total error: {totalError}");
                break;

            case Mode.Advanced:
                //Parallel.For(0, testCases.Count, i =>
                //{
                for (int i = 0; i < testCases.Count; i++)
                    totalError += AdvancedTrain(testCases[i].Image, testCases[i].Labels);
                Console.WriteLine($"Total error: {totalError}");
                //});
                break;

            default:
                throw new ArgumentOutOfRangeException(nameof(mode));
        }
    }

    private double AdvancedTrain(double[] input, double[] output)
    {
        FeedForward(input);
        BackpropagateAdvanced(output);
        UpdateWeightsBiases();

        return output.Zip(outputs[^1], (t, o) => 0.5 * Math.Pow(t - o, 2)).Sum();
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
            var biases = this.biases[L];
            var weightedSums = this.weightedSums[L + 1];

            for (int j = 0; j < layerSizes[L + 1]; j++)
            {
                double sum = 0.0;
                for (int i = 0; i < layerSizes[L]; i++)
                {
                    // Compute the weighted sum of the previous layer's outputs
                    // to get the inputs to the current neuron
                    sum += weights[i, j] * prevActivations[i];
                    if (double.IsNaN(sum) || double.IsInfinity(sum)) throw new Exception();
                }
                weightedSums[j] = sum - biases[j];
                if (double.IsNaN(weightedSums[j]) || double.IsInfinity(weightedSums[j])) throw new Exception();
                // Apply the activation function (sigmoid) to the neuron's inputs
                newActivations[j] = Sigmoid(weightedSums[j]);
            }
        }
    }


    // Maybe mistakes
    private void BackpropagateAdvanced(double[] output)
    {
        // Compute the error for the output layer neurons
        for (int i = 0; i < layerSizes[^1]; i++)
        {
            errors[^1][i] = outputs[^1][i] - output[i];
        }


        double bottomLast = 1.0;
        for (int k = 0; k < layerSizes[^2]; k++)
        {
            bottomLast += Math.Pow(outputs[^2][k], 2);
        }

        for (int j = 0; j < layerSizes[^1]; j++)
        {
            double ln = Math.Log((outputs[^1][j] - errors[^1][j]) / (1 - outputs[^1][j] + errors[^1][j]));
            if (double.IsNaN(ln) || double.IsInfinity(ln)) throw new Exception();
            deltas[^1][j] = (weightedSums[^1][j] - ln) / bottomLast;
        }


        for (int L = Layers - 2; L > 0; L--)
        {
            var activations = this.outputs[L + 1];
            var prevActivations = this.outputs[L - 1];
            var prevWeightedSums = this.weightedSums[L + 1];
            var weightedSums = this.weightedSums[L];
            var weights = this.weights[L];
            var deltas = this.deltas[L];
            var errors = this.errors[L];

            double bottom = 1.0;
            for (int k = 0; k < layerSizes[L - 1]; k++)
            {
                bottom += Math.Pow(prevActivations[k], 2);
            }

            for (int j = 0; j < layerSizes[L]; j++)
            {
                // mistake?
                double sum = 0.0;
                for (int i = 0; i < layerSizes[L + 1]; i++)
                {
                    sum += activations[i] * weights[j, i] * SigmoidDerv(prevWeightedSums[i]);
                }
                errors[j] = sum;

                double ln = Math.Log((activations[j] - errors[j]) / (1 - activations[j] + errors[j]));
                if (double.IsNaN(ln) || double.IsInfinity(ln)) throw new Exception();
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

            for (int j = 0; j < weights.GetLength(1); j++)
            {
                for (int i = 0; i < weights.GetLength(0); i++)
                {
                    weights[i, j] -= deltas[j] * activations[i];
                }

                biases[j] += deltas[j];
            }
        }
    }



    private double RegularTrain(double[] inputs, double[] targets, double learningRate)
    {
        var input = inputs;
        var target = targets;

        // Forward pass

        FeedForward(input);

        // Backward pass
        for (int j = 0; j < Outputs; j++)
        {
            errors[^1][j] = (outputs[^1][j] - target[j]) * outputs[^1][j] * (1 - outputs[^1][j]);
            biases[^1][j] += learningRate * errors[^1][j];

            for (int k = 0; k < Hiddens[^1]; k++)
            {
                weights[^1][k, j] -= learningRate * errors[^1][j] * outputs[^2][k];
            }
        }

        for (int l = layerSizes.Length - 2; l > 0; l--)
        {
            var prevWeights = this.weights[l];
            var weights = this.weights[l - 1];
            var prevErrors = this.errors[l + 1];
            var errors = this.errors[l];
            var outputs = this.outputs[l];
            var prevOutputs = this.outputs[l - 1];
            var biases = this.biases[l - 1];


            for (int j = 0; j < layerSizes[l]; j++)
            {
                var weightedSum = 0.0;
                for (int k = 0; k < layerSizes[l + 1]; k++)
                {
                    weightedSum += prevErrors[k] * prevWeights[j, k];
                }
                errors[j] = weightedSum * outputs[j] * (1 - outputs[j]);
                biases[j] += learningRate * errors[j];
                for (int k = 0; k < layerSizes[l - 1]; k++)
                {
                    weights[k, j] -= learningRate * errors[j] * prevOutputs[k];
                }
            }
        }

        return target.Zip(outputs[^1], (t, o) => 0.5 * Math.Pow(t - o, 2)).Sum();
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





    public void Evaluate(List<(int Label, double[] Image)> checkCases)
    {
        int rightCounter = 0;
        Parallel.ForEach(checkCases, (testCase) =>
        {
            if (testCase.Label == Predict(testCase.Image))
            {
                rightCounter++;
            }
        });

        Console.WriteLine($"Right guesses: {rightCounter * 100.0 / checkCases.Count}%");
    }

















}