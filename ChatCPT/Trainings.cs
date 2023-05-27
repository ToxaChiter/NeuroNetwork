namespace ChatCPT;

public enum Mode
{
    Regular,
    AdvancedOutput,
    Advanced,
}


public class Setup
{
    public string Name { get; set; }
    public double LearningRate { get; set; }
    public int Batch { get; set; }
    public bool IsParallel { get; set; }
    public Mode Mode { get; set; }
    public int EpochMax { get; set; }
    public string Directory { get; set; }

    public Action<Setup, int> ChangeSetupFromEpoch { get; set; }

    public Setup()
    {

    }
    public Setup(Setup setup)
    {
        LearningRate = setup.LearningRate;
        Batch = setup.Batch;
        IsParallel = setup.IsParallel;
        Directory = setup.Directory;
    }

    public Setup Copy()
    {
        return new Setup(this)
        {
            Name = this.Name,
            Mode = this.Mode,
            EpochMax = this.EpochMax,
            ChangeSetupFromEpoch = this.ChangeSetupFromEpoch
        };
    }

    public void ChangeSetup(int epoch)
    {
        ChangeSetupFromEpoch?.Invoke(this, epoch);
    }
}


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
    private List<double[,]> weightDeltas;
    private List<double[]> biaseDeltas;


    public MyNeuroNetwork(int inputs, int[] hiddens, int output)
    {
        Layers = 1 + hiddens.Length + 1;
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

        biaseDeltas = new List<double[]>(Layers - 1);
        weightDeltas = new List<double[,]>(Layers - 1);

        for (int L = 0; L < Layers - 1; L++)
        {
            var weights = new double[layerSizes[L], layerSizes[L + 1]];
            var biases = new double[layerSizes[L + 1]];

            biaseDeltas.Add(new double[layerSizes[L + 1]]);
            weightDeltas.Add(new double[layerSizes[L], layerSizes[L + 1]]);

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

        for (int i = 0; i < layerSizes.Length; i++)
        {
            int n = layerSizes[i];

            outputs.Add(new double[n]);
            weightedSums.Add(new double[n]);
            errors.Add(new double[n]);
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

        biaseDeltas = new List<double[]>(Layers - 1);
        weightDeltas = new List<double[,]>(Layers - 1);

        for (int L = 0; L < Layers - 1; L++)
        {
            biaseDeltas.Add(new double[layerSizes[L + 1]]);
            weightDeltas.Add(new double[layerSizes[L], layerSizes[L + 1]]);
        }

        outputs = new List<double[]>(layerSizes.Length);
        weightedSums = new List<double[]>(layerSizes.Length);
        errors = new List<double[]>(layerSizes.Length);

        for (int i = 0; i < layerSizes.Length; i++)
        {
            int n = layerSizes[i];

            outputs.Add(new double[n]);
            weightedSums.Add(new double[n]);
            errors.Add(new double[n]);
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

    public double Train(List<(double[] Labels, double[] Image)> trainCases, Mode mode, double learningRate, int batch = 1)
    {
        // random peek from traincases
        var randTrainCases = RandomList(trainCases);

        var totalError = 0.0;


        switch (mode)
        {
            case Mode.Regular:
                for (int i = 0; i < randTrainCases.Count; i += batch)
                {
                    ResetDeltas();
                    for (int j = 0; j < batch && i < randTrainCases.Count; j++, i++)
                    {
                        RegularTrain(randTrainCases[i].Image, randTrainCases[i].Labels, learningRate);
                        totalError += randTrainCases[i].Labels.Zip(outputs[^1], (t, o) => 0.5 * Math.Pow(t - o, 2)).Sum();
                    }
                    UpdateWeightsBiases(batch);
                }

                return totalError;

            case Mode.AdvancedOutput:
                for (int i = 0; i < randTrainCases.Count;)
                {
                    ResetDeltas();
                    for (int j = 0; j < batch && i < randTrainCases.Count; j++, i++)
                    {
                        AdvancedOutputTrain(randTrainCases[i].Image, randTrainCases[i].Labels, learningRate);
                        totalError += randTrainCases[i].Labels.Zip(outputs[^1], (t, o) => 0.5 * Math.Pow(t - o, 2)).Sum();
                    }
                    UpdateWeightsBiases(batch);
                }

                return totalError;

            case Mode.Advanced:
                for (int i = 0; i < randTrainCases.Count;)
                {
                    ResetDeltas();
                    for (int j = 0; j < batch && i < randTrainCases.Count; j++, i++)
                    {
                        AdvancedTrain(randTrainCases[i].Image, randTrainCases[i].Labels);
                        totalError += randTrainCases[i].Labels.Zip(outputs[^1], (t, o) => 0.5 * Math.Pow(t - o, 2)).Sum();
                    }
                    UpdateWeightsBiases(batch);
                }

                return totalError;

            default:
                throw new ArgumentOutOfRangeException(nameof(mode));
        }
    }

    public double TrainParallel(List<(double[] Labels, double[] Image)> trainCases, Mode mode, double learningRate, int batch = 1)
    {
        // random peek from traincases
        var randTrainCases = RandomList(trainCases);

        var totalError = 0.0;


        switch (mode)
        {
            case Mode.Regular:
                for (int i = 0; i < randTrainCases.Count; i += batch)
                {
                    ResetDeltas();

                    for (int j = 0; j < batch && i < randTrainCases.Count; j++, i++)
                    {
                        RegularTrain(randTrainCases[i].Image, randTrainCases[i].Labels, learningRate, true);
                        totalError += randTrainCases[i].Labels.Zip(outputs[^1], (t, o) => 0.5 * Math.Pow(t - o, 2)).Sum();
                    }
                    UpdateWeightsBiases(batch);
                }


                return totalError;

            case Mode.AdvancedOutput:


                for (int i = 0; i < randTrainCases.Count;)
                {
                    ResetDeltas();
                    for (int j = 0; j < batch && i < randTrainCases.Count; j++, i++)
                    {
                        AdvancedOutputTrain(randTrainCases[i].Image, randTrainCases[i].Labels, learningRate, true);
                        totalError += randTrainCases[i].Labels.Zip(outputs[^1], (t, o) => 0.5 * Math.Pow(t - o, 2)).Sum();
                    }
                    UpdateWeightsBiases(batch);
                }


                return totalError;

            case Mode.Advanced:


                for (int i = 0; i < randTrainCases.Count;)
                {
                    ResetDeltas();
                    for (int j = 0; j < batch && i < randTrainCases.Count; j++, i++)
                    {
                        AdvancedTrain(randTrainCases[i].Image, randTrainCases[i].Labels, true);
                        totalError += randTrainCases[i].Labels.Zip(outputs[^1], (t, o) => 0.5 * Math.Pow(t - o, 2)).Sum();
                    }
                    UpdateWeightsBiases(batch);
                }


                return totalError;

            default:
                throw new ArgumentOutOfRangeException(nameof(mode));
        }
    }

    private void ResetDeltas()
    {
        foreach (var delta in weightDeltas)
        {
            for (int i = 0; i < delta.GetLength(0); i++)
            {
                for (int j = 0; j < delta.GetLength(1); j++)
                {
                    delta[i, j] = 0.0;
                }
            }
        }

        foreach (var delta in biaseDeltas)
        {
            for (int i = 0; i < delta.Length; i++)
            {
                delta[i] = 0.0;
            }
        }
    }

    private void AdvancedTrain(double[] input, double[] output, bool isParallel = false)
    {
        if (isParallel)
        {
            FeedForwardParallel(input);
            BackpropagateAdvancedParallel(output);
        }
        else
        {
            FeedForward(input);
            BackpropagateAdvanced(output);
        }
    }

    private void AdvancedOutputTrain(double[] input, double[] output, double learningRate, bool isParallel = false)
    {
        if (isParallel)
        {
            FeedForwardParallel(input);
            BackpropagateAdvancedOutputParallel(output, learningRate);
        }
        else
        {
            FeedForward(input);
            BackpropagateAdvancedOutput(output, learningRate);
        }
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
                }
                weightedSums[j] = sum - biases[j];
                // Apply the activation function (sigmoid) to the neuron's inputs
                newActivations[j] = Sigmoid(weightedSums[j]);
            }
        }
    }

    private void FeedForwardParallel(double[] inputs)
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

            Parallel.For(0, layerSizes[L + 1], j =>
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
            });
        }
    }


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
            double D = outputs[^1][j] - errors[^1][j];

            if (D <= 0.0) D = 0.001;
            if (D >= 1.0) D = 0.999;

            double ln = Math.Log(D / (1 - D));
            var delta = (weightedSums[^1][j] - ln) / bottomLast;

            biaseDeltas[^1][j] += delta;
            for (int i = 0; i < layerSizes[^2]; i++)
            {
                weightDeltas[^1][i, j] += delta * outputs[^2][i];
            }
        }


        for (int L = Layers - 2; L > 0; L--)
        {
            var nextActivations = this.outputs[L + 1];
            var activations = this.outputs[L];
            var prevActivations = this.outputs[L - 1];
            var nextWeightedSums = this.weightedSums[L + 1];
            var weightedSums = this.weightedSums[L];
            var weights = this.weights[L];
            var biaseDeltas = this.biaseDeltas[L - 1];
            var weightDeltas = this.weightDeltas[L - 1];
            var errors = this.errors[L];
            var nextErrors = this.errors[L + 1];

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
                    sum += nextErrors[i] * weights[j, i] * SigmoidDerv(nextWeightedSums[i]);
                }
                errors[j] = sum;

                // many questions
                double D = activations[j] - errors[j];

                if (D <= 0.0) D = 0.001;
                if (D >= 1.0) D = 0.999;

                double ln = Math.Log(D / (1 - D));
                //if (double.IsNaN(ln))
                //{
                //    int ex = 0;
                //}

                var delta = (weightedSums[j] - ln) / bottom;
                biaseDeltas[j] += delta;
                for (int i = 0; i < layerSizes[L - 1]; i++)
                {
                    weightDeltas[i, j] += delta * prevActivations[i];
                }
            }
        }
    }

    private void BackpropagateAdvancedParallel(double[] output)
    {
        // Compute the error for the output layer neurons
        for (int i = 0; i < layerSizes[^1]; i++)
        {
            errors[^1][i] = outputs[^1][i] - output[i];
        }


        double bottomLast = 1.0;
        Parallel.For(0, layerSizes[^2], k =>
        {
            bottomLast += Math.Pow(outputs[^2][k], 2);
        });

        Parallel.For(0, layerSizes[^1], j =>
        {
            double D = outputs[^1][j] - errors[^1][j];

            if (D <= 0.0) D = 0.001;
            if (D >= 1.0) D = 0.999;
            double ln = Math.Log(D / (1 - D));

            var delta = (weightedSums[^1][j] - ln) / bottomLast;

            biaseDeltas[^1][j] += delta;
            for (int i = 0; i < layerSizes[^2]; i++)
            {
                weightDeltas[^1][i, j] += delta * outputs[^2][i];
            }
        });


        for (int L = Layers - 2; L > 0; L--)
        {
            var nextActivations = this.outputs[L + 1];
            var activations = this.outputs[L];
            var prevActivations = this.outputs[L - 1];
            var nextWeightedSums = this.weightedSums[L + 1];
            var weightedSums = this.weightedSums[L];
            var weights = this.weights[L];
            var biaseDeltas = this.biaseDeltas[L - 1];
            var weightDeltas = this.weightDeltas[L - 1];
            var errors = this.errors[L];
            var nextErrors = this.errors[L + 1];

            double bottom = 1.0;
            Parallel.For(0, layerSizes[L - 1], k =>
            {
                bottom += Math.Pow(prevActivations[k], 2);
            });

            Parallel.For(0, layerSizes[L], j =>
            {
                double sum = 0.0;
                for (int i = 0; i < layerSizes[L + 1]; i++)
                {
                    sum += nextErrors[i] * weights[j, i] * SigmoidDerv(nextWeightedSums[i]);
                }
                errors[j] = sum;

                double D = activations[j] - errors[j];

                if (D <= 0.0) D = 0.001;
                if (D >= 1.0) D = 0.999;

                double ln = Math.Log(D / (1 - D));

                var delta = (weightedSums[j] - ln) / bottom;
                biaseDeltas[j] += delta;
                for (int i = 0; i < layerSizes[L - 1]; i++)
                {
                    weightDeltas[i, j] += delta * prevActivations[i];
                }
            });
        }
    }


    private void BackpropagateAdvancedOutput(double[] output, double learningRate)
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
            double D = outputs[^1][j] - errors[^1][j];

            if (D <= 0.0) D = 0.001;
            if (D >= 1.0) D = 0.999;
            double ln = Math.Log(D / (1 - D));

            var delta = (weightedSums[^1][j] - ln) / bottomLast;

            biaseDeltas[^1][j] += delta;
            for (int i = 0; i < layerSizes[^2]; i++)
            {
                weightDeltas[^1][i, j] += delta * outputs[^2][i];
            }
        }



        // regular train
        for (int L = layerSizes.Length - 2; L > 0; L--)
        {
            var prevWeights = this.weights[L];
            var prevErrors = this.errors[L + 1];
            var errors = this.errors[L];
            var outputs = this.outputs[L];
            var prevOutputs = this.outputs[L - 1];
            var weightDeltas = this.weightDeltas[L - 1];
            var biaseDeltas = this.biaseDeltas[L - 1];


            for (int j = 0; j < layerSizes[L]; j++)
            {
                var weightedSum = 0.0;
                for (int k = 0; k < layerSizes[L + 1]; k++)
                {
                    weightedSum += prevErrors[k] * prevWeights[j, k];
                }
                errors[j] = weightedSum * outputs[j] * (1 - outputs[j]);

                biaseDeltas[j] += learningRate * errors[j];
                for (int k = 0; k < layerSizes[L - 1]; k++)
                {
                    weightDeltas[k, j] += learningRate * errors[j] * prevOutputs[k];
                }
            }
        }
    }

    private void BackpropagateAdvancedOutputParallel(double[] output, double learningRate)
    {
        // Compute the error for the output layer neurons
        for (int i = 0; i < layerSizes[^1]; i++)
        {
            errors[^1][i] = outputs[^1][i] - output[i];
        }


        double bottomLast = 1.0;
        Parallel.For(0, layerSizes[^2], k =>
        {
            bottomLast += Math.Pow(outputs[^2][k], 2);
        });

        Parallel.For(0, layerSizes[^1], j =>
        {
            double D = outputs[^1][j] - errors[^1][j];

            if (D <= 0.0) D = 0.001;
            if (D >= 1.0) D = 0.999;
            double ln = Math.Log(D / (1 - D));

            var delta = (weightedSums[^1][j] - ln) / bottomLast;

            biaseDeltas[^1][j] += delta;
            for (int i = 0; i < layerSizes[^2]; i++)
            {
                weightDeltas[^1][i, j] += delta * outputs[^2][i];
            }
        });



        // regular train
        for (int L = layerSizes.Length - 2; L > 0; L--)
        {
            var prevWeights = this.weights[L];
            var prevErrors = this.errors[L + 1];
            var errors = this.errors[L];
            var outputs = this.outputs[L];
            var prevOutputs = this.outputs[L - 1];
            var weightDeltas = this.weightDeltas[L - 1];
            var biaseDeltas = this.biaseDeltas[L - 1];


            Parallel.For(0, layerSizes[L], j =>
            {
                var weightedSum = 0.0;
                for (int k = 0; k < layerSizes[L + 1]; k++)
                {
                    weightedSum += prevErrors[k] * prevWeights[j, k];
                }
                errors[j] = weightedSum * outputs[j] * (1 - outputs[j]);

                biaseDeltas[j] += learningRate * errors[j];
                for (int k = 0; k < layerSizes[L - 1]; k++)
                {
                    weightDeltas[k, j] += learningRate * errors[j] * prevOutputs[k];
                }
            });
        }
    }

    private void UpdateWeightsBiases(int batch)
    {
        for (int L = 0; L < layerSizes.Length - 1; L++)
        {
            var weights = this.weights[L];
            var weightDeltas = this.weightDeltas[L];
            var biaseDeltas = this.biaseDeltas[L];
            var biases = this.biases[L];

            for (int j = 0; j < weights.GetLength(1); j++)
            {
                for (int i = 0; i < weights.GetLength(0); i++)
                {
                    weights[i, j] -= weightDeltas[i, j] / batch;
                }

                biases[j] += biaseDeltas[j] / batch;
            }
        }
    }

    private void UpdateWeightsBiasesParallel(int batch)
    {
        for (int L = 0; L < layerSizes.Length - 1; L++)
        {
            var weights = this.weights[L];
            var weightDeltas = this.weightDeltas[L];
            var biaseDeltas = this.biaseDeltas[L];
            var biases = this.biases[L];

            Parallel.For(0, weights.GetLength(1), j =>
            {
                for (int i = 0; i < weights.GetLength(0); i++)
                {
                    weights[i, j] -= weightDeltas[i, j] / batch;
                }

                biases[j] += biaseDeltas[j] / batch;
            });
        }
    }

    private void RegularTrain(double[] input, double[] output, double learningRate, bool isParallel = false)
    {
        if (isParallel)
        {
            FeedForwardParallel(input);
            BackpropagateRegular(output, learningRate);
        }
        else
        {
            FeedForwardParallel(input);
            BackpropagateRegularParallel(output, learningRate);
        }
    }

    private void BackpropagateRegular(double[] output, double learningRate)
    {
        for (int j = 0; j < Outputs; j++)
        {
            errors[^1][j] = (outputs[^1][j] - output[j]) * outputs[^1][j] * (1 - outputs[^1][j]);

            biaseDeltas[^1][j] += learningRate * errors[^1][j];

            for (int k = 0; k < Hiddens[^1]; k++)
            {
                weightDeltas[^1][k, j] += learningRate * errors[^1][j] * outputs[^2][k];
            }
        }

        for (int L = layerSizes.Length - 2; L > 0; L--)
        {
            var prevWeights = this.weights[L];
            var prevErrors = this.errors[L + 1];
            var errors = this.errors[L];
            var outputs = this.outputs[L];
            var prevOutputs = this.outputs[L - 1];
            var weightDeltas = this.weightDeltas[L - 1];
            var biaseDeltas = this.biaseDeltas[L - 1];


            for (int j = 0; j < layerSizes[L]; j++)
            {
                var weightedSum = 0.0;
                for (int k = 0; k < layerSizes[L + 1]; k++)
                {
                    weightedSum += prevErrors[k] * prevWeights[j, k];
                }
                errors[j] = weightedSum * outputs[j] * (1 - outputs[j]);

                biaseDeltas[j] += learningRate * errors[j];
                for (int k = 0; k < layerSizes[L - 1]; k++)
                {
                    weightDeltas[k, j] += learningRate * errors[j] * prevOutputs[k];
                }
            }
        }
    }

    private void BackpropagateRegularParallel(double[] output, double learningRate)
    {
        for (int j = 0; j < Outputs; j++)
        {
            errors[^1][j] = (outputs[^1][j] - output[j]) * outputs[^1][j] * (1 - outputs[^1][j]);

            biaseDeltas[^1][j] += learningRate * errors[^1][j];

            for (int k = 0; k < Hiddens[^1]; k++)
            {
                weightDeltas[^1][k, j] += learningRate * errors[^1][j] * outputs[^2][k];
            }
        }

        for (int L = layerSizes.Length - 2; L > 0; L--)
        {
            var prevWeights = this.weights[L];
            var prevErrors = this.errors[L + 1];
            var errors = this.errors[L];
            var outputs = this.outputs[L];
            var prevOutputs = this.outputs[L - 1];
            var weightDeltas = this.weightDeltas[L - 1];
            var biaseDeltas = this.biaseDeltas[L - 1];


            Parallel.For(0, layerSizes[L], j =>
            {
                var weightedSum = 0.0;
                for (int k = 0; k < layerSizes[L + 1]; k++)
                {
                    weightedSum += prevErrors[k] * prevWeights[j, k];
                }
                errors[j] = weightedSum * outputs[j] * (1 - outputs[j]);

                biaseDeltas[j] += learningRate * errors[j];
                for (int k = 0; k < layerSizes[L - 1]; k++)
                {
                    weightDeltas[k, j] += learningRate * errors[j] * prevOutputs[k];
                }
            });
        }
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





    public double Evaluate(List<(int Label, double[] Image)> checkCases)
    {
        int rightCounter = 0;
        Parallel.ForEach(checkCases, (testCase) =>
        {
            if (testCase.Label == Predict(testCase.Image))
            {
                rightCounter++;
            }
        });

        return rightCounter * 100.0 / checkCases.Count;
    }

    public double Evaluate(List<(double[] Label, double[] Image)> checkCases)
    {
        int rightCounter = 0;
        Parallel.ForEach(checkCases, (testCase) =>
        {
            int index = -1;
            for (int i = 0; i < testCase.Label.Length; i++)
            {
                if (testCase.Label[i] == 1.0)
                {
                    index = i;
                    break;
                }
            }

            if (index == Predict(testCase.Image))
            {
                rightCounter++;
            }
        });

        return rightCounter * 100.0 / checkCases.Count;
    }



    public static List<T> RandomList<T>(List<T> list)
    {
        var random = new Random();
        var data = new T[list.Count];
        for (int i = 0; i < list.Count; i++)
        {
            int j = random.Next(i + 1);
            if (j != i)
                data[i] = data[j];
            data[j] = list[i];
        }
        return new List<T>(data);
    }
}