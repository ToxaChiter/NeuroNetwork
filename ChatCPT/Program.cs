//Конечно, я могу написать простую нейросеть на языке C# используя .NET 6 для распознавания 
//рукописных цифр из датасета MNIST. Для начала, нам нужно определить структуру нейросети и ее параметры.

//Структура нейросети:

//Входной слой с 784 нейронами, по одному на каждый пиксель изображения MNIST
//Скрытый слой с 16 нейронами
//Выходной слой с 10 нейронами, по одному на каждую цифру
//Параметры нейросети:

//Коэффициент обучения: 0.1
//Количество эпох: 20
//Функция активации: Сигмоидная функция
//Метод обратного распространения ошибки
//Вот как может выглядеть код для реализации такой нейросети на C#:

using ChatCPT;
using Microsoft.VisualBasic.FileIO;
using MNIST.IO;
using System.Diagnostics;

//public class NeuralNetwork
//{
//    private int inputNodes = 784;
//    private int hiddenNodes = 16;
//    private int outputNodes = 10;
//    private double[][] weightsInputHidden;
//    private double[][] weightsHiddenOutput;
//    private double[] hiddenBiases;
//    private double[] outputBiases;
//    private double learningRate = 0.1;
//    private Func<double, double> activationFunction = Sigmoid;

//    public NeuralNetwork()
//    {
//        // Initialize weights and biases with random values
//        var random = new Random();
//        weightsInputHidden = Enumerable.Range(0, inputNodes)
//            .Select(i => Enumerable.Range(0, hiddenNodes)
//                .Select(j => random.NextDouble() - 0.5)
//                .ToArray())
//            .ToArray();
//        weightsHiddenOutput = Enumerable.Range(0, hiddenNodes)
//            .Select(i => Enumerable.Range(0, outputNodes)
//                .Select(j => random.NextDouble() - 0.5)
//                .ToArray())
//            .ToArray();
//        hiddenBiases = Enumerable.Range(0, hiddenNodes)
//            .Select(i => random.NextDouble() - 0.5)
//            .ToArray();
//        outputBiases = Enumerable.Range(0, outputNodes)
//            .Select(i => random.NextDouble() - 0.5)
//            .ToArray();
//    }

//    public void Train(double[][] inputs, double[][] targets, int epochs)
//    {
//        for (int epoch = 0; epoch < epochs; epoch++)
//        {
//            double totalError = 0;
//            Parallel.For(0, inputs.Length,
//                i =>
//                {
//                    var input = inputs[i];
//                    var target = targets[i];

//                    // Forward pass
//                    var hiddenOutputs = new double[hiddenNodes];
//                    for (int j = 0; j < hiddenNodes; j++)
//                    {
//                        var weightedSum = 0.0;
//                        for (int k = 0; k < inputNodes; k++)
//                        {
//                            weightedSum += input[k] * weightsInputHidden[k][j];
//                        }
//                        hiddenOutputs[j] = activationFunction(weightedSum + hiddenBiases[j]);
//                    }
//                    var outputs = new double[outputNodes];
//                    for (int j = 0; j < outputNodes; j++)
//                    {
//                        var weightedSum = 0.0;
//                        for (int k = 0; k < hiddenNodes; k++)
//                        {
//                            weightedSum += hiddenOutputs[k] * weightsHiddenOutput[k][j];
//                        }
//                        outputs[j] = activationFunction(weightedSum + outputBiases[j]);
//                    }

//                    // Backward pass
//                    var outputErrors = new double[outputNodes];
//                    for (int j = 0; j < outputNodes; j++)
//                    {
//                        outputErrors[j] = (outputs[j] - target[j]) * outputs[j] * (1 - outputs[j]);
//                        outputBiases[j] -= learningRate * outputErrors[j];
//                        for (int k = 0; k < hiddenNodes; k++)
//                        {
//                            weightsHiddenOutput[k][j] -= learningRate * outputErrors[j] * hiddenOutputs[k];
//                        }
//                    }
//                    var hiddenErrors = new double[hiddenNodes];
//                    for (int j = 0; j < hiddenNodes; j++)
//                    {
//                        var weightedSum = 0.0;
//                        for (int k = 0; k < outputNodes; k++)
//                        {
//                            weightedSum += outputErrors[k] * weightsHiddenOutput[j][k];
//                        }
//                        hiddenErrors[j] = weightedSum * hiddenOutputs[j] * (1 - hiddenOutputs[j]);
//                        hiddenBiases[j] -= learningRate * hiddenErrors[j];
//                        for (int k = 0; k < inputNodes; k++)
//                        {
//                            weightsInputHidden[k][j] -= learningRate * hiddenErrors[j] * input[k];
//                        }
//                    }

//                    totalError += target.Zip(outputs, (t, o) => 0.5 * Math.Pow(t - o, 2)).Sum();
//                });

//            //for (int i = 0; i < inputs.Length; i++)
//            //{
//            //    var input = inputs[i];
//            //    var target = targets[i];

//            //    // Forward pass
//            //    var hiddenOutputs = new double[hiddenNodes];
//            //    for (int j = 0; j < hiddenNodes; j++)
//            //    {
//            //        var weightedSum = 0.0;
//            //        for (int k = 0; k < inputNodes; k++)
//            //        {
//            //            weightedSum += input[k] * weightsInputHidden[k][j];
//            //        }
//            //        hiddenOutputs[j] = activationFunction(weightedSum + hiddenBiases[j]);
//            //    }
//            //    var outputs = new double[outputNodes];
//            //    for (int j = 0; j < outputNodes; j++)
//            //    {
//            //        var weightedSum = 0.0;
//            //        for (int k = 0; k < hiddenNodes; k++)
//            //        {
//            //            weightedSum += hiddenOutputs[k] * weightsHiddenOutput[k][j];
//            //        }
//            //        outputs[j] = activationFunction(weightedSum + outputBiases[j]);
//            //    }

//            //    // Backward pass
//            //    var outputErrors = new double[outputNodes];
//            //    for (int j = 0; j < outputNodes; j++)
//            //    {
//            //        outputErrors[j] = (outputs[j] - target[j]) * outputs[j] * (1 - outputs[j]);
//            //        outputBiases[j] -= learningRate * outputErrors[j];
//            //        for (int k = 0; k < hiddenNodes; k++)
//            //        {
//            //            weightsHiddenOutput[k][j] -= learningRate * outputErrors[j] * hiddenOutputs[k];
//            //        }
//            //    }
//            //    var hiddenErrors = new double[hiddenNodes];
//            //    for (int j = 0; j < hiddenNodes; j++)
//            //    {
//            //        var weightedSum = 0.0;
//            //        for (int k = 0; k < outputNodes; k++)
//            //        {
//            //            weightedSum += outputErrors[k] * weightsHiddenOutput[j][k];
//            //        }
//            //        hiddenErrors[j] = weightedSum * hiddenOutputs[j] * (1 - hiddenOutputs[j]);
//            //        hiddenBiases[j] -= learningRate * hiddenErrors[j];
//            //        for (int k = 0; k < inputNodes; k++)
//            //        {
//            //            weightsInputHidden[k][j] -= learningRate * hiddenErrors[j] * input[k];
//            //        }
//            //    }

//            //    totalError += target.Zip(outputs, (t, o) => 0.5 * Math.Pow(t - o, 2)).Sum();
//            //}

//            Console.WriteLine($"Epoch {epoch}, Error = {totalError}");

//            if (epoch % 5 == 0)
//            {
//                Evaluate();
//            }
//        }
//    }

//    public int Predict(double[] input)
//    {
//        var hiddenOutputs = new double[hiddenNodes];
//        for (int j = 0; j < hiddenNodes; j++)
//        {
//            var weightedSum = 0.0;
//            for (int k = 0; k < inputNodes; k++)
//            {
//                weightedSum += input[k] * weightsInputHidden[k][j];
//            }
//            hiddenOutputs[j] = activationFunction(weightedSum + hiddenBiases[j]);
//        }
//        var outputs = new double[outputNodes];
//        for (int j = 0; j < outputNodes; j++)
//        {
//            var weightedSum = 0.0;
//            for (int k = 0; k < hiddenNodes; k++)
//            {
//                weightedSum += hiddenOutputs[k] * weightsHiddenOutput[k][j];
//            }
//            outputs[j] = activationFunction(weightedSum + outputBiases[j]);
//        }

//        return Array.IndexOf(outputs, outputs.Max());
//    }

//    private static double Sigmoid(double x)
//    {
//        return 1 / (1 + Math.Exp(-x));
//    }

//    public List<(int Label, double[] Image)> trainCases;
//    public List<(int Label, double[] Image)> checkCases;

//    public void Evaluate()
//    {
//        int rightCounter = 0;
//        Parallel.ForEach(checkCases, (testCase) =>
//        {
//            if (testCase.Label == Predict(testCase.Image))
//            {
//                rightCounter++;
//            }
//        });
//        //foreach (var (Label, Image) in trainCases)
//        //{
//        //    if (Label == Predict(Image))
//        //    {
//        //        rightCounter++;
//        //    }
//        //}
//        Console.WriteLine($"Right guesses: {rightCounter * 100.0 / checkCases.Count}%");
//    }

//    //public void WriteToFile(string filename)
//    //{
//    //    using StreamWriter writer = new StreamWriter(filename);
//    //    writer.
//    //}
//}

class Program
{
    static List<(double[] Labels, double[] Image)> trainCases = new(60_000);
    static List<(int Label, double[] Image)> checkCases = new(10_000);

    static void Main()
    {
        var data = FileReaderMNIST.LoadImagesAndLables(
            "../../../data/train-labels-idx1-ubyte.gz",
            "../../../data/train-images-idx3-ubyte.gz");

        var checkData = FileReaderMNIST.LoadImagesAndLables(
            "../../../data/t10k-labels-idx1-ubyte.gz",
            "../../../data/t10k-images-idx3-ubyte.gz");

        Console.WriteLine("Loaded");




        foreach (var testCase in data)
        {
            var temp = testCase.Image.Cast<byte>().Select(b => b / 256.0).ToArray();
            if (temp.Any(e => double.IsNaN(e) || double.IsInfinity(e))) throw new Exception();

            //var arr = new double[28 * 28];
            //for (int i = 0; i < 28; i++)
            //{
            //    for (int j = 0; j < 28; j++)
            //    {
            //        arr[i * 28 + j] = Convert.ToDouble(temp[i, j]) / 256.0;
            //    }
            //}

            var label = new double[10];
            label.Initialize();
            label[testCase.Label] = 1.0;

            trainCases.Add((label, temp));
        }

        foreach (var testCase in checkData)
        {
            var temp = testCase.Image.Cast<byte>().Select(b => b / 256.0).ToArray();
            checkCases.Add((testCase.Label, temp));
        }

        #region PreviousTrain
        //for (int i = 0; i < 60_000; i++)
        //{
        //    trainCases.Add((new double[10], new double[784]));
        //}

        //for (int i = 0; i < 10_000; i++)
        //{
        //    checkCases.Add((new int(), new double[784]));
        //}

        //Console.WriteLine("Performed");

        //MyNeuroNetwork regularNeuroNetwork = new MyNeuroNetwork(28 * 28, new int[] { 80, 16 }, 10);
        //MyNeuroNetwork advancedOutputNeuroNetwork = new MyNeuroNetwork(regularNeuroNetwork);
        //MyNeuroNetwork advancedNeuroNetwork = new MyNeuroNetwork(regularNeuroNetwork);

        //MyNeuroNetwork regularNeuroNetworkParallel = new MyNeuroNetwork(regularNeuroNetwork);
        //MyNeuroNetwork advancedOutputNeuroNetworkParallel = new MyNeuroNetwork(regularNeuroNetwork);
        //MyNeuroNetwork advancedNeuroNetworkParallel = new MyNeuroNetwork(regularNeuroNetwork);

        //MyNeuroNetwork combinedAdvancedOutputRegularNetwork = new MyNeuroNetwork(regularNeuroNetwork);
        //MyNeuroNetwork combinedAdvancedRegularNetwork = new MyNeuroNetwork(regularNeuroNetwork);

        //MyNeuroNetwork combinedAdvancedOutputRegularNetworkParallel = new MyNeuroNetwork(regularNeuroNetwork);
        //MyNeuroNetwork combinedAdvancedRegularNetworkParallel = new MyNeuroNetwork(regularNeuroNetwork);


        //var learningRate = 0.1;
        //var batch = 10;

        //double error = 0.0;
        //var eval = regularNeuroNetwork.Evaluate(checkCases);
        //Console.WriteLine($"Init precision: {eval}%");

        //Stopwatch errorStopwatch = new Stopwatch();
        //Stopwatch evalStopwatch = new Stopwatch();

        //for (int epoch = 1; epoch < 51; epoch++)
        //{
        //    Console.WriteLine($"\n\nAfter epoch #{epoch}:\n");


        //    errorStopwatch.Start();
        //    error = regularNeuroNetwork.Train(trainCases, Mode.Regular, learningRate, batch);
        //    errorStopwatch.Stop();
        //    evalStopwatch.Start();
        //    eval = regularNeuroNetwork.Evaluate(checkCases);
        //    evalStopwatch.Stop();
        //    Console.WriteLine($"Regular training:                     error - {error:F3} ({errorStopwatch.Elapsed}),   precision - {eval}% ({regularNeuroNetwork.Evaluate(trainCases):F3}%)");
        //    errorStopwatch.Reset();
        //    evalStopwatch.Reset();


        //    errorStopwatch.Start();
        //    error = advancedOutputNeuroNetwork.Train(trainCases, Mode.AdvancedOutput, learningRate, batch);
        //    errorStopwatch.Stop();
        //    evalStopwatch.Start();
        //    eval = advancedOutputNeuroNetwork.Evaluate(checkCases);
        //    evalStopwatch.Stop();
        //    Console.WriteLine($"Advanced-regular training:            error - {error:F3} ({errorStopwatch.Elapsed}),   precision - {eval}% ({advancedOutputNeuroNetwork.Evaluate(trainCases):F3}%)");
        //    errorStopwatch.Reset();
        //    evalStopwatch.Reset();


        //    errorStopwatch.Start();
        //    error = advancedNeuroNetwork.Train(trainCases, Mode.Advanced, learningRate, batch);
        //    errorStopwatch.Stop();
        //    evalStopwatch.Start();
        //    eval = advancedNeuroNetwork.Evaluate(checkCases);
        //    evalStopwatch.Stop();
        //    Console.WriteLine($"Advanced training:                    error - {error:F3} ({errorStopwatch.Elapsed}),   precision - {eval}% ({advancedNeuroNetwork.Evaluate(trainCases):F3}%)");
        //    errorStopwatch.Reset();
        //    evalStopwatch.Reset();

        //    Console.WriteLine();

        //    //errorStopwatch.Start();
        //    //error = regularNeuroNetworkParallel.TrainParallel(trainCases, Mode.Regular, learningRate, batch);
        //    //errorStopwatch.Stop();
        //    //evalStopwatch.Start();
        //    //eval = regularNeuroNetworkParallel.Evaluate(checkCases);
        //    //evalStopwatch.Stop();
        //    //Console.WriteLine($"Regular parallel training:            error - {error:F3} ({errorStopwatch.Elapsed}),   precision - {eval}%");
        //    //errorStopwatch.Reset();
        //    //evalStopwatch.Reset();


        //    //errorStopwatch.Start();
        //    //error = advancedOutputNeuroNetworkParallel.TrainParallel(trainCases, Mode.AdvancedOutput, learningRate, batch);
        //    //errorStopwatch.Stop();
        //    //evalStopwatch.Start();
        //    //eval = advancedOutputNeuroNetworkParallel.Evaluate(checkCases);
        //    //errorStopwatch.Stop();
        //    //Console.WriteLine($"Advanced-regular parallel training:   error - {error:F3} ({errorStopwatch.Elapsed}),   precision - {eval}% ({advancedOutputNeuroNetworkParallel.Evaluate(trainCases):F3}%)");
        //    //errorStopwatch.Reset();
        //    //evalStopwatch.Reset();


        //    //errorStopwatch.Start();
        //    //error = advancedNeuroNetworkParallel.TrainParallel(trainCases, Mode.Advanced, learningRate, batch);
        //    //errorStopwatch.Stop();
        //    //evalStopwatch.Start();
        //    //eval = advancedNeuroNetworkParallel.Evaluate(checkCases);
        //    //errorStopwatch.Stop();
        //    //Console.WriteLine($"Advanced parallel training:           error - {error:F3} ({errorStopwatch.Elapsed}),   precision - {eval}% ({advancedNeuroNetworkParallel.Evaluate(trainCases):F3}%)");
        //    //errorStopwatch.Reset();
        //    //evalStopwatch.Reset();

        //    Console.WriteLine();

        //    errorStopwatch.Start();
        //    if (epoch < 4) error = combinedAdvancedOutputRegularNetwork.Train(trainCases, Mode.AdvancedOutput, learningRate, batch);
        //    else error = combinedAdvancedOutputRegularNetwork.Train(trainCases, Mode.Regular, learningRate, batch);
        //    errorStopwatch.Stop();
        //    evalStopwatch.Start();
        //    eval = combinedAdvancedOutputRegularNetwork.Evaluate(checkCases);
        //    evalStopwatch.Stop();
        //    Console.WriteLine($"adv-reg + reg training:               error - {error:F3} ({errorStopwatch.Elapsed}),   precision - {eval}% ({combinedAdvancedOutputRegularNetwork.Evaluate(trainCases):F3}%)");
        //    errorStopwatch.Reset();
        //    evalStopwatch.Reset();


        //    errorStopwatch.Start();
        //    if (epoch < 4) error = combinedAdvancedRegularNetwork.Train(trainCases, Mode.Advanced, learningRate, batch);
        //    else error = combinedAdvancedRegularNetwork.Train(trainCases, Mode.Regular, learningRate, batch);
        //    errorStopwatch.Stop();
        //    evalStopwatch.Start();
        //    eval = combinedAdvancedRegularNetwork.Evaluate(checkCases);
        //    evalStopwatch.Stop();
        //    Console.WriteLine($"adv + reg training:                   error - {error:F3} ({errorStopwatch.Elapsed}),   precision - {eval}% ({combinedAdvancedRegularNetwork.Evaluate(trainCases):F3}%)");
        //    errorStopwatch.Reset();
        //    evalStopwatch.Reset();

        //    //Console.WriteLine();

        //    //errorStopwatch.Start();
        //    //if (epoch < 4) error = combinedAdvancedOutputRegularNetworkParallel.TrainParallel(trainCases, Mode.AdvancedOutput, learningRate, batch);
        //    //else error = combinedAdvancedOutputRegularNetworkParallel.Train(trainCases, Mode.Regular, learningRate, batch);
        //    //errorStopwatch.Stop();
        //    //evalStopwatch.Start();
        //    //eval = combinedAdvancedOutputRegularNetworkParallel.Evaluate(checkCases);
        //    //evalStopwatch.Stop();
        //    //Console.WriteLine($"adv-reg parallel + reg training:      error - {error:F3} ({errorStopwatch.Elapsed}),   precision - {eval}%");
        //    //errorStopwatch.Reset();
        //    //evalStopwatch.Reset();


        //    //errorStopwatch.Start();
        //    //if (epoch < 4) error = combinedAdvancedRegularNetworkParallel.TrainParallel(trainCases, Mode.Advanced, learningRate, batch);
        //    //else error = combinedAdvancedRegularNetworkParallel.Train(trainCases, Mode.Regular, learningRate, batch);
        //    //errorStopwatch.Stop();
        //    //evalStopwatch.Start();
        //    //eval = combinedAdvancedRegularNetworkParallel.Evaluate(checkCases);
        //    //evalStopwatch.Stop();
        //    //Console.WriteLine($"adv parallel + reg training:          error - {error:F3} ({errorStopwatch.Elapsed}),   precision - {eval}%");
        //    //errorStopwatch.Reset();
        //    //evalStopwatch.Reset();
        //}
        #endregion

        Console.WriteLine("Performed");

        MyNeuroNetwork baseNeuroNetwork = new(28 * 28, new int[] { 100, 100, 100, 100, 100, 100, 100, 100, 100, 100 }, 10);
        Setup baseSetup = new() { IsParallel = false, LearningRate = 0.1, Batch = 20 };
        baseSetup.Name =
            $"Setup {baseNeuroNetwork.Inputs} {string.Join(" ", baseNeuroNetwork.Hiddens)} {baseNeuroNetwork.Outputs} " +
            $"{baseSetup.LearningRate:F2} {baseSetup.Batch} {baseSetup.IsParallel} extra"
            ;

        var directory = FileSystem.CombinePath("../../../", $"Tests/{baseSetup.Name}");
        FileSystem.CreateDirectory(directory);

        baseSetup.Directory = directory;

        List<(MyNeuroNetwork neuroNetwork, Setup setup)> cases = new()
        {
            (new MyNeuroNetwork(baseNeuroNetwork), new Setup(baseSetup)
            {
                Name = "regular", Mode = Mode.Regular, EpochMax = 50, ChangeSetupFromEpoch = null
            }),

            (new MyNeuroNetwork(baseNeuroNetwork), new Setup(baseSetup)
            {
                Name = "adv-reg", Mode = Mode.AdvancedOutput, EpochMax = 20, ChangeSetupFromEpoch = null
            }),

            (new MyNeuroNetwork(baseNeuroNetwork), new Setup(baseSetup)
            {
                Name = "advanced", Mode = Mode.Advanced, EpochMax = 20, ChangeSetupFromEpoch = null
            }),



            //(new MyNeuroNetwork(baseNeuroNetwork), new Setup(baseSetup)
            //{
            //    Name = "adv-reg reg", Mode = Mode.AdvancedOutput, EpochMax = 50, ChangeSetupFromEpoch = (Setup setup, int epoch) =>
            //    {
            //        if (epoch > 5) setup.Mode = Mode.Regular;
            //    }
            //}),

            //(new MyNeuroNetwork(baseNeuroNetwork), new Setup(baseSetup)
            //{
            //    Name = "adv reg", Mode = Mode.Advanced, EpochMax = 50, ChangeSetupFromEpoch = (Setup setup, int epoch) =>
            //    {
            //        if (epoch > 5) setup.Mode = Mode.Regular;
            //    }
            //}),



            //(new MyNeuroNetwork(baseNeuroNetwork), new Setup(baseSetup)
            //{
            //    Name = "reg adv-reg", Mode = Mode.Regular, EpochMax = 20, ChangeSetupFromEpoch = (Setup setup, int epoch) =>
            //    {
            //        if (epoch > 5) setup.Mode = Mode.AdvancedOutput;
            //    }
            //}),

            //(new MyNeuroNetwork(baseNeuroNetwork), new Setup(baseSetup)
            //{
            //    Name = "reg adv", Mode = Mode.Regular, EpochMax = 20, ChangeSetupFromEpoch = (Setup setup, int epoch) =>
            //    {
            //        if (epoch > 5) setup.Mode = Mode.Advanced;
            //    }
            //}),



            //(new MyNeuroNetwork(baseNeuroNetwork), new Setup(baseSetup)
            //{
            //    Name = "adv-reg + reg", Mode = Mode.AdvancedOutput, EpochMax = 25, ChangeSetupFromEpoch = (Setup setup, int epoch) =>
            //    {
            //        if (epoch % 2 == 1) setup.Mode = Mode.Regular;
            //        else setup.Mode = Mode.AdvancedOutput;
            //    }
            //}),

            //(new MyNeuroNetwork(baseNeuroNetwork), new Setup(baseSetup)
            //{
            //    Name = "adv + reg", Mode = Mode.Advanced, EpochMax = 25, ChangeSetupFromEpoch = (Setup setup, int epoch) =>
            //    {
            //        if (epoch % 2 == 1) setup.Mode = Mode.Regular;
            //        else setup.Mode = Mode.Advanced;
            //    }
            //}),
        };

        Stopwatch sw = Stopwatch.StartNew();
        Parallel.ForEach(cases, TrainWithSetup);
        sw.Stop();
        Console.WriteLine($"Time spent: {sw.Elapsed}");
    }

    static void TrainWithSetup((MyNeuroNetwork neuroNetwork, Setup setup) @case)
    {
        var neuroNetwork = @case.neuroNetwork;
        var setup = @case.setup;

        var stream = File.OpenWrite($"{setup.Directory}/{setup.Name}.txt");
        StreamWriter writer = new(stream);

        Stopwatch stopwatch = new();

        try
        {
            for (int epoch = 1; epoch < setup.EpochMax + 1; epoch++)
            {
                stopwatch.Reset();
                stopwatch.Start();
                var error = neuroNetwork.Train(trainCases, setup.Mode, setup.LearningRate, setup.Batch);
                stopwatch.Stop();

                var evalTest = neuroNetwork.Evaluate(checkCases);
                var evalTrain = neuroNetwork.Evaluate(trainCases);

                var str =
                    $"Epoch #{epoch}\n" +
                    $"Error - {error:F3} ({stopwatch.Elapsed})\n" +
                    $"Precision - {evalTest}% ({evalTrain:F3}%)\n\n";

                Console.WriteLine($"#{epoch} - {stopwatch.Elapsed}");
                writer.WriteLine(str);

                setup.ChangeSetup(epoch);
            }

            writer.Close();
        }
        finally
        {
            writer.Close();
        }
    }
}





//In this code, we first download and normalize the MNIST dataset. Then we create a `NeuralNetwork` class
//which has a constructor that takes the number of input nodes, hidden nodes, output nodes, learning rate,
//and activation function as parameters. The `Train` method takes the training data, training labels,
//and number of epochs as parameters and trains the neural network using the backpropagation algorithm.
//The `Predict` method takes an input and returns the predicted output class. Finally, in the `Main` method,
//we create a neural network with 784 input nodes, 16 hidden nodes, and 10 output nodes, and train 
//it using the first 60000 examples of the MNIST dataset. We then test the neural network on the 
//first 10 examples of the test set and print the predicted and actual labels.
