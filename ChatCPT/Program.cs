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
using MNIST.IO;
using System.Reflection.Emit;

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
    static void Main()
    {
        var data = FileReaderMNIST.LoadImagesAndLables(
            "../../../data/train-labels-idx1-ubyte.gz",
            "../../../data/train-images-idx3-ubyte.gz");

        var checkData = FileReaderMNIST.LoadImagesAndLables(
            "../../../data/t10k-labels-idx1-ubyte.gz", 
            "../../../data/t10k-images-idx3-ubyte.gz");

        Console.WriteLine("Loaded");
        
        List<(double[] Labels, double[] Image)> trainCases = new(60_000);
        List<(int Label, double[] Image)> checkCases = new(10_000);


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

        //for (int i = 0; i < 60_000; i++)
        //{
        //    trainCases.Add((new double[10], new double[784]));
        //}

        //for (int i = 0; i < 10_000; i++)
        //{
        //    checkCases.Add((new int(), new double[784]));
        //}

        Console.WriteLine("Performed");
        
        MyNeuroNetwork regularNeuroNetwork = new MyNeuroNetwork(28 * 28, new int[] { 80, 16 }, 10);
        MyNeuroNetwork advancedOutputNeuroNetwork = new MyNeuroNetwork(regularNeuroNetwork);
        MyNeuroNetwork advancedNeuroNetwork = new MyNeuroNetwork(regularNeuroNetwork);

        var learningRate = 0.1;
        var batch = 24;

        double error = 0.0;
        var eval = regularNeuroNetwork.Evaluate(checkCases);
        Console.WriteLine($"Init precision: {eval}%");

        for (int epoch = 1; epoch < 51; epoch++)
        {
            Console.WriteLine($"\n\nAfter epoch #{epoch}:\n");

            error = regularNeuroNetwork.Train(trainCases, Mode.Regular, learningRate, batch);
            eval = regularNeuroNetwork.Evaluate(checkCases);
            Console.WriteLine($"Regular training:   error - {error},   precision - {eval}%");

            error = advancedOutputNeuroNetwork.Train(trainCases, Mode.AdvancedOutput, learningRate, batch);
            eval = advancedOutputNeuroNetwork.Evaluate(checkCases);
            Console.WriteLine($"Advanced-regular training:   error - {error},   precision - {eval}%");

            error = advancedNeuroNetwork.Train(trainCases, Mode.Advanced, learningRate, batch);
            eval = advancedNeuroNetwork.Evaluate(checkCases);
            Console.WriteLine($"Advanced training:   error - {error},   precision - {eval}%");
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
