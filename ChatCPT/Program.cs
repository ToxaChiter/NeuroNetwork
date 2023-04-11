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

using MNIST.IO;

public class NeuralNetwork
{
    private int inputNodes = 784;
    private int hiddenNodes = 16;
    private int outputNodes = 10;
    private double[][] weightsInputHidden;
    private double[][] weightsHiddenOutput;
    private double[] hiddenBiases;
    private double[] outputBiases;
    private double learningRate = 0.1;
    private Func<double, double> activationFunction = Sigmoid;

    public NeuralNetwork()
    {
        // Initialize weights and biases with random values
        var random = new Random();
        weightsInputHidden = Enumerable.Range(0, inputNodes)
            .Select(i => Enumerable.Range(0, hiddenNodes)
                .Select(j => random.NextDouble() - 0.5)
                .ToArray())
            .ToArray();
        weightsHiddenOutput = Enumerable.Range(0, hiddenNodes)
            .Select(i => Enumerable.Range(0, outputNodes)
                .Select(j => random.NextDouble() - 0.5)
                .ToArray())
            .ToArray();
        hiddenBiases = Enumerable.Range(0, hiddenNodes)
            .Select(i => random.NextDouble() - 0.5)
            .ToArray();
        outputBiases = Enumerable.Range(0, outputNodes)
            .Select(i => random.NextDouble() - 0.5)
            .ToArray();
    }

    public void Train(double[][] inputs, double[][] targets, int epochs)
    {
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            double totalError = 0;
            Parallel.For(0, inputs.Length,
                i =>
                {
                    var input = inputs[i];
                    var target = targets[i];

                    // Forward pass
                    var hiddenOutputs = new double[hiddenNodes];
                    for (int j = 0; j < hiddenNodes; j++)
                    {
                        var weightedSum = 0.0;
                        for (int k = 0; k < inputNodes; k++)
                        {
                            weightedSum += input[k] * weightsInputHidden[k][j];
                        }
                        hiddenOutputs[j] = activationFunction(weightedSum + hiddenBiases[j]);
                    }
                    var outputs = new double[outputNodes];
                    for (int j = 0; j < outputNodes; j++)
                    {
                        var weightedSum = 0.0;
                        for (int k = 0; k < hiddenNodes; k++)
                        {
                            weightedSum += hiddenOutputs[k] * weightsHiddenOutput[k][j];
                        }
                        outputs[j] = activationFunction(weightedSum + outputBiases[j]);
                    }

                    // Backward pass
                    var outputErrors = new double[outputNodes];
                    for (int j = 0; j < outputNodes; j++)
                    {
                        outputErrors[j] = (outputs[j] - target[j]) * outputs[j] * (1 - outputs[j]);
                        outputBiases[j] -= learningRate * outputErrors[j];
                        for (int k = 0; k < hiddenNodes; k++)
                        {
                            weightsHiddenOutput[k][j] -= learningRate * outputErrors[j] * hiddenOutputs[k];
                        }
                    }
                    var hiddenErrors = new double[hiddenNodes];
                    for (int j = 0; j < hiddenNodes; j++)
                    {
                        var weightedSum = 0.0;
                        for (int k = 0; k < outputNodes; k++)
                        {
                            weightedSum += outputErrors[k] * weightsHiddenOutput[j][k];
                        }
                        hiddenErrors[j] = weightedSum * hiddenOutputs[j] * (1 - hiddenOutputs[j]);
                        hiddenBiases[j] -= learningRate * hiddenErrors[j];
                        for (int k = 0; k < inputNodes; k++)
                        {
                            weightsInputHidden[k][j] -= learningRate * hiddenErrors[j] * input[k];
                        }
                    }

                    totalError += target.Zip(outputs, (t, o) => 0.5 * Math.Pow(t - o, 2)).Sum();
                });

            //for (int i = 0; i < inputs.Length; i++)
            //{
            //    var input = inputs[i];
            //    var target = targets[i];

            //    // Forward pass
            //    var hiddenOutputs = new double[hiddenNodes];
            //    for (int j = 0; j < hiddenNodes; j++)
            //    {
            //        var weightedSum = 0.0;
            //        for (int k = 0; k < inputNodes; k++)
            //        {
            //            weightedSum += input[k] * weightsInputHidden[k][j];
            //        }
            //        hiddenOutputs[j] = activationFunction(weightedSum + hiddenBiases[j]);
            //    }
            //    var outputs = new double[outputNodes];
            //    for (int j = 0; j < outputNodes; j++)
            //    {
            //        var weightedSum = 0.0;
            //        for (int k = 0; k < hiddenNodes; k++)
            //        {
            //            weightedSum += hiddenOutputs[k] * weightsHiddenOutput[k][j];
            //        }
            //        outputs[j] = activationFunction(weightedSum + outputBiases[j]);
            //    }

            //    // Backward pass
            //    var outputErrors = new double[outputNodes];
            //    for (int j = 0; j < outputNodes; j++)
            //    {
            //        outputErrors[j] = (outputs[j] - target[j]) * outputs[j] * (1 - outputs[j]);
            //        outputBiases[j] -= learningRate * outputErrors[j];
            //        for (int k = 0; k < hiddenNodes; k++)
            //        {
            //            weightsHiddenOutput[k][j] -= learningRate * outputErrors[j] * hiddenOutputs[k];
            //        }
            //    }
            //    var hiddenErrors = new double[hiddenNodes];
            //    for (int j = 0; j < hiddenNodes; j++)
            //    {
            //        var weightedSum = 0.0;
            //        for (int k = 0; k < outputNodes; k++)
            //        {
            //            weightedSum += outputErrors[k] * weightsHiddenOutput[j][k];
            //        }
            //        hiddenErrors[j] = weightedSum * hiddenOutputs[j] * (1 - hiddenOutputs[j]);
            //        hiddenBiases[j] -= learningRate * hiddenErrors[j];
            //        for (int k = 0; k < inputNodes; k++)
            //        {
            //            weightsInputHidden[k][j] -= learningRate * hiddenErrors[j] * input[k];
            //        }
            //    }

            //    totalError += target.Zip(outputs, (t, o) => 0.5 * Math.Pow(t - o, 2)).Sum();
            //}

            Console.WriteLine($"Epoch {epoch}, Error = {totalError}");

            if (epoch % 5 == 0)
            {
                Evaluate();
            }
        }
    }

    public int Predict(double[] input)
    {
        var hiddenOutputs = new double[hiddenNodes];
        for (int j = 0; j < hiddenNodes; j++)
        {
            var weightedSum = 0.0;
            for (int k = 0; k < inputNodes; k++)
            {
                weightedSum += input[k] * weightsInputHidden[k][j];
            }
            hiddenOutputs[j] = activationFunction(weightedSum + hiddenBiases[j]);
        }
        var outputs = new double[outputNodes];
        for (int j = 0; j < outputNodes; j++)
        {
            var weightedSum = 0.0;
            for (int k = 0; k < hiddenNodes; k++)
            {
                weightedSum += hiddenOutputs[k] * weightsHiddenOutput[k][j];
            }
            outputs[j] = activationFunction(weightedSum + outputBiases[j]);
        }

        return Array.IndexOf(outputs, outputs.Max());
    }
    
    private static double Sigmoid(double x)
    {
        return 1 / (1 + Math.Exp(-x));
    }

    public List<(int Label, double[] Image)> testCases;
    public List<(int Label, double[] Image)> checkCases;

    public void Evaluate()
    {
        int rightCounter = 0;
        Parallel.ForEach(checkCases, (testCase) =>
        {
            if (testCase.Label == Predict(testCase.Image))
            {
                rightCounter++;
            }
        });
        //foreach (var (Label, Image) in testCases)
        //{
        //    if (Label == Predict(Image))
        //    {
        //        rightCounter++;
        //    }
        //}
        Console.WriteLine($"Right guesses: {rightCounter * 100.0 / checkCases.Count}%");
    }

    //public void WriteToFile(string filename)
    //{
    //    using StreamWriter writer = new StreamWriter(filename);
    //    writer.
    //}
}

class Program
{
    static void Main()
    {
        // Download and deserialize MNIST dataset
        //var client = new HttpClient();
        //var mnistData = await client.GetStreamAsync("https://raw.githubusercontent.com/hsjeong5/MNIST-for-Numpy/master/mnist.npz");
        //var archive = new ZipArchive(mnistData);
        //var images = LoadData(archive.GetEntry("x_train.npy").Open());
        //var labels = LoadData(archive.GetEntry("y_train.npy").Open());

        var data = FileReaderMNIST.LoadImagesAndLables(
            "../../../data/train-labels-idx1-ubyte.gz",
            "../../../data/train-images-idx3-ubyte.gz");

        var checkData = FileReaderMNIST.LoadImagesAndLables(
            "../../../data/train-labels-idx1-ubyte.gz",
            "../../../data/train-images-idx3-ubyte.gz");

        Console.WriteLine("Loaded");
        List<double[]> images = new List<double[]>(60_000);
        List<double[]> labels = new List<double[]>(60_000);

        List<(int Label, double[] Image)> testCases = new(60_000);
        List<(int Label, double[] Image)> checkCases = new(10_000);

        //var result = Parallel.ForEach(data,
        //    testCase =>
        //    {
        //        var temp = testCase.Image.Cast<byte>().Select(b => (double)b).ToArray();
        //        images.Add(temp);
        //        var label = new double[10];
        //        label.Initialize();
        //        label[testCase.Label] = 1.0;
        //        labels.Add(label);

        //        testCases.Add((testCase.Label, temp));
        //    });

        //await Console.Out.WriteLineAsync(result.IsCompleted.ToString());


        foreach (var testCase in data)
        {
            var temp = testCase.Image.Cast<byte>().Select(b => (double)b).ToArray();


            //var temp = new double[28 * 28];
            //Parallel.For(0, 28,
            //    i =>
            //    {
            //        for (int j = 0; j < 28; j++)
            //        {
            //            temp[i] = testCase.Image[i, j];
            //        }
            //    });

            images.Add(temp);
            var label = new double[10];
            label.Initialize();
            label[testCase.Label] = 1.0;
            labels.Add(label);

            testCases.Add((testCase.Label, temp));
        }

        foreach (var testCase in checkData)
        {
            var temp = testCase.Image.Cast<byte>().Select(b => (double)b).ToArray();

            //var temp = new double[28 * 28];
            //Parallel.For(0, 28,
            //    i =>
            //    {
            //        for (int j = 0; j < 28; j++)
            //        {
            //            temp[i] = testCase.Image[i, j];
            //        }
            //    });

            images.Add(temp);
            var label = new double[10];
            label.Initialize();
            label[testCase.Label] = 1.0;
            labels.Add(label);

            checkCases.Add((testCase.Label, temp));
        }

        Console.WriteLine("Performed");


        // Normalize image data
        Parallel.ForEach(images, 
            image =>
            {
                for (int i = 0; i < images[i].Length; i++)
                {
                    image[i] /= 255.0;
                }
            });
        //for (int i = 0; i < images.Count; i++)
        //{
        //    for (int j = 0; j < images[i].Length; j++)
        //    {
        //        images[i][j] /= 255.0;
        //    }
        //}
        Console.WriteLine("Normalized");



        // Create neural network
        var neuralNetwork = new NeuralNetwork();
        neuralNetwork.testCases = testCases;
        neuralNetwork.checkCases = checkCases;

        neuralNetwork.Evaluate();
        // Train neural network
        neuralNetwork.Train(images.ToArray(), labels.ToArray(), 51);

        //// Test neural network on a few examples
        //var testImages = LoadData(archive.GetEntry("x_test.npy").Open()).Take(10).ToArray();
        //var testLabels = LoadData(archive.GetEntry("y_test.npy").Open()).Take(10).ToArray();
        //for (int i = 0; i < testImages.Length; i++)
        //{
        //    var predicted = neuralNetwork.Predict(testImages[i]);
        //    Console.WriteLine($"Predicted: {predicted}, Actual: {testLabels[i]}");
        //}
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
