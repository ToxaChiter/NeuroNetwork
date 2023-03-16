using MNIST.IO;

namespace NeuroNetwork;

internal class Program
{
    static void Main()
    {
        var sigmoid = (double input) =>
        {
            return 1.0 / (1.0 + Math.Pow(Math.E, -input));
        };

        var randomInputWeights = RandomMatrix(28 * 28, 4, 5);
        var randomHiddenWeights = RandomMatrix(4, 10, 5);


        NeuroNetwork<double> neuroNetwork = new(activationFunction: sigmoid, hiddenMatrix: randomInputWeights, outputMatrix: randomHiddenWeights);

        var result = neuroNetwork.Guess(RandomMatrix(1, 28 * 28, 5));

        Console.WriteLine(string.Join(",\n", result));

        







        //var data = FileReaderMNIST.LoadImagesAndLables(
        //    "./data/train-labels-idx1-ubyte.gz",
        //    "./data/train-images-idx3-ubyte.gz");
    }

    static Matrix<double> RandomMatrix(int rows, int columns, int seed)
    {
        var random = new Random(seed);
        var array = new double[rows, columns];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                array[i, j] = (random.NextDouble() - 0.5) / 1.0;
            }
        }
        return new Matrix<double>(array);
    }
}