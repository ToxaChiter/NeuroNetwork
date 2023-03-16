using MNIST.IO;
using System.Numerics;

namespace NeuroNetwork;

internal class NeuroNetwork<T> where T : INumber<T>
{
    public Layer<T> HiddenLayer { get; set; }
    public Layer<T> OutputLayer { get; set; }
    public Func<T, T> ActivationFunction { get; set; }

    public NeuroNetwork(Matrix<T> hiddenMatrix, Matrix<T> outputMatrix, Func<T, T> activationFunction)
    {
        HiddenLayer = new Layer<T>(hiddenMatrix);
        OutputLayer = new Layer<T>(outputMatrix);
        ActivationFunction = activationFunction;
    }

    public List<T> Guess(Matrix<T> inputMatrix)
    {
        var hiddenMatrix = inputMatrix * HiddenLayer.Matrix;
        var outputMatrix = hiddenMatrix * OutputLayer.Matrix;

        var outputList = outputMatrix.ToList();

        return outputList.Select(x => ActivationFunction(x)).ToList();
    }
}