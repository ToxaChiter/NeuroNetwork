using System.Numerics;

namespace NeuroNetwork;

internal class Layer<T> where T : INumber<T>
{
    public Matrix<T> Matrix { get; set; }

    public Matrix<T> Inputs { get; set; }

    public Layer(Matrix<T> matrix)
    {
        Matrix = matrix;
    }

    public void ChangeWeights()
    {

    }
}