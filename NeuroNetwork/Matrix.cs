using System.Numerics;

namespace NeuroNetwork;

internal class Matrix<T> where T : INumber<T>
{
    private T[,] _matrix;
    public int Rows { get; private set; }
    public int Columns { get; private set; }

    public Matrix(T[,] matrix)
    {
        Rows = matrix.GetLength(0);
        Columns = matrix.GetLength(1);
        _matrix = new T[Rows, Columns];

        Array.Copy(matrix, _matrix, _matrix.Length);
    }

    private Matrix(int rows, int columns)
    {
        Rows = rows;
        Columns = columns;
        _matrix = new T[Rows, Columns];
        _matrix.Initialize();
    }

    public Matrix<T> Multiply(Matrix<T> right)
    {
        if (this._matrix.GetLength(1) != right._matrix.GetLength(0))
        {
            throw new ArgumentException("Matrixes cannot be multiplied");
        }

        var result = new Matrix<T>(this._matrix.GetLength(0), right._matrix.GetLength(1));

        var leftMatrix = this._matrix;
        var rightMatrix = right._matrix;
        var resultMatrix = result._matrix;

        for (int i = 0; i < result.Rows; i++)
        {
            for (int j = 0; j < result.Columns; j++)
            {
                for (int k = 0; k < this._matrix.GetLength(1); k++)
                {
                    resultMatrix[i, j] += leftMatrix[i, k] * rightMatrix[k, j];
                }
            }
        }

        return result;
    }

    public static Matrix<T> operator * (Matrix<T> left, Matrix<T> right)
    {
        return left.Multiply(right);
    }

    public List<T> ToList()
    {
        return new List<T>(_matrix.Cast<T>());
    }
}
