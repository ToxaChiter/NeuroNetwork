using BenchmarkDotNet.Running;

namespace Testing;

internal class Program
{
    static void Main()
    {
        // BenchmarkRunner.Run<Enumerator>();

        var array = new int[2, 2] { { 1, 2 }, { 3, 4 } };

        Console.WriteLine(string.Join(", ", array.Cast<int>()));
        Console.WriteLine();

        var copy = new int[2, 2];
        Array.Copy(array, copy, copy.Length);
        array[0, 0] = 0;

        Console.WriteLine(string.Join(", ", copy.Cast<int>()));
    }
}