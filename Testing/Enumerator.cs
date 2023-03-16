using BenchmarkDotNet.Attributes;

namespace Testing;

public class Enumerator
{
    byte[,] bytes = GetBytes();

    public static byte[,] GetBytes()
    {
        var random = new Random(5);
        var bytes = new byte[100, 100];
        for (int i = 0; i < 100; i++)
        {
            for (int j = 0; j < 100; j++)
            {
                bytes[i, j] = (byte)random.Next(255);
            }
        }

        return bytes;
    }


    [Benchmark]
    public int ForEachSum()
    {
        int sum = 0;
        foreach (var i in bytes)
        {
            sum += i;
        }
        return sum;
    }

    [Benchmark(Baseline = true)]
    public int ForSum()
    {
        int sum = 0;
        for (int i = 0; i < 100; i++)
        {
            for (int j = 0; j < 100; j++)
            {
                sum += bytes[i, j];
            }
        }
        return sum;
    }

    [Benchmark]
    public int EnumeratorSum()
    {
        int sum = 0;
        var enumerator = bytes.GetEnumerator();
        while (enumerator.MoveNext())
        {
            sum += (byte)enumerator.Current;
        }
        return sum;
    }
}
