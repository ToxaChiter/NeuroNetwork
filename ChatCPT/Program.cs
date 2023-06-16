using ChatCPT;
using Microsoft.VisualBasic.FileIO;
using MNIST.IO;
using System.Diagnostics;

class Program
{
    static List<(double[] Labels, double[] Image)> trainCases = new(60_000);
    static List<(int Label, double[] Image)> checkCases = new(10_000);

    static int Repeats;

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

        Console.WriteLine("Performed");



        // baseNeuroNetwork configurates the basic architecture for other networks (can be changed manually)
        MyNeuroNetwork baseNeuroNetwork = new(28 * 28, new int[] { 200, 80, 16 }, 10);

        // baseSetup configurates basic parameters for all trainings (can be changed manually)
        Setup baseSetup = new() { IsParallel = false, LearningRate = 0.1, Batch = 100 };
        baseSetup.Name =
            $"Setup {baseNeuroNetwork.Inputs} {string.Join(" ", baseNeuroNetwork.Hiddens)} {baseNeuroNetwork.Outputs} " +
            $"{baseSetup.LearningRate:F2} {baseSetup.Batch} {baseSetup.IsParallel} avg parallel"
            ;


        Repeats = 1;


        var directory = FileSystem.CombinePath("../../../", $"Tests/{baseSetup.Name}");
        if (Directory.Exists(directory)) throw new ArgumentException($"The {directory} directory already exists");
        FileSystem.CreateDirectory(directory);

        baseSetup.Directory = directory;



        // the most important setings for each network
        List<(MyNeuroNetwork neuroNetwork, Setup setup)> cases = new()
        {
            //// the very regular simple network, nothing special
            (new MyNeuroNetwork(baseNeuroNetwork), new Setup(baseSetup)
            {
                Name = "regular", Mode = Mode.Regular, EpochMax = 25, ChangeSetupFromEpoch = null
            }),

            // more advanced simple network
            (new MyNeuroNetwork(baseNeuroNetwork), new Setup(baseSetup)
            {
                Name = "adv-reg", Mode = Mode.AdvancedOutput, EpochMax = 25, ChangeSetupFromEpoch = null
            }),

            //// the most advanced simple network
            (new MyNeuroNetwork(baseNeuroNetwork), new Setup(baseSetup)
            {
                Name = "advanced", Mode = Mode.Advanced, EpochMax = 25, ChangeSetupFromEpoch = null
            }),



            //// not so simple network, it changes its Mode because of current epoch 
            //(new MyNeuroNetwork(baseNeuroNetwork), new Setup(baseSetup)
            //{
            //    Name = "adv-reg reg", Mode = Mode.AdvancedOutput, EpochMax = 50, ChangeSetupFromEpoch = (Setup setup, int epoch) =>
            //    {
            //        // the logic of changing Setup (you can change the whole Setup instanse in the way you want to)
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

        var avgCases = new List<List<(MyNeuroNetwork neuroNetwork, Setup setup)>>();
        foreach (var (neuroNetwork, setup) in cases)
        {
            var list = new List<(MyNeuroNetwork neuroNetwork, Setup setup)>(Repeats);
            for (int i = 0; i < Repeats; i++)
            {
                list.Add((new MyNeuroNetwork(neuroNetwork), setup.Copy()));
            }
            avgCases.Add(list);
        }

        Stopwatch sw = Stopwatch.StartNew();
        foreach (var item in avgCases)
        {
            var list = new List<List<(MyNeuroNetwork neuroNetwork, Setup setup)>>() { item };
            var result = Parallel.ForEach(list, TrainWithSetupAvg);
            Console.WriteLine("Next");
        }

        sw.Stop();
        Console.WriteLine($"Time spent: {sw.Elapsed} (x{Repeats})");

        //foreach (var item in cases)
        //{
        //    Thread thread = new Thread(() => TrainWithSetup(item));
        //    thread.Start();
        //}
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


    static void TrainWithSetupAvg(List<(MyNeuroNetwork neuroNetwork, Setup setup)> @case)
    {
        var stream = File.OpenWrite($"{@case[0].setup.Directory}/{@case[0].setup.Name}.txt");
        StreamWriter writer = new(stream);

        Stopwatch stopwatch = new();

        try
        {
            for (int epoch = 1; epoch < @case[0].setup.EpochMax + 1; epoch++)
            {
                var error = 0.0;
                stopwatch.Reset();
                stopwatch.Start();
                for (int i = 0; i < Repeats; i++)
                {
                    var neuroNetwork = @case[i].neuroNetwork;
                    var setup = @case[i].setup;

                    error += neuroNetwork.Train(trainCases, setup.Mode, setup.LearningRate, setup.Batch);

                    setup.ChangeSetup(epoch);
                }
                stopwatch.Stop();

                var evalTest = 0.0;
                var evalTrain = 0.0;
                for (int i = 0; i < Repeats; i++)
                {
                    var neuroNetwork = @case[i].neuroNetwork;
                    var setup = @case[i].setup;

                    evalTest += neuroNetwork.Evaluate(checkCases);
                    evalTrain += neuroNetwork.Evaluate(trainCases);
                }

                var str =
                    $"Epoch #{epoch}\n" +
                    $"Error - {error / Repeats:F3} ({stopwatch.Elapsed / Repeats})\n" +
                    $"Precision - {evalTest / Repeats}% ({evalTrain / Repeats:F3}%)\n\n";

                Console.WriteLine($"#{epoch} - {stopwatch.Elapsed / Repeats}");
                writer.WriteLine(str);
            }
            Console.WriteLine("End");
            writer.Close();
        }
        finally
        {
            writer.Close();
        }
    }
}
