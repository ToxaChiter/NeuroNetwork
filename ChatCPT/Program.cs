using ChatCPT;
using Microsoft.VisualBasic.FileIO;
using MNIST.IO;
using System.Diagnostics;

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
        MyNeuroNetwork baseNeuroNetwork = new(28 * 28, new int[] { 80, 16 }, 10);

        // baseSetup configurates basic parameters for all trainings (can be changed manually)
        Setup baseSetup = new() { IsParallel = false, LearningRate = 0.1, Batch = 5 };
        baseSetup.Name =
            $"Setup {baseNeuroNetwork.Inputs} {string.Join(" ", baseNeuroNetwork.Hiddens)} {baseNeuroNetwork.Outputs} " +
            $"{baseSetup.LearningRate:F2} {baseSetup.Batch} {baseSetup.IsParallel} extra"
            ;



        var directory = FileSystem.CombinePath("../../../", $"Tests/{baseSetup.Name}");
        if (Directory.Exists(directory)) throw new ArgumentException($"The {directory} directory already exists");
        FileSystem.CreateDirectory(directory);

        baseSetup.Directory = directory;



        // the most important setings for each network
        List<(MyNeuroNetwork neuroNetwork, Setup setup)> cases = new()
        {
            // the very regular simple network, nothing special
            (new MyNeuroNetwork(baseNeuroNetwork), new Setup(baseSetup)
            {
                Name = "regular", Mode = Mode.Regular, EpochMax = 50, ChangeSetupFromEpoch = null
            }),

            // more advanced simple network
            (new MyNeuroNetwork(baseNeuroNetwork), new Setup(baseSetup)
            {
                Name = "adv-reg", Mode = Mode.AdvancedOutput, EpochMax = 20, ChangeSetupFromEpoch = null, LearningRate = 0.9
            }),

            // the most advanced simple network
            (new MyNeuroNetwork(baseNeuroNetwork), new Setup(baseSetup)
            {
                Name = "advanced", Mode = Mode.Advanced, EpochMax = 20, ChangeSetupFromEpoch = null
            }),



            // not so simple network, it changes its Mode because of current epoch 
            (new MyNeuroNetwork(baseNeuroNetwork), new Setup(baseSetup)
            {
                Name = "adv-reg reg", Mode = Mode.AdvancedOutput, EpochMax = 50, ChangeSetupFromEpoch = (Setup setup, int epoch) =>
                {
                    // the logic of changing Setup (you can change the whole Setup instanse in the way you want to)
                    if (epoch > 5) setup.Mode = Mode.Regular;
                }
            }),

            (new MyNeuroNetwork(baseNeuroNetwork), new Setup(baseSetup)
            {
                Name = "adv reg", Mode = Mode.Advanced, EpochMax = 50, ChangeSetupFromEpoch = (Setup setup, int epoch) =>
                {
                    if (epoch > 5) setup.Mode = Mode.Regular;
                }
            }),



            (new MyNeuroNetwork(baseNeuroNetwork), new Setup(baseSetup)
            {
                Name = "reg adv-reg", Mode = Mode.Regular, EpochMax = 20, ChangeSetupFromEpoch = (Setup setup, int epoch) =>
                {
                    if (epoch > 5) setup.Mode = Mode.AdvancedOutput;
                }
            }),

            (new MyNeuroNetwork(baseNeuroNetwork), new Setup(baseSetup)
            {
                Name = "reg adv", Mode = Mode.Regular, EpochMax = 20, ChangeSetupFromEpoch = (Setup setup, int epoch) =>
                {
                    if (epoch > 5) setup.Mode = Mode.Advanced;
                }
            }),



            (new MyNeuroNetwork(baseNeuroNetwork), new Setup(baseSetup)
            {
                Name = "adv-reg + reg", Mode = Mode.AdvancedOutput, EpochMax = 25, ChangeSetupFromEpoch = (Setup setup, int epoch) =>
                {
                    if (epoch % 2 == 1) setup.Mode = Mode.Regular;
                    else setup.Mode = Mode.AdvancedOutput;
                }
            }),

            (new MyNeuroNetwork(baseNeuroNetwork), new Setup(baseSetup)
            {
                Name = "adv + reg", Mode = Mode.Advanced, EpochMax = 25, ChangeSetupFromEpoch = (Setup setup, int epoch) =>
                {
                    if (epoch % 2 == 1) setup.Mode = Mode.Regular;
                    else setup.Mode = Mode.Advanced;
                }
            }),
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
