// See https://aka.ms/new-console-template for more information
using SA_ILP;
using System.Diagnostics;
using Gurobi;
using CommandLine;
using System.Text;
using MathNet.Numerics.Distributions;
using System.Reflection;
using System.Collections.ObjectModel;

string baseDir = "../../../../../";

var solver = new Solver();
Stopwatch watch = new Stopwatch();

//PROBLEM might not want to create new routes it seems like

//var arguments = Environment.GetCommandLineArgs();

if (args.Length >= 1)
{
    //string mode = args[0];
    //string instance = args[1];
    //int timeLimit = 60;
    Options? opts = null;

    Parser.Default.ParseArguments<Options>(args)
      .WithParsed<Options>(o =>
       {
           opts = o;
       });
    if (opts == null)
        return;

    
    //String baseSaveDirectory = "/Users/stijnvanderwal/Documents/GitHub/SA-ILP_new/SA-ILP/SA-ILP/Test8Maa";
    String baseSaveDirectory = "/Users/stijnvanderwal/Documents/GitHub/Data";
    //String baseSaveDirectory = "/Users/stijnvanderwal/Documents/GitHub/Simulator";
    String solutionDir = baseSaveDirectory + "/FinalSchedules";
    //Path.Join(opts.SolutionDir, DateTime.Now.ToString("dd-MM-yy_HH-mm-ss") + opts.TestName);
    String benchDir = Path.Join("Benchmarks", DateTime.Now.ToString("dd-MM-yy_HH-mm-ss") + opts.TestName);
    // Here we need the directory in which we can find the txtfiles
    //String ObservationsDirectory = baseSaveDirectory + "/Test19Mrt/TXTfiles";
    String ObservationsDirectory = "/Users/stijnvanderwal/Documents/GitHub/Data/TXTfilesFullETT";

    if (opts.Mode == "vrpltt") //this does both SA and ILP
    {
        solver.SolveVRPLTTInstance(opts.Instance, numLoadLevels: opts.NumLoadLevels, numIterations: opts.Iterations, timelimit: opts.TimeLimitLS * 1000, bikeMinMass: opts.BikeMinWeight, bikeMaxMass: opts.BikeMaxWeight, inputPower: opts.BikePower);

    }
    else if (opts.Mode == "vrplttKDE") //this does both SA and ILP
    {
        solver.SolveVRPLTTInstanceKDE(opts.Instance, KDEdata.ObservationsMatrix, numLoadLevels: opts.NumLoadLevels, numIterations: opts.Iterations, timelimit: opts.TimeLimitLS * 1000, bikeMinMass: opts.BikeMinWeight, bikeMaxMass: opts.BikeMaxWeight, inputPower: opts.BikePower);
    }
    else if (opts.Mode == "vrptw")
    {
        // this does only SA
        solver.SolveSolomonInstance(opts.Instance, numIterations: opts.Iterations, timeLimit: opts.TimeLimitLS * 1000);
    }
    else if (opts.Mode == "vrplttmt")
    {
        await solver.SolveVRPLTTInstanceAsync(opts.Instance, numLoadLevels: opts.NumLoadLevels, numIterations: opts.Iterations, timelimit: opts.TimeLimitLS * 1000, bikeMinMass: opts.BikeMinWeight, bikeMaxMass: opts.BikeMaxWeight, inputPower: opts.BikePower, numStarts: opts.NumStarts, numThreads: opts.NumThreads, config: LocalSearchConfigs.VRPLTTFinal);
    }
    else if (opts.Mode == "vrptwmt")
    {
        await solver.SolveSolomonInstanceAsync(opts.Instance, numIterations: opts.Iterations, timeLimit: opts.TimeLimitLS * 1000, numStarts: opts.NumStarts, numThreads: opts.NumThreads);
    }
    else if (opts.Mode == "allvrpltt")
    {
        if (opts.Config != "")
        {
            PropertyInfo propertyInfo = typeof(LocalSearchConfigs).GetProperty(opts.Config);
            LocalSearchConfiguration something = (LocalSearchConfiguration)propertyInfo.GetValue(null, null);
            await Tests.RunVRPLTTTests(baseSaveDirectory + opts.Instance, solutionDir, opts.NumRepeats, opts, something);
        }
        else
        {
            await Tests.RunVRPLTTTests(baseSaveDirectory + opts.Instance, solutionDir, opts.NumRepeats, opts);
        }
    }
    else if (opts.Mode == "allvrplttKDE")
    {
        if (opts.Config != "")
        {
            PropertyInfo propertyInfo = typeof(LocalSearchConfigs).GetProperty(opts.Config);
            LocalSearchConfiguration something = (LocalSearchConfiguration)propertyInfo.GetValue(null, null);
            await Tests.RunVRPLTTTestsKDE(baseSaveDirectory+ opts.Instance, ObservationsDirectory, solutionDir, opts.NumRepeats, opts, something);
        }
        else
        {
            await Tests.RunVRPLTTTestsKDE(opts.Instance, ObservationsDirectory, solutionDir, opts.NumRepeats, opts);
        }
    }
    else if (opts.Mode == "KDETests")
    {
        /*
        MAKE SURE THE "baseSaveDirectory" IS SET CORRECTLY AND ALL FOLDERS ARE IN THE SAME DIRECTORY
        */
        if (opts.Config != "")
        {
            PropertyInfo propertyInfo = typeof(LocalSearchConfigs).GetProperty(opts.Config);
            LocalSearchConfiguration something = (LocalSearchConfiguration)propertyInfo.GetValue(null, null);
            await Tests.KDETesting(baseSaveDirectory + opts.Instance, ObservationsDirectory, solutionDir, opts.NumRepeats, opts, something);
        }
        else
        {
            await Tests.KDETesting(opts.Instance, ObservationsDirectory, solutionDir, opts.NumRepeats, opts);
        }
    }
    else if (opts.Mode == "KDETestParams")
    {
        /*
        MAKE SURE THE "baseSaveDirectory" IS SET CORRECTLY AND ALL FOLDERS ARE IN THE SAME DIRECTORY
        */
        opts.TimeLimitILP = 10;
        if (opts.Config != "")
        {
            PropertyInfo propertyInfo = typeof(LocalSearchConfigs).GetProperty(opts.Config);
            LocalSearchConfiguration something = (LocalSearchConfiguration)propertyInfo.GetValue(null, null);
            await Tests.KDETestingParams(baseSaveDirectory + opts.Instance, ObservationsDirectory, solutionDir, opts.NumRepeats, opts, something);
        }
        else
        {
            await Tests.KDETestingParams(opts.Instance, ObservationsDirectory, solutionDir, opts.NumRepeats, opts);
        }
    }
    else if (opts.Mode == "KDETestWithWeights")
    {
        if (opts.Config != "")
        {
            PropertyInfo propertyInfo = typeof(LocalSearchConfigs).GetProperty(opts.Config);
            LocalSearchConfiguration something = (LocalSearchConfiguration)propertyInfo.GetValue(null, null);
            await Tests.KDETestingWithWeights(opts.Instance, ObservationsDirectory, solutionDir, opts.NumRepeats, opts, something);
        }
        else
        {
            await Tests.KDETestingWithWeights(opts.Instance, ObservationsDirectory, solutionDir, opts.NumRepeats, opts);
        }
    }
    else if (opts.Mode == "allvrpsltt")
    {
        await Tests.RunVRPSLTTTests(opts.Instance, solutionDir, opts.NumRepeats, opts);
    }
    else if (opts.Mode == "allvrplttWind")
    {
        await Tests.RunVRPLTTWindtests(baseSaveDirectory+ opts.Instance, solutionDir, opts.NumRepeats, opts);
    }
    else if (opts.Mode == "allvrp")
    {
        var skip = new List<string>();
        await Tests.SolveAllAsync(opts.Instance, solutionDir, skip, opts);
    }
    else if (opts.Mode == "benchmark")
    {
        await Tests.BenchmarkLocalSearchSpeed(opts.Instance, benchDir, opts.NumRepeats, opts);
    }
    else if (opts.Mode == "compareoperators")
    {
        await Tests.TestAllOperatorConfigurations(opts.Instance, solutionDir, opts);
    }
    else if(opts.Mode == "lastvrpsltt")
    {
        await Tests.RunLastVRPSLTTTests(opts.Instance, solutionDir, opts.NumRepeats, opts);
    }
    else
    {
        Console.WriteLine("Enter correct mode");
        return;
    }

    return;
}


//Used to analyze results planned without any wind for al directionss
static void Anaylzyze()
{
    string baseDir = "../../../../../";

    using (var csvwriter = new StreamWriter("out.csv"))
    {
        csvwriter.WriteLine("instance;WT(1,0);WT(0,-1);WT(0,1);WT(-1,0);V(1,0);V(0,-1);V(0,1);V(-1,0);T(1,0);T(0,-1);T(0,1);T(-1,0)");
        foreach (var file in Directory.GetFiles(@"C:\Users\samca\Desktop\Test"))
        {
            var fn = Path.GetFileNameWithoutExtension(file);

            var split = fn.Split('_');

            var instanceName = $"{split[0]}_{split[1]}.csv";
            var instance = Path.Join(baseDir, "vrpltt_instances/large", instanceName);
            //var config = LocalSearchConfigs.VRPLTTWithWind;
            var parsed = VRPLTT.ParseVRPLTTInstance(instance);
            var sol = new List<List<int>>();
            using (var reader = new StreamReader(file))
            {
                reader.ReadLine();
                string line;
                while ((line = reader.ReadLine()) != null && line[0] != '{')
                {

                    //var route = new Route(parsed.customers[0],null,);
                    //lines.Add(line);
                    var a = line.Replace("(", "").Replace(")", "");
                    var b = a.Split(',').ToList().ConvertAll(x => int.Parse(x));
                    sol.Add(b);
                }
            }


            var winddir1 = new double[] { 1, 0 };
            var winddir2 = new double[] { 0, -1 };
            var winddir3 = new double[] { 0, 1 };
            var winddir4 = new double[] { -1, 0 };
            var route = new List<Route>();
            var windresult1 = VRPLTT.CalculateWindCyclingTime(instance, 140, 290, 10, 350, winddir1, route, sol);
            var windresult2 = VRPLTT.CalculateWindCyclingTime(instance, 140, 290, 10, 350, winddir2, new List<Route>(), sol);
            var windresult3 = VRPLTT.CalculateWindCyclingTime(instance, 140, 290, 10, 350, winddir3, new List<Route>(), sol);
            var windresult4 = VRPLTT.CalculateWindCyclingTime(instance, 140, 290, 10, 350, winddir4, new List<Route>(), sol);


            var dirs = new List<double[]>() { winddir1, winddir2, winddir3, winddir4 };
            var resulst = new List<double>();
            var results2 = new List<double>();
            foreach (var dir in dirs)
            {
                var tempConfig = LocalSearchConfigs.VRPLTTWithWind;
                tempConfig.WindDirection = dir;
                tempConfig.WindSpeed = 6.75;

                tempConfig.PenalizeLateArrival = false;
                tempConfig.PenalizeDeterministicEarlyArrival = false;
                //If the windspeed is 0 we want to check what the travel time would be if we plan without it
                (double[,] distances, List<Customer> customers) = VRPLTT.ParseVRPLTTInstance(instance);

                Console.WriteLine("Calculating travel time matrix");
                (double[,,] matrix, Gamma[,,] distributionMatrix, IContinuousDistribution[,,] approximationMatrix, _) = VRPLTT.CalculateLoadDependentTimeMatrix(customers, distances, 140, 290, 10, 350, tempConfig.WindSpeed, tempConfig.WindDirection, tempConfig.DefaultDistribution is Normal);
                LocalSearch ls = new LocalSearch(tempConfig, 1);
                double total = 0;
                ;
                int valid = route.Count;
                foreach (var r in route)
                {
                    var rs = new RouteStore(r.CreateIdList(), 0);
                    var newRoute = new Route(customers, rs, customers[0], matrix, distributionMatrix, approximationMatrix, 150, ls);
                    total += newRoute.CalcObjective();
                    if (newRoute.CheckRouteValidity())
                        valid--;
                }
                resulst.Add((double)valid / route.Count);
                results2.Add(total);
            }
            //cycleSpeedWithWind = total;
            csvwriter.WriteLine($"{Path.GetFileNameWithoutExtension(instance)};{windresult1};{windresult2};{windresult3};{windresult4};{resulst[0]};{resulst[1]};{resulst[2]};{resulst[3]};{results2[0]};{results2[1]};{results2[2]};{results2[3]}");


        }
    }

}


async Task RunTestAsync()
{
    double total = 0;
    double best = double.MaxValue;
    double worst = double.MinValue;
    int num = 10;
    watch.Start();
    //LAPTOP
    //Zonder reverse operator sydney:                 Average score: 145,097, best score: 139,5727, Total time: 324,000
    //Met reverse operator sydney:                    Average score: 143,894, best score: 141,1361, Total time: 326,000
    //Met nieuwe move operator sydney:  Finsihed all. Average score: 143,777, best score: 140,0621, Total time: 325,000
    //Met 4 threads, zelfde timelimit:  Finsihed all. Average score: 143,849, best score: 140,196, worst score: 158,264, Total time: 345,000
    //Met 4 threads, 45s timelimit:     Finsihed all. Average score: 142,300, best score: 139,281, worst score: 144,666, Total time: 505,000

    //avg 600

    //Madrid full
    //Met greedy move op                Average score: 586,591, best score: 548,0854, Total time: 325,000


    //DESKTOP
    //Seattle full
    //Zonder greedy move op             Finsihed all. Average score: 459,158, best score: 400,1997, Total time: 341,000 WAARSCHJNLIJK NIET VALID
    //Met greedy move op                Finsihed all. Average score: 398,136, best score: 394,7354, Total time: 340,000 WAARSCHJNLIJK NIET VALID
    //Met greedy move op (less chance)  Finsihed all. Average score: 397,662, best score: 393,0073, Total time: 340,000 WAARSCHJNLIJK NIET VALID



    for (int i = 0; i < num; i++)
    {
        (bool failed, List<Route> ilpSol, double ilpVal, double ilpTime, double lsTime, double lsVal, string solutionJSON) = await solver.SolveVRPLTTInstanceAsync(Path.Join(baseDir, "vrpltt_instances/large", "madrid_full.csv"), numLoadLevels: 10, numIterations: 50000000, timelimit: 30000);
        if (ilpVal < best)
            best = ilpVal;
        if (ilpVal > worst)
            worst = ilpVal;
        total += ilpVal;
    }
    Console.WriteLine($"Finsihed all. Average score: {(total / num).ToString("0.000")}, best score: {best.ToString("0.000")}, worst score: {worst.ToString("0.000")}, Total time: {(watch.ElapsedMilliseconds / 1000).ToString("0.000")}");
}



class Options
{
    //[Option('r', "read", Required = true, HelpText = "Input files to be processed.")]
    //public IEnumerable<string> InputFiles { get; set; }

    //// Omitting long name, defaults to name of property, ie "--verbose"
    //[Option(
    //  Default = false,
    //  HelpText = "Prints all messages to standard output.")]
    //public bool Verbose { get; set; }

    //[Option("stdin",
    //  Default = false,
    //  HelpText = "Read from stdin")]
    //public bool stdin { get; set; }

    //[Value(0, MetaName = "offset", HelpText = "File offset.")]
    //public long? Offset { get; set; }

    [Value(0, MetaName = "mode", HelpText = "Program mode.\n\tOptions:\n\t\t- vrpltt\n\t\t- vrptw\n\t\t- vrplttmt\n\t\t- vrptwmt\n\t\t- allvrpltt\n\t\t- allvrpsltt\n\t\t- allvrplttWind")]
    public string Mode { get; set; }

    [Value(1, MetaName = "instance", HelpText = "Path to instance")]
    public string Instance { get; set; }

    [Option("iterations", Default = 50000000, HelpText = "LS iteration limit.")]
    public int Iterations { get; set; }

    [Option("threads", Default = 4, HelpText = "Number of threads in multithreaded modes.")]
    public int NumThreads { get; set; }

    [Option("starts", Default = 4, HelpText = "Number of starts in multithreaded modes.")]
    public int NumStarts { get; set; }

    [Option("lstl", Default = 60, HelpText = "LS timelimit.")]
    public int TimeLimitLS { get; set; }

    [Option("ilptl", Default = 3600, HelpText = "ILP timelimit.")]
    public int TimeLimitILP { get; set; }

    [Option("power", Default = 350, HelpText = "Input power.")]
    public int BikePower { get; set; }
    [Option("minweight", Default = 140, HelpText = "Min bikeweight in vrpltt.")]
    public int BikeMinWeight { get; set; }

    [Option("maxweight", Default = 290, HelpText = "Max bikeweight in vrpltt.")]
    public int BikeMaxWeight { get; set; }

    [Option("loadlevels", Default = 150, HelpText = "Number of loadlevels in vrpltt.")]
    public int NumLoadLevels { get; set; }

    [Option("repeats", Default = 5, HelpText = "Number of reptitions per instance for the selected tests. This only works for the collection of tests")]
    public int NumRepeats { get; set; }

    [Option("testname", Default = "", HelpText = "Name of the test currently ran")]
    public string TestName { get; set; }

    [Option("solutiondir", Default = "Solutions", HelpText = "Directory for solutions files")]
    public string SolutionDir { get; set; }

    [Option("config", Default = "", HelpText = "The configuration of LS settings")]
    public string Config { get; set; }
    public override string ToString()
    {
        return GetType().GetProperties()
    .Select(info => (info.Name, Value: info.GetValue(this, null) ?? "(null)"))
    .Aggregate(
        new StringBuilder(),
        (sb, pair) => sb.AppendLine($"{pair.Name}: {pair.Value}"),
        sb => sb.ToString());
    }
}
