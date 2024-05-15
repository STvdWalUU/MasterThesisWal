using MathNet.Numerics.Distributions;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.Tracing;
using System.IO.Enumeration;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SA_ILP
{
    internal static class Tests
    {
        static Solver solver = new Solver();
        public static async Task RunVRPLTTTests(string dir, string solDir, int numRepeats, Options opts,LocalSearchConfiguration? configInput = null)
        {
            // this is run during mode 'allvrpltt'
            Console.WriteLine("Testing on all vrpltt instances");
            if (!Directory.Exists(solDir))
                Directory.CreateDirectory(solDir);

            using (var totalWriter = new StreamWriter(Path.Join(solDir, $"allSolutionsVRPLTT{opts.TestName}.txt")))
            {
                totalWriter.WriteLine("Command line arguments:");
                totalWriter.WriteLine(opts.ToString());
                LocalSearchConfiguration config;
                
                if(configInput == null)
                 config = LocalSearchConfigs.VRPLTTFinal;
                else
                    config = (LocalSearchConfiguration)configInput;

                totalWriter.WriteLine("Config:");
                totalWriter.WriteLine(config);
                totalWriter.Flush();
                using (var csvWriter = new StreamWriter(Path.Join(solDir, $"allSolutionsVRPLTT{opts.TestName}.csv")))
                {
                    csvWriter.WriteLine("SEP=;");
                    csvWriter.WriteLine("Instance;AllowWaiting;Score;N;ILP time;LS time;LS score;ILP imp(%)");

                    foreach (var file in Directory.GetFiles(dir))
                    {
                        Console.WriteLine($"Testing on { Path.GetFileNameWithoutExtension(file)}");
                        double totalValue = 0.0;
                        //bool newInstance = false;
                        
                        if (Path.GetExtension(file) != ".csv")
                            continue;

                        for (int i = 0; i < numRepeats; i++)
                        {
                            string fileNameNow = Path.Join(solDir, Path.GetFileNameWithoutExtension(file) + $"_NoWind_{i}.txt");
                            if (File.Exists(fileNameNow))
                            {
                                Console.WriteLine($"Already done {fileNameNow}");
                                //continue;
                            }
                            (bool failed, List<Route> ilpSol, double ilpVal, double ilpTime, double lsTime, double lsVal, string solutionJSON) = await solver.SolveVRPLTTInstanceAsync(file, numLoadLevels: opts.NumLoadLevels, numIterations: opts.Iterations, timelimit: opts.TimeLimitLS * 1000, bikeMinMass: opts.BikeMinWeight, bikeMaxMass: opts.BikeMaxWeight, inputPower: opts.BikePower, numStarts: opts.NumStarts, numThreads: opts.NumThreads, config: config,ilpTimelimit:opts.TimeLimitILP);//solver.SolveSolomonInstanceAsync(file, numThreads: numThreads, numIterations: numIterations, timeLimit: 30 * 1000);
                            using (var writer = new StreamWriter(fileNameNow))
                            {
                                if (failed)
                                    writer.Write("FAIL did not meet check");
                                writer.WriteLine($"Score: {ilpVal}, ilpTime: {ilpTime}, lsTime: {lsTime}, Early arrival allowed {config.AllowDeterministicEarlyArrival}");
                                foreach (var route in ilpSol)
                                {
                                    writer.WriteLine($"{route}");
                                }
                                writer.WriteLine(solutionJSON);
                            }
                            if (failed)
                                totalWriter.Write("FAIL did not meet check");

                            totalValue += ilpVal;

                            totalWriter.WriteLine($"Instance: { Path.GetFileNameWithoutExtension(file)},Early arrival allowed {config.AllowDeterministicEarlyArrival}, Score: {Math.Round(ilpVal, 3)}, Vehicles: {ilpSol.Count}, ilpTime: {Math.Round(ilpTime, 3)}, lsTime: {Math.Round(lsTime, 3)}, lsVal: {Math.Round(lsVal, 3)}, ilpImp: {Math.Round((lsVal - ilpVal) / lsVal * 100, 3)}%");
                            csvWriter.WriteLine($"{ Path.GetFileNameWithoutExtension(file)};{config.AllowDeterministicEarlyArrival};{Math.Round(ilpVal, 3)};{ilpSol.Count};{Math.Round(ilpTime, 3)};{Math.Round(lsTime, 3)};{Math.Round(lsVal, 3)};{Math.Round((lsVal - ilpVal) / lsVal * 100, 3)}");
                            totalWriter.Flush();
                            csvWriter.Flush();
                        }
                    }
                }
            }
        }
        public static async Task RunVRPLTTTestsKDE(string CSVdirectory, string ObservationsDirectory, string solDir, int numRepeats, Options opts,LocalSearchConfiguration? configInput = null)
        {
            // this function needs to match .csv files with .txt files in order to 
            // get customers from csv and the distances from the txt
            // this is run during mode 'allvrplttkde'
            Console.WriteLine("......Testing on all vrpltt instances......");
            if (!Directory.Exists(solDir))
                Directory.CreateDirectory(solDir);

            using (var totalWriter = new StreamWriter(Path.Join(solDir, $"allSolutionsVRPLTT{opts.TestName}.txt")))
            {
                totalWriter.WriteLine("Command line arguments:");
                totalWriter.WriteLine(opts.ToString());
                LocalSearchConfiguration config;
                
                if(configInput == null)
                 config = LocalSearchConfigs.VRPLTTFinal;
                else
                    config = (LocalSearchConfiguration)configInput;

                totalWriter.WriteLine("Config:");
                totalWriter.WriteLine(config);
                totalWriter.Flush();
                using (var csvWriter = new StreamWriter(Path.Join(solDir, $"allSolutionsVRPLTT{opts.TestName}.csv")))
                {
                    csvWriter.WriteLine("SEP=;");
                    csvWriter.WriteLine("Instance;AllowWaiting;Score;N;ILP time;LS time;LS score;ILP imp(%)");

                    // This is the loop over the csv files that represent the different instances.
                    foreach (var CSVfile in Directory.GetFiles(CSVdirectory))
                    {
                        if (Path.GetExtension(CSVfile) != ".csv")
                        {
                            Console.WriteLine($"{ Path.GetFileNameWithoutExtension(CSVfile)} is not an instance.");
                            continue;
                        }
                        foreach (var txtFile in Directory.GetFiles(ObservationsDirectory))
                        {
                            if (!Path.GetFileNameWithoutExtension(txtFile).Contains($"{Path.GetFileNameWithoutExtension(CSVfile)}") || Path.GetExtension(txtFile)!= ".txt")
                            {
                                continue;
                            }
                            Console.WriteLine($".....Testing on {Path.GetFileNameWithoutExtension(txtFile)}.....");
                            
                            // Here we find the corresponding .txt file, assuming that it has the same name.
                            string fileNameTXT = Path.Join(ObservationsDirectory,$"{Path.GetFileNameWithoutExtension(txtFile)}.txt");
                            double[,,,] observationsTXT = null;
                            try
                            {
                                observationsTXT= KDEdata.ConvertTXTtoMatrix(fileNameTXT);
                            }
                            catch{
                                Console.WriteLine($"FAIL: Cannot find {fileNameTXT}");
                                continue;
                            }

                            double totalValue = 0.0;
                            Console.WriteLine($"Starting local search with txtfile {txtFile}");
                            for (int i = 0; i < numRepeats; i++)
                            {
                                // the 'failed' boolean needs to be looked at
                                (bool failed, List<Route> ilpSol, double ilpVal, double ilpTime, double lsTime, double lsVal, string solutionJSON) = await solver.SolveVRPLTTInstanceAsyncKDE(CSVfile, observationsTXT, numLoadLevels: opts.NumLoadLevels, numIterations: opts.Iterations, timelimit: opts.TimeLimitLS * 1000, bikeMinMass: opts.BikeMinWeight, bikeMaxMass: opts.BikeMaxWeight, inputPower: opts.BikePower, numStarts: opts.NumStarts, numThreads: opts.NumThreads, config: config,ilpTimelimit:opts.TimeLimitILP);//solver.SolveSolomonInstanceAsync(file, numThreads: numThreads, numIterations: numIterations, timeLimit: 30 * 1000);
                                string fileNameNow = Path.Join(solDir, Path.GetFileNameWithoutExtension(txtFile) + $"_{i}.txt");
                                if (File.Exists(fileNameNow))
                                    continue;
                                using (var writer = new StreamWriter(fileNameNow))
                                {
                                    if (failed)
                                        writer.Write("FAIL: did not meet check");
                                    writer.WriteLine($"Score: {ilpVal}, ilpTime: {ilpTime}, lsTime: {lsTime}, Early arrival allowed {config.AllowDeterministicEarlyArrival}");
                                    foreach (var route in ilpSol)
                                    {
                                        writer.WriteLine($"{route}");
                                    }
                                    writer.WriteLine(solutionJSON);
                                }
                                if (failed)
                                    totalWriter.Write("FAIL did not meet check");

                                totalValue += ilpVal;
                                totalWriter.WriteLine($"Instance: { Path.GetFileNameWithoutExtension(txtFile)},Early arrival allowed {config.AllowDeterministicEarlyArrival}, Score: {Math.Round(ilpVal, 3)}, Vehicles: {ilpSol.Count}, ilpTime: {Math.Round(ilpTime, 3)}, lsTime: {Math.Round(lsTime, 3)}, lsVal: {Math.Round(lsVal, 3)}, ilpImp: {Math.Round((lsVal - ilpVal) / lsVal * 100, 3)}%");
                                csvWriter.WriteLine($"{ Path.GetFileNameWithoutExtension(txtFile)};{config.AllowDeterministicEarlyArrival};{Math.Round(ilpVal, 3)};{ilpSol.Count};{Math.Round(ilpTime, 3)};{Math.Round(lsTime, 3)};{Math.Round(lsVal, 3)};{Math.Round((lsVal - ilpVal) / lsVal * 100, 3)}");
                                totalWriter.Flush();
                                csvWriter.Flush();
                            }
                        }
                    }
                }
            }
        }
        public static async Task KDETesting(string CSVdirectory, string ObservationsDirectory, string solDir, int numRepeats, Options opts,LocalSearchConfiguration? configInput = null)
        {
            // This function is designed for trying different settings for the KDE method
            // This is only used to find the ideal settings
            Console.WriteLine("......Testing on all vrpltt instances......");
            if (!Directory.Exists(solDir))
                Directory.CreateDirectory(solDir);

            using (var totalWriter = new StreamWriter(Path.Join(solDir, $"allSolutionsVRPLTT{opts.TestName}.txt")))
            {
                totalWriter.WriteLine("Command line arguments:");
                totalWriter.WriteLine(opts.ToString());
                LocalSearchConfiguration config;
                
                if(configInput == null)
                 config = LocalSearchConfigs.VRPLTTFinal;
                else
                    config = (LocalSearchConfiguration)configInput;

                totalWriter.WriteLine("Config:");
                totalWriter.WriteLine(config);
                totalWriter.Flush();
                using (var csvWriter = new StreamWriter(Path.Join(solDir, $"allSolutionsVRPLTT{opts.TestName}.csv")))
                {
                    csvWriter.WriteLine("SEP=;");
                    csvWriter.WriteLine("Instance;AllowWaiting;Score;N;ILP time;LS time;LS score;ILP imp(%)");

                    // This is the loop over the csv files that represent the different instances.
                    foreach (var CSVfile in Directory.GetFiles(CSVdirectory))
                    {
                        if (Path.GetExtension(CSVfile) != ".csv")
                        {
                            Console.WriteLine($"{ Path.GetFileNameWithoutExtension(CSVfile)} is not an instance.");
                            continue;
                        }
                        foreach (var txtFile in Directory.GetFiles(ObservationsDirectory))
                        {
                            if (!Path.GetFileNameWithoutExtension(txtFile).Contains($"{Path.GetFileNameWithoutExtension(CSVfile)}") || Path.GetExtension(txtFile)!= ".txt")
                            {
                                // This if statement is designed to skip unmatched txtfiles and csvfiles 
                                // i.e. we skip files that refer to different instances
                                continue;
                            }
                            Console.WriteLine($".....Testing on {Path.GetFileNameWithoutExtension(txtFile)}.....");
                            // Here we find the corresponding .txt file, assuming that it has the same name.
                            string fileNameTXT = Path.Join(ObservationsDirectory,$"{Path.GetFileNameWithoutExtension(txtFile)}.txt");
                            double[,,,] observationsTXT = null;
                            try
                            {
                                observationsTXT= KDEdata.ConvertTXTtoMatrix(fileNameTXT);
                            }
                            catch{
                                Console.WriteLine($"FAIL: Cannot find {fileNameTXT}");
                                continue;
                            }

                            double totalValue = 0.0;
                            Console.WriteLine($"Starting local search with txtfile {txtFile}");
                            for (int i = 0; i < numRepeats; i++)
                            {
                                // the 'failed' boolean needs to be looked at
                                (bool failed, List<Route> ilpSol, double ilpVal, double ilpTime, double lsTime, double lsVal, string solutionJSON) = await solver.SolveVRPLTTInstanceAsyncKDE(CSVfile, observationsTXT, numLoadLevels: opts.NumLoadLevels, numIterations: opts.Iterations, timelimit: opts.TimeLimitLS * 1000, bikeMinMass: opts.BikeMinWeight, bikeMaxMass: opts.BikeMaxWeight, inputPower: opts.BikePower, numStarts: opts.NumStarts, numThreads: opts.NumThreads, config: config,ilpTimelimit:opts.TimeLimitILP);//solver.SolveSolomonInstanceAsync(file, numThreads: numThreads, numIterations: numIterations, timeLimit: 30 * 1000);
                                using (var writer = new StreamWriter(Path.Join(solDir, Path.GetFileNameWithoutExtension(txtFile) + $"_{i}.txt")))
                                {
                                    if (failed)
                                        writer.Write("FAIL: did not meet check");
                                    writer.WriteLine($"Score: {ilpVal}, ilpTime: {ilpTime}, lsTime: {lsTime}, Early arrival allowed {config.AllowDeterministicEarlyArrival}");
                                    foreach (var route in ilpSol)
                                    {
                                        writer.WriteLine($"{route}");
                                    }
                                    writer.WriteLine(solutionJSON);
                                }
                                if (failed)
                                    totalWriter.Write("FAIL did not meet check");

                                totalValue += ilpVal;

                                totalWriter.WriteLine($"Instance: { Path.GetFileNameWithoutExtension(txtFile)},Early arrival allowed {config.AllowDeterministicEarlyArrival}, Score: {Math.Round(ilpVal, 3)}, Vehicles: {ilpSol.Count}, ilpTime: {Math.Round(ilpTime, 3)}, lsTime: {Math.Round(lsTime, 3)}, lsVal: {Math.Round(lsVal, 3)}, ilpImp: {Math.Round((lsVal - ilpVal) / lsVal * 100, 3)}%");
                                csvWriter.WriteLine($"{ Path.GetFileNameWithoutExtension(txtFile)};{config.AllowDeterministicEarlyArrival};{Math.Round(ilpVal, 3)};{ilpSol.Count};{Math.Round(ilpTime, 3)};{Math.Round(lsTime, 3)};{Math.Round(lsVal, 3)};{Math.Round((lsVal - ilpVal) / lsVal * 100, 3)}");
                                totalWriter.Flush();
                                csvWriter.Flush();
                            }
                        }
                    }
                }
            }
        }
        public static async Task KDETestingParams(string CSVdirectory, string ObservationsDirectory, string solDir, int numRepeats, Options opts,LocalSearchConfiguration? configInput = null)
        {
            // This function is designed for trying different settings for the KDE method
            // This is only used to find the ideal settings
            Console.WriteLine("......Testing on all vrpltt instances......");
            if (!Directory.Exists(solDir))
                Directory.CreateDirectory(solDir);

            using (var totalWriter = new StreamWriter(Path.Join(solDir, $"allSolutionsVRPLTT{opts.TestName}.txt")))
            {
                totalWriter.WriteLine("Command line arguments:");
                totalWriter.WriteLine(opts.ToString());
                LocalSearchConfiguration config;
                
                if(configInput == null)
                 config = LocalSearchConfigs.VRPLTTFinal;
                else
                    config = (LocalSearchConfiguration)configInput;
                
                //string[] kernels = new string[] {"Epanechnikov","None"};
                //string[] bandwidths = new string[] {"fixed", "AMISE", "gamma"};

                // This phrasing ensures the code can also run on .Net6.0
                string[] kernels = new string[] {"Epanechnikov", "None"};
                string[] bandwidths = new string[] {"AMISE", "fixed", "gamma"};

                totalWriter.Flush();
                using (var csvWriter = new StreamWriter(Path.Join(solDir, $"allSolutionsVRPLTT{opts.TestName}.csv")))
                {
                    csvWriter.WriteLine("SEP=;");
                    csvWriter.WriteLine("Instance;AllowWaiting;Score;N;ILP time;LS time;LS score;ILP imp(%)");

                    // This is the loop over the csv files that represent the different instances.
                    foreach (var CSVfile in Directory.GetFiles(CSVdirectory))
                    {
                        if (Path.GetExtension(CSVfile) != ".csv")
                        {
                            Console.WriteLine($"{ Path.GetFileNameWithoutExtension(CSVfile)} is not an instance.");
                            continue;
                        }
                        foreach (var txtFile in Directory.GetFiles(ObservationsDirectory))
                        {
                            if (!Path.GetFileNameWithoutExtension(txtFile).Contains($"{Path.GetFileNameWithoutExtension(CSVfile)}") || Path.GetExtension(txtFile)!= ".txt")
                            {
                                // This if statement is designed to skip unmatched txtfiles and csvfiles 
                                // i.e. we skip files that refer to different instances
                                continue;
                            }
                            Console.WriteLine($".....Testing on {Path.GetFileNameWithoutExtension(txtFile)}.....");
                            // Here we find the corresponding .txt file, assuming that it has the same name.
                            string fileNameTXT = Path.Join(ObservationsDirectory,$"{Path.GetFileNameWithoutExtension(txtFile)}.txt");
                            double[,,,] observationsTXT = null;
                            try
                            {
                                observationsTXT= KDEdata.ConvertTXTtoMatrix(fileNameTXT);
                            }
                            catch{
                                Console.WriteLine($"FAIL: Cannot find {fileNameTXT}");
                                continue;
                            }
                            foreach (string kernel in kernels)
                            {
                                foreach (string bandwidth in bandwidths)
                                {
                                    config.kernelChoice = kernel;
                                    config.bandwidth = bandwidth;
                                    totalWriter.WriteLine($"Config: {config.kernelChoice} with bandwidth {config.bandwidth}"); 
                                    config.txtFileName = Path.GetFileNameWithoutExtension(txtFile);

                                    double totalValue = 0.0;
                                    Console.WriteLine($"Starting local search with {config.kernelChoice} and {config.bandwidth} on {Path.GetFileNameWithoutExtension(txtFile)}");
                                    for (int i = 0; i < numRepeats; i++)
                                    {
                                        // the 'failed' boolean needs to be looked at
                                        (bool failed, List<Route> ilpSol, double ilpVal, double ilpTime, double lsTime, double lsVal, string solutionJSON) = await solver.SolveVRPLTTInstanceAsyncKDE(CSVfile, observationsTXT, numLoadLevels: opts.NumLoadLevels, numIterations: opts.Iterations, timelimit: opts.TimeLimitLS * 1000, bikeMinMass: opts.BikeMinWeight, bikeMaxMass: opts.BikeMaxWeight, inputPower: opts.BikePower, numStarts: opts.NumStarts, numThreads: opts.NumThreads, config: config,ilpTimelimit:opts.TimeLimitILP, repetition: i);//solver.SolveSolomonInstanceAsync(file, numThreads: numThreads, numIterations: numIterations, timeLimit: 30 * 1000);
                                        using (var writer = new StreamWriter(Path.Join(solDir, Path.GetFileNameWithoutExtension(txtFile) + $"_{config.WaitingPenalty}_{config.LatenessPenalty}_{i}.txt")))
                                        {
                                            if (failed)
                                                writer.Write("FAIL: did not meet check..");
                                            writer.WriteLine($"Score: {ilpVal}, ilpTime: {ilpTime}, lsTime: {lsTime}, Early arrival allowed {config.AllowDeterministicEarlyArrival}");
                                            foreach (var route in ilpSol)
                                            {
                                                writer.WriteLine($"{route}");
                                            }
                                            writer.WriteLine(solutionJSON);
                                        }
                                        if (failed)
                                            totalWriter.Write("FAIL did not meet check..");

                                        totalValue += ilpVal;

                                        totalWriter.WriteLine($"Instance: { Path.GetFileNameWithoutExtension(txtFile)+ $"_{kernel}_{bandwidth}"},Early arrival allowed {config.AllowDeterministicEarlyArrival}, Score: {Math.Round(ilpVal, 3)}, Vehicles: {ilpSol.Count}, ilpTime: {Math.Round(ilpTime, 3)}, lsTime: {Math.Round(lsTime, 3)}, lsVal: {Math.Round(lsVal, 3)}, ilpImp: {Math.Round((lsVal - ilpVal) / lsVal * 100, 3)}%");
                                        csvWriter.WriteLine($"{ Path.GetFileNameWithoutExtension(txtFile)};{config.AllowDeterministicEarlyArrival};{Math.Round(ilpVal, 3)};{ilpSol.Count};{Math.Round(ilpTime, 3)};{Math.Round(lsTime, 3)};{Math.Round(lsVal, 3)};{Math.Round((lsVal - ilpVal) / lsVal * 100, 3)}");
                                        totalWriter.Flush();
                                        csvWriter.Flush();
                                    }
                                }
                            }
                        }
                    }
                }  
            }
        }
        public static async Task KDETestingWithWeights(string CSVdirectory, string ObservationsDirectory, string solDir, int numRepeats, Options opts,LocalSearchConfiguration? configInput = null)
        {
            // This function is designed for trying different settings for the KDE method
            // This is only used to find the ideal settings
            Console.WriteLine("......Testing on all vrpltt instances......");
            if (!Directory.Exists(solDir))
                Directory.CreateDirectory(solDir);

            using (var totalWriter = new StreamWriter(Path.Join(solDir, $"allSolutionsVRPLTT{opts.TestName}.txt")))
            {
                totalWriter.WriteLine("Command line arguments:");
                totalWriter.WriteLine(opts.ToString());
                LocalSearchConfiguration config;
                
                if(configInput == null)
                 config = LocalSearchConfigs.VRPLTTFinal;
                else
                    config = (LocalSearchConfiguration)configInput;
                
                double[] WaitingPenaltiesList = [1.1, 2, 1];
                double[] LatenessPenaltiesList = [20, 50, 1, 100];

                totalWriter.WriteLine("Config:");
                totalWriter.WriteLine(config.WaitingPenalty);
                totalWriter.WriteLine(config.LatenessPenalty);
                totalWriter.Flush();
                using (var csvWriter = new StreamWriter(Path.Join(solDir, $"allSolutionsVRPLTT{opts.TestName}.csv")))
                {
                    csvWriter.WriteLine("SEP=;");
                    csvWriter.WriteLine("Instance;AllowWaiting;Score;N;ILP time;LS time;LS score;ILP imp(%)");

                    // This is the loop over the csv files that represent the different instances.
                    foreach (var CSVfile in Directory.GetFiles(CSVdirectory))
                    {
                        if (Path.GetExtension(CSVfile) != ".csv")
                        {
                            Console.WriteLine($"{ Path.GetFileNameWithoutExtension(CSVfile)} is not an instance.");
                            continue;
                        }
                        foreach (var txtFile in Directory.GetFiles(ObservationsDirectory))
                        {
                            if (!Path.GetFileNameWithoutExtension(txtFile).Contains($"{Path.GetFileNameWithoutExtension(CSVfile)}") || Path.GetExtension(txtFile)!= ".txt")
                            {
                                // This if statement is designed to skip unmatched txtfiles and csvfiles 
                                // i.e. we skip files that refer to different instances
                                continue;
                            }
                            Console.WriteLine($".....Testing on {Path.GetFileNameWithoutExtension(txtFile)}.....");
                            // Here we find the corresponding .txt file, assuming that it has the same name.
                            string fileNameTXT = Path.Join(ObservationsDirectory,$"{Path.GetFileNameWithoutExtension(txtFile)}.txt");
                            double[,,,] observationsTXT = null;
                            try
                            {
                                observationsTXT= KDEdata.ConvertTXTtoMatrix(fileNameTXT);
                            }
                            catch{
                                Console.WriteLine($"FAIL: Cannot find {fileNameTXT}");
                                continue;
                            }
                            foreach (double WaitingPenaltyLocal in WaitingPenaltiesList)
                            {
                                foreach (double LatenessPenaltyLocal in LatenessPenaltiesList)
                                {
                                    config.WaitingPenalty = WaitingPenaltyLocal;
                                    config.LatenessPenalty = LatenessPenaltyLocal;

                                    double totalValue = 0.0;
                                    Console.WriteLine($"Starting local search with txtfile {txtFile}");
                                    for (int i = 0; i < numRepeats; i++)
                                    {
                                        // the 'failed' boolean needs to be looked at
                                        (bool failed, List<Route> ilpSol, double ilpVal, double ilpTime, double lsTime, double lsVal, string solutionJSON) = await solver.SolveVRPLTTInstanceAsyncKDE(CSVfile, observationsTXT, numLoadLevels: opts.NumLoadLevels, numIterations: opts.Iterations, timelimit: opts.TimeLimitLS * 1000, bikeMinMass: opts.BikeMinWeight, bikeMaxMass: opts.BikeMaxWeight, inputPower: opts.BikePower, numStarts: opts.NumStarts, numThreads: opts.NumThreads, config: config,ilpTimelimit:opts.TimeLimitILP);//solver.SolveSolomonInstanceAsync(file, numThreads: numThreads, numIterations: numIterations, timeLimit: 30 * 1000);
                                        using (var writer = new StreamWriter(Path.Join(solDir, Path.GetFileNameWithoutExtension(txtFile) + $"_{config.WaitingPenalty}_{config.LatenessPenalty}_{i}.txt")))
                                        {
                                            if (failed)
                                                writer.Write("FAIL: did not meet check..");
                                            writer.WriteLine($"Score: {ilpVal}, ilpTime: {ilpTime}, lsTime: {lsTime}, Early arrival allowed {config.AllowDeterministicEarlyArrival}");
                                            foreach (var route in ilpSol)
                                            {
                                                writer.WriteLine($"{route}");
                                            }
                                            writer.WriteLine(solutionJSON);
                                        }
                                        if (failed)
                                            totalWriter.Write("FAIL did not meet check..");

                                        totalValue += ilpVal;

                                        totalWriter.WriteLine($"Instance: { Path.GetFileNameWithoutExtension(txtFile)+ $"_{config.WaitingPenalty}_{config.LatenessPenalty}"},Early arrival allowed {config.AllowDeterministicEarlyArrival}, Score: {Math.Round(ilpVal, 3)}, Vehicles: {ilpSol.Count}, ilpTime: {Math.Round(ilpTime, 3)}, lsTime: {Math.Round(lsTime, 3)}, lsVal: {Math.Round(lsVal, 3)}, ilpImp: {Math.Round((lsVal - ilpVal) / lsVal * 100, 3)}%");
                                        csvWriter.WriteLine($"{ Path.GetFileNameWithoutExtension(txtFile)};{config.AllowDeterministicEarlyArrival};{Math.Round(ilpVal, 3)};{ilpSol.Count};{Math.Round(ilpTime, 3)};{Math.Round(lsTime, 3)};{Math.Round(lsVal, 3)};{Math.Round((lsVal - ilpVal) / lsVal * 100, 3)}");
                                        totalWriter.Flush();
                                        csvWriter.Flush();
                                    }
                                }
                            }
                        }
                    }
                }  
            }
        }
        public static async Task TestAllOperatorConfigurations(string dir,string solDir, Options opts)
        {
            var config = LocalSearchConfigs.VRPLTTOriginal;

            //ORIGINAL
            var newSolDir = Path.Join(solDir, "Originial");
            var newOpts = opts;
            newOpts.TestName = "Original";
            await RunVRPLTTTests(dir, newSolDir, opts.NumRepeats, opts, config);


            //NO REPEAT
            newSolDir = Path.Join(solDir, "NoRepeat");
            newOpts = opts;
            newOpts.TestName = "NoRepeat";
            config = LocalSearchConfigs.VRPLTTOriginal;
            config.Operators = new List<(Operator, double, string, int)>()
            {
                ((x, y, z, w, v) =>Operators.AddRandomRemovedCustomer(x, y, z, w, v), 1, "add", 1), //repeated 1 time
                (Operators.RemoveRandomCustomer, 1, "remove", 1), //repeated 1 time
                ((routes, viableRoutes, random, removed, temp) => Operators.MoveRandomCustomerToRandomCustomer(routes, viableRoutes, random), 1, "move", 1),//repeated 1 time
               ((x, y, z, w, v) => Operators.GreedilyMoveRandomCustomer(x, y, z), 0.1, "move_to_best",1), //repeated 1 time
                ((x, y, z, w, v) => Operators.MoveRandomCustomerToRandomRoute(x, y, z), 1, "move_to_random_route", 1), //repeated 4 times
                ((x, y, z, w, v) => Operators.SwapRandomCustomers(x, y, z), 1, "swap", 1), //repeated 4 times
                ((x, y, z, w, v) => Operators.SwapInsideRoute(x, y, z), 1, "swap_inside_route", 1), //repeated 4 times
                ((x, y, z, w, v) => Operators.ReverseOperator(x, y, z), 1, "reverse",1), //repeated 1 time
                ((x, y, z, w, v) => Operators.ScrambleSubRoute(x, y, z), 1, "scramble",1), //Repeated 1 time
                ((x, y, z, w, v) => Operators.SwapRandomTails(x, y, z), 1, "swap_tails",1), //Repeated 1 time
            };
            await RunVRPLTTTests(dir, newSolDir, opts.NumRepeats, opts, config);


            //ONLY SIMPLE REPEAT
            newSolDir = Path.Join(solDir, "OnlySimpleRepeat");
            newOpts = opts;
            newOpts.TestName = "OnlySimpleRepeat";
            config = LocalSearchConfigs.VRPLTTOriginal;
            config.Operators = new List<(Operator, double, string, int)>()
            {
                ((x, y, z, w, v) =>Operators.AddRandomRemovedCustomer(x, y, z, w, v), 1, "add", 4), //repeated 1 time
                (Operators.RemoveRandomCustomer, 1, "remove", 4), //repeated 1 time
                ((routes, viableRoutes, random, removed, temp) => Operators.MoveRandomCustomerToRandomCustomer(routes, viableRoutes, random), 1, "move", 4),//repeated 1 time
               ((x, y, z, w, v) => Operators.GreedilyMoveRandomCustomer(x, y, z), 0.1, "move_to_best",1), //repeated 1 time
                ((x, y, z, w, v) => Operators.MoveRandomCustomerToRandomRoute(x, y, z), 1, "move_to_random_route", 1), //repeated 4 times
                ((x, y, z, w, v) => Operators.SwapRandomCustomers(x, y, z), 1, "swap", 4), //repeated 4 times
                ((x, y, z, w, v) => Operators.SwapInsideRoute(x, y, z), 1, "swap_inside_route", 4), //repeated 4 times
                ((x, y, z, w, v) => Operators.ReverseOperator(x, y, z), 1, "reverse",1), //repeated 1 time
                ((x, y, z, w, v) => Operators.ScrambleSubRoute(x, y, z), 1, "scramble",1), //Repeated 1 time
                ((x, y, z, w, v) => Operators.SwapRandomTails(x, y, z), 1, "swap_tails",1), //Repeated 1 time
            };
            await RunVRPLTTTests(dir, newSolDir, opts.NumRepeats, opts, config);


            //ONLY SIMPLE EXCEPT ADD REMVOE
            newSolDir = Path.Join(solDir, "OnlySimpleRepeatExceptAddRemove");
            newOpts = opts;
            newOpts.TestName = "OnlySimpleRepeatExceptAddRemove";
            config = LocalSearchConfigs.VRPLTTOriginal;
            config.Operators = new List<(Operator, double, string, int)>()
            {
                ((x, y, z, w, v) =>Operators.AddRandomRemovedCustomer(x, y, z, w, v), 1, "add", 1), //repeated 1 time
                (Operators.RemoveRandomCustomer, 1, "remove", 1), //repeated 1 time
                ((routes, viableRoutes, random, removed, temp) => Operators.MoveRandomCustomerToRandomCustomer(routes, viableRoutes, random), 1, "move", 4),//repeated 1 time
               ((x, y, z, w, v) => Operators.GreedilyMoveRandomCustomer(x, y, z), 0.1, "move_to_best",1), //repeated 1 time
                ((x, y, z, w, v) => Operators.MoveRandomCustomerToRandomRoute(x, y, z), 1, "move_to_random_route", 1), //repeated 4 times
                ((x, y, z, w, v) => Operators.SwapRandomCustomers(x, y, z), 1, "swap", 4), //repeated 4 times
                ((x, y, z, w, v) => Operators.SwapInsideRoute(x, y, z), 1, "swap_inside_route", 4), //repeated 4 times
                ((x, y, z, w, v) => Operators.ReverseOperator(x, y, z), 1, "reverse",1), //repeated 1 time
                ((x, y, z, w, v) => Operators.ScrambleSubRoute(x, y, z), 1, "scramble",1), //Repeated 1 time
                ((x, y, z, w, v) => Operators.SwapRandomTails(x, y, z), 1, "swap_tails",1), //Repeated 1 time
            };
            await RunVRPLTTTests(dir, newSolDir, opts.NumRepeats, opts, config);

            // ORIGINAL WITHOUT ADD REMOVE
            newSolDir = Path.Join(solDir, "OriginalWithoutAddRemove");
            newOpts = opts;
            newOpts.TestName = "OriginalWithoutAddRemove";
            config = LocalSearchConfigs.VRPLTTOriginal;
            config.Operators = new List<(Operator, double, string, int)>()
            {
                ((routes, viableRoutes, random, removed, temp) => Operators.MoveRandomCustomerToRandomCustomer(routes, viableRoutes, random), 1, "move", 1),//repeated 1 time
               ((x, y, z, w, v) => Operators.GreedilyMoveRandomCustomer(x, y, z), 0.1, "move_to_best",1), //repeated 1 time
                ((x, y, z, w, v) => Operators.MoveRandomCustomerToRandomRoute(x, y, z), 1, "move_to_random_route", 4), //repeated 4 times
                ((x, y, z, w, v) => Operators.SwapRandomCustomers(x, y, z), 1, "swap", 4), //repeated 4 times
                ((x, y, z, w, v) => Operators.SwapInsideRoute(x, y, z), 1, "swap_inside_route", 4), //repeated 4 times
                ((x, y, z, w, v) => Operators.ReverseOperator(x, y, z), 1, "reverse",1), //repeated 1 time
                ((x, y, z, w, v) => Operators.ScrambleSubRoute(x, y, z), 1, "scramble",1), //Repeated 1 time
                ((x, y, z, w, v) => Operators.SwapRandomTails(x, y, z), 1, "swap_tails",1), //Repeated 1 time
            };
            await RunVRPLTTTests(dir, newSolDir, opts.NumRepeats, opts, config);

            // ORIGINAL GREEDY ROUTE REPEATED TWICE
            newSolDir = Path.Join(solDir, "OriginalGreedyRoute2");
            newOpts = opts;
            newOpts.TestName = "OriginalGreedyRoute2";
            config = LocalSearchConfigs.VRPLTTOriginal;
            config.Operators = new List<(Operator, double, string, int)>()
            {
                ((x, y, z, w, v) =>Operators.AddRandomRemovedCustomer(x, y, z, w, v), 1, "add", 1), //repeated 1 time
                (Operators.RemoveRandomCustomer, 1, "remove", 1), //repeated 1 time
                ((routes, viableRoutes, random, removed, temp) => Operators.MoveRandomCustomerToRandomCustomer(routes, viableRoutes, random), 1, "move", 1),//repeated 1 time
               ((x, y, z, w, v) => Operators.GreedilyMoveRandomCustomer(x, y, z), 0.1, "move_to_best",1), //repeated 1 time
                ((x, y, z, w, v) => Operators.MoveRandomCustomerToRandomRoute(x, y, z), 1, "move_to_random_route", 2), //repeated 4 times
                ((x, y, z, w, v) => Operators.SwapRandomCustomers(x, y, z), 1, "swap", 4), //repeated 4 times
                ((x, y, z, w, v) => Operators.SwapInsideRoute(x, y, z), 1, "swap_inside_route", 4), //repeated 4 times
                ((x, y, z, w, v) => Operators.ReverseOperator(x, y, z), 1, "reverse",1), //repeated 1 time
                ((x, y, z, w, v) => Operators.ScrambleSubRoute(x, y, z), 1, "scramble",1), //Repeated 1 time
                ((x, y, z, w, v) => Operators.SwapRandomTails(x, y, z), 1, "swap_tails",1), //Repeated 1 time
            };
            await RunVRPLTTTests(dir, newSolDir, opts.NumRepeats, opts, config);


            // ORIGINAL, BUT FASTER
            newSolDir = Path.Join(solDir, "FasterOriginal");
            newOpts = opts;
            newOpts.TestName = "FasterOriginal";
            config = LocalSearchConfigs.VRPLTTOriginal;
            config.IterationsPerAlphaChange = 5000;
            config.NumIterationsOfNoChangeBeforeRestarting = 200000;
            await RunVRPLTTTests(dir, newSolDir, opts.NumRepeats, opts, config);

            // ORIGINAL, HIGHER ADD REMOVE PENALTY
            newSolDir = Path.Join(solDir, "HigherAddRemovePenalty");
            newOpts = opts;
            newOpts.TestName = "HigherAddRemovePenalty";
            config = LocalSearchConfigs.VRPLTTOriginal;
            config.BaseRemovedCustomerPenalty = 2;
            await RunVRPLTTTests(dir, newSolDir, opts.NumRepeats, opts, config);

            // ORIGINAL, LINEAR HIGHER ADD REMOVE PENALTY
            newSolDir = Path.Join(solDir, "LinearHigherAddRemovePenalty");
            newOpts = opts;
            newOpts.TestName = "LinearHigherAddRemovePenalty";
            config = LocalSearchConfigs.VRPLTTOriginal;
            config.BaseRemovedCustomerPenalty = 4;
            config.BaseRemovedCustomerPenaltyPow = 1;
            await RunVRPLTTTests(dir, newSolDir, opts.NumRepeats, opts, config);


            // ORIGINAL, LINEAR HIGHER ADD REMOVE PENALTY LINEAR temperature Pow
            newSolDir = Path.Join(solDir, "LinearHigherAddRemovePenaltyLinearTempPow");
            newOpts = opts;
            newOpts.TestName = "LinearHigherAddRemovePenaltyLinearTempPow";
            config = LocalSearchConfigs.VRPLTTOriginal;
            config.BaseRemovedCustomerPenalty = 8;
            config.BaseRemovedCustomerPenaltyPow = 1;
            config.RemovedCustomerTemperaturePow = 1;
            await RunVRPLTTTests(dir, newSolDir, opts.NumRepeats, opts, config);

        }
        public static async Task RunVRPLTTWindtests(string dir, string solDir, int numRepeats, Options opts)
        {
            Console.WriteLine("Testing with wind on all vrpltt instances");

            if (!Directory.Exists(solDir))
                Directory.CreateDirectory(solDir);

            using (var totalWriter = new StreamWriter(Path.Join(solDir, $"allSolutionsVRPLTTWind{opts.TestName}.txt")))
            {
                totalWriter.WriteLine(opts.ToString());
                totalWriter.Flush();
                using (var csvWriter = new StreamWriter(Path.Join(solDir, $"allSolutionsVRPLTTWind{opts.TestName}.csv")))
                {
                    csvWriter.WriteLine("SEP=;");
                    csvWriter.WriteLine("Instance;Windspeed;WindDirection;Score;N;ILP time;LS time;LS score;ILP imp;TimeCycledAgainstWind;TimeWithWindIncluded;NumberOfValidRoutes");

                    LocalSearchConfiguration config = LocalSearchConfigs.VRPLTTWithWind;

                    double testWindSpeed = config.WindSpeed;

                    for (int test = 0; test < 1; test++)
                    {
                        foreach (var file in Directory.GetFiles(dir))
                        {
                            // These lines are needed for MAC OS users.
                            if (file.EndsWith(".DS_Store"))
                                continue;
                            for (int repeat = 0; repeat < numRepeats; repeat++)
                            {
                                string fileNameNow = Path.Join(solDir, Path.GetFileNameWithoutExtension(file) + $"_Ws{config.WindSpeed.ToString().Replace(',','.')}_Wd0_Sam_0_{test}_{repeat}.txt");
                                if (File.Exists(fileNameNow))
                                {
                                    Console.WriteLine($"File already exists for {fileNameNow}");
                                    continue;
                                }
                                //Running the test
                                (bool failed, List<Route> ilpSol, double ilpVal, double ilpTime, double lsTime, double lsVal, string solutionJSON) = await solver.SolveVRPLTTInstanceAsync(file, numLoadLevels: opts.NumLoadLevels, numIterations: opts.Iterations, timelimit: opts.TimeLimitLS * 1000, bikeMinMass: opts.BikeMinWeight, bikeMaxMass: opts.BikeMaxWeight, inputPower: opts.BikePower, numStarts: opts.NumStarts, numThreads: opts.NumThreads, config: config,ilpTimelimit:opts.TimeLimitILP);//solver.SolveSolomonInstanceAsync(file, numThreads: numThreads, numIterations: numIterations, timeLimit: 30 * 1000);

                                var windResult = VRPLTT.CalculateWindCyclingTime(file, opts.BikeMinWeight, opts.BikeMaxWeight, opts.NumLoadLevels, opts.BikePower, config.WindDirection, ilpSol);
                                using (var writer = new StreamWriter(fileNameNow))
                                {
                                    if (failed)
                                        writer.Write("FAIL did not meet check");
                                    writer.WriteLine($"Score: {ilpVal}, ilpTime: {ilpTime}, lsTime: {lsTime}, Early arrival allowed {config.AllowDeterministicEarlyArrival}");
                                    foreach (var route in ilpSol)
                                    {
                                        writer.WriteLine($"{route}");
                                    }
                                    writer.WriteLine(solutionJSON);
                                    writer.WriteLine($"WindDir: ({config.WindDirection[0]},{config.WindDirection[1]})");
                                }
                                double cycleSpeedWithWind = ilpVal;
                                int valid = ilpSol.Count;
                                if (config.WindSpeed == 0)
                                {
                                    var tempConfig = config;
                                    tempConfig.WindSpeed = testWindSpeed;

                                    tempConfig.PenalizeLateArrival = false;
                                    tempConfig.PenalizeDeterministicEarlyArrival = false;
                                    //If the windspeed is 0 we want to check what the travel time would be if we plan without it
                                    (double[,] distances, List<Customer> customers) = VRPLTT.ParseVRPLTTInstance(file);

                                    Console.WriteLine("Calculating travel time matrix");
                                    LocalSearch ls = new LocalSearch(tempConfig,1);
                                    double total = 0;
                                   ;
                                    foreach( var r in ilpSol)
                                    {
                                        var rs = new RouteStore(r.CreateIdList(), 0);
                                        //var newRoute = new Route(customers, rs, customers[0], opts.BikeMaxWeight - opts.BikeMinWeight, ls);
                                        //total += newRoute.CalcObjectiveKDE();
                                        //if(newRoute.CheckRouteValidity())
                                           // valid--;
                                    }
                                    cycleSpeedWithWind = total;
                                }

                                Console.WriteLine("------------------------------------------------------------------" + cycleSpeedWithWind + "  " + (double)valid/ilpSol.Count);

                                csvWriter.WriteLine($"{Path.GetFileNameWithoutExtension(file)};{config.WindSpeed};({config.WindDirection[0]},{config.WindDirection[1]});{ilpVal};{ilpTime};{lsTime};{lsVal};{(ilpVal - lsVal) / lsVal * 100};{windResult};{cycleSpeedWithWind};{(double)valid / ilpSol.Count}");
                                csvWriter.Flush();
                                
                                if (failed)
                                    totalWriter.Write("FAIL did not meet check");
                                totalWriter.Flush();
                            }
                        }
                    }
                }
            }
        }
        public static async Task BenchmarkLocalSearchSpeed(string dir, string solDir,int numRepeats,Options opts)
        {
            Console.WriteLine("Benchmarking localsearch");
            if (!Directory.Exists(solDir))
                Directory.CreateDirectory(solDir);
            var config = LocalSearchConfigs.VRPLTTFinal;
            var solver = new Solver();
            using (var totalWriter = new StreamWriter(Path.Join(solDir, "allSolutionsVRPLTT.txt")))
            {
                totalWriter.WriteLine(opts.ToString());
                using (var csvWriter = new StreamWriter(Path.Join(solDir, "allSolutionsVRPLTT.csv")))
                {
                    csvWriter.WriteLine("SEP=;");
                    csvWriter.WriteLine("Instance;Score;N;LS time");

                    for (int test =0; test< opts.NumRepeats;test++)
                    foreach (var file in Directory.GetFiles(dir))
                    {
                        Console.WriteLine($"Testing on { Path.GetFileNameWithoutExtension(file)}");

                        (double[,] distances, List<Customer> customers) = VRPLTT.ParseVRPLTTInstance(file);
                        Console.WriteLine("Calculating travel time matrix");
                        (double[,,] matrix, Gamma[,,] distributionMatrix, IContinuousDistribution[,,] approximationMatrix, _) = VRPLTT.CalculateLoadDependentTimeMatrix(customers, distances, opts.BikeMinWeight, opts.BikeMaxWeight, opts.NumLoadLevels, opts.BikePower, ((LocalSearchConfiguration)config).WindSpeed, ((LocalSearchConfiguration)config).WindDirection, config.DefaultDistribution is Normal);

                        (var allColumns, var bestSolution, var LSTime, var LSSCore) = await solver.LocalSearchInstancAsync("", customers.Count - 1, opts.BikeMaxWeight - opts.BikeMinWeight, customers, matrix, distributionMatrix, approximationMatrix, opts.NumThreads, opts.NumStarts, opts.Iterations, opts.TimeLimitLS * 1000, config);

                        using (var writer = new StreamWriter(Path.Join(solDir, Path.GetFileNameWithoutExtension(file) + $"_{test}.txt")))
                        {
                            writer.WriteLine($"Score: {LSSCore}, lsTime: {LSTime}, Early arrival allowed {config.AllowDeterministicEarlyArrival}");
                            foreach (var route in bestSolution)
                            {
                                if(route.route.Count > 2)
                                    writer.WriteLine($"{route}");
                            }
                        }

                        totalWriter.WriteLine($"Instance: { Path.GetFileNameWithoutExtension(file)},Early arrival allowed {config.AllowDeterministicEarlyArrival}, Score: {Math.Round(LSSCore, 3)}, Vehicles: {bestSolution.Where(x=>x.route.Count > 2).Count()}, , lsTime: {Math.Round(LSTime, 3)}");
                        csvWriter.WriteLine($"{ Path.GetFileNameWithoutExtension(file)};{Math.Round(LSSCore, 3)};{bestSolution.Where(x => x.route.Count > 2).Count()};{Math.Round(LSTime, 3)}");

                            totalWriter.Flush();
                            csvWriter.Flush();
                        }
                }
            }
        }
        public static async Task RunLastVRPSLTTTests(string dir, string solDir, int numRepeats,Options opts)
        {
            LocalSearchConfiguration config;
            config = LocalSearchConfigs.VRPSLTTWithoutWaitingNormalInBetweenMaximizaton;
            var newSolDir = Path.Join(solDir, "WithoutWaitingNormalInBetweenMaximizaton");
            opts.TestName = "WithoutWaitingNormalInBetweenMaximizaton";
            await RunVRPSLTTTest(dir, newSolDir, numRepeats, opts, config);

            config = LocalSearchConfigs.VRPSLTTWithWaitingNormalInBetweenMaximizaton;
            newSolDir = Path.Join(solDir, "WithWaitingNormalInBetweenMaximizaton");
            opts.TestName = "WithWaitingNormalInBetweenMaximizaton";
            await RunVRPSLTTTest(dir, newSolDir, numRepeats, opts, config);
        }
        public static async Task RunVRPSLTTTests(string dir, string solDir, int numRepeats, Options opts)
        {
            LocalSearchConfiguration config;

            //WAITING NOT ALLOWED

            //True distribution without waiting
            config = LocalSearchConfigs.VRPSLTTWithoutWaiting;
            var newSolDir = Path.Join(solDir, "WithoutWaiting");
            opts.TestName = "WithoutWaiting";
            await RunVRPSLTTTest(dir,newSolDir,numRepeats,opts,config);

            //Normal approximation without waiting
            config = LocalSearchConfigs.VRPSLTTWithoutWaitingNormal;
            newSolDir = Path.Join(solDir, "WithoutWaitingNormal");
            opts.TestName = "WithoutWaitingNormal";
            await RunVRPSLTTTest(dir, newSolDir, numRepeats, opts, config);

            //Cut normal approximation without waiting
            config = LocalSearchConfigs.VRPSLTTWithoutWaitingCutNormal;
            newSolDir = Path.Join(solDir, "WithoutWaitingCutNormal");
            opts.TestName = "WithoutWaitingCutNormal";
            await RunVRPSLTTTest(dir, newSolDir, numRepeats, opts, config);

            //Just mean without waiting
            config = LocalSearchConfigs.VRPSLTTWithoutWaitingJustMean;
            newSolDir = Path.Join(solDir, "WithoutWaitingJustMean");
            opts.TestName = "WithoutWaitingJustMean";
            await RunVRPSLTTTest(dir, newSolDir, numRepeats, opts, config);

            //WAITING ALLOWED

            //Normal approximation with waiting
            config = LocalSearchConfigs.VRPSLTTWithWaitingNormal;
            newSolDir = Path.Join(solDir, "WithWaitingNormal");
            opts.TestName = "WithWaitingNormal";
            await RunVRPSLTTTest(dir, newSolDir, numRepeats, opts, config);

            //Cut normal approximation with waiting
            config = LocalSearchConfigs.VRPSLTTWithWaitingCutNormal;
            newSolDir = Path.Join(solDir, "WithWaitingCutNormal");
            opts.TestName = "WithWaitingCutNormal";
            await RunVRPSLTTTest(dir, newSolDir, numRepeats, opts, config);

            //Just mean with waiting
            config = LocalSearchConfigs.VRPSLTTWithWaitingJustMean;
            newSolDir = Path.Join(solDir, "WithWaitingJustMean");
            opts.TestName = "WithWaitingJustMean";
            await RunVRPSLTTTest(dir, newSolDir, numRepeats, opts, config);

            //Dumb gamma planning (planning with deterministic with waiting with gamma distribution on top)
            config = LocalSearchConfigs.VRPSLTTWithWaitingStupidGamma;
            newSolDir = Path.Join(solDir, "WithWaitingStupidGamma");
            opts.TestName = "WithWaitingStupidGamma";
            await RunVRPSLTTTest(dir, newSolDir, numRepeats, opts, config);

        }
        public static async Task RunVRPSLTTTest(string dir, string solDir, int numRepeats, Options opts,LocalSearchConfiguration config)
        {
            Console.WriteLine("Testing on all vrpltt instances");
            if (!Directory.Exists(solDir))
                Directory.CreateDirectory(solDir);

            using (var totalWriter = new StreamWriter(Path.Join(solDir, $"allSolutionsVRPSLTT{opts.TestName}.txt")))
            {
                totalWriter.WriteLine(opts.ToString());
                using (var csvWriter = new StreamWriter(Path.Join(solDir, $"allSolutionsVRPSLTT{opts.TestName}.csv")))
                {
                    csvWriter.WriteLine("SEP=;");
                    csvWriter.WriteLine("Instance;Uncertanty penalty;UseMean;Score;N;ILP time;LS time;LS score;ILP imp(%);avg simulation distance;avg sim OTP; avg sim worst OTP;worst sim OTP;worst sim route;worst sim cust index;avg est OTP; avg est worst OTP;worst est OTP;worst est route;worst est cust index");
                        foreach (var file in Directory.GetFiles(dir))
                        {
                            Console.WriteLine($"Testing on { Path.GetFileNameWithoutExtension(file)}");
                            double totalValue = 0.0;
                            bool newInstance = false;
                            if (Path.GetExtension(file) != ".csv")
                                continue;

                            for (int r = 0; r < numRepeats; r++)
                            {
                                (bool failed, List<Route> ilpSol, double ilpVal, double ilpTime, double lsTime, double lsVal, string solutionJSON) = await solver.SolveVRPLTTInstanceAsync(file, numLoadLevels: opts.NumLoadLevels, numIterations: opts.Iterations, timelimit: opts.TimeLimitLS * 1000, bikeMinMass: opts.BikeMinWeight, bikeMaxMass: opts.BikeMaxWeight, inputPower: opts.BikePower, numStarts: opts.NumStarts, numThreads: opts.NumThreads, config: config,ilpTimelimit:opts.TimeLimitILP);//solver.SolveSolomonInstanceAsync(file, numThreads: numThreads, numIterations: numIterations, timeLimit: 30 * 1000);

                                double totalDist = 0;
                                double totalOntimePercentage = 0;
                                double averageWorstOntimePercentage = 0;
                                double worstOntimePercentage = double.MaxValue;
                                Route? worstRoute = null;
                                int worstRouteCustomer = -1;

                                double estimatedTotalOntimePercentage = 0;
                                double estimatedAverageWorstOntimePercentage = 0;
                                double estimatedWorstOntimePercentage = double.MaxValue;
                                Route? estimatedWorstRoute = null;
                                int estimatedWorstRouteCustomer = -1;

                                foreach (var route in ilpSol)
                                {
                                    if (route.route.Count != 2)
                                    {
                                        double avg = 0;
                                        int total = 0;
                                        double worst = double.MaxValue;
                                        Customer? worstCust = null;
                                        int worstIndex = -1;
                                        for (int i = 0; i < route.route.Count-1; i++)
                                        {
                                            if (route.customerDistributions[i] != null)
                                            {
                                                total += 1;

                                                double difUp = route.route[i].TWEnd - route.arrival_times[i];
                                                double difDown = route.route[i].TWStart - route.arrival_times[i];

                                                if (difUp < 0)
                                                    difUp = 0;
                                                if (difDown < 0)
                                                    difDown = 0;

                                                double pEarly = route.customerDistributions[i].CumulativeDistribution(difDown);

                                                double pOnTime = (route.customerDistributions[i].CumulativeDistribution(difUp));

                                                if (Double.IsNaN(pOnTime))
                                                    pOnTime = 1;

                                                if (((LocalSearchConfiguration)config).AllowEarlyArrivalInSimulation)
                                                    pEarly = 0;

                                                double p = pOnTime - pEarly;

                                                avg += p;
                                                if (p < worst)
                                                {
                                                    worstCust = route.route[i];
                                                    worst = p;
                                                    worstIndex = i;
                                                }
                                            }
                                        }
                                        estimatedTotalOntimePercentage += avg/total;
                                        estimatedAverageWorstOntimePercentage += worst;
                                        if (worst < estimatedWorstOntimePercentage)
                                        {
                                            estimatedWorstOntimePercentage = worst;
                                            estimatedWorstRoute = route;
                                            estimatedWorstRouteCustomer = worstIndex;
                                        }
                                        Console.WriteLine($"On time performance: {avg/total} worst: {worst} at {worstCust} at {worstIndex}");

                                        var res = route.Simulate(1000000);
                                        totalDist += res.AverageTravelTime;
                                        totalOntimePercentage += res.OnTimePercentage;

                                        int minIndex = -1;
                                        double min = double.MaxValue;
                                        for (int j = 0; j < res.CustomerOnTimePercentage.Length; j++)
                                        {
                                            if (res.CustomerOnTimePercentage[j] < min)
                                            {
                                                min = res.CustomerOnTimePercentage[j];
                                                minIndex = j;
                                            }

                                        }
                                        averageWorstOntimePercentage += min;
                                        if (min < worstOntimePercentage)
                                        {
                                            worstOntimePercentage = min;
                                            worstRoute = route;
                                            worstRouteCustomer = minIndex;
                                        }
                                        Console.WriteLine($"Simmulated on time perfomance: {res.OnTimePercentage} worst: {min} at {minIndex}\n");
                                    }
                                }

                                using (var writer = new StreamWriter(Path.Join(solDir, Path.GetFileNameWithoutExtension(file) + $"_{r}.txt")))
                                {
                                    if (failed)
                                        writer.Write("FAIL did not meet check");
                                    writer.WriteLine($"Score: {ilpVal}, ilpTime: {ilpTime}, lsTime: {lsTime}, Early arrival allowed {config.AllowDeterministicEarlyArrival}");
                                    foreach (var route in ilpSol)
                                    {
                                        writer.WriteLine($"{route}");
                                    }
                                writer.WriteLine(solutionJSON);
                                }
                                if (failed)
                                    totalWriter.Write("FAIL did not meet check");

                                totalValue += ilpVal;

                                totalWriter.WriteLine($"Instance: { Path.GetFileNameWithoutExtension(file)},Early arrival allowed {config.AllowDeterministicEarlyArrival}, Score: {Math.Round(ilpVal, 3)}, Vehicles: {ilpSol.Count}, ilpTime: {Math.Round(ilpTime, 3)}, lsTime: {Math.Round(lsTime, 3)}, lsVal: {Math.Round(lsVal, 3)}, ilpImp: {Math.Round((lsVal - ilpVal) / lsVal * 100, 3)}%");
                                csvWriter.WriteLine($"{ Path.GetFileNameWithoutExtension(file)};{config.ExpectedEarlinessPenalty};{config.UseMeanOfDistributionForTravelTime};{Math.Round(ilpVal, 3)};{ilpSol.Count};{Math.Round(ilpTime, 3)};{Math.Round(lsTime, 3)};{Math.Round(lsVal, 3)};{Math.Round((lsVal - ilpVal) / lsVal * 100, 3)};{totalDist.ToString("0.000")};{totalOntimePercentage / ilpSol.Count};{averageWorstOntimePercentage / ilpSol.Count};{worstOntimePercentage};{worstRoute};{worstRouteCustomer};{estimatedTotalOntimePercentage / ilpSol.Count};{estimatedAverageWorstOntimePercentage / ilpSol.Count};{estimatedWorstOntimePercentage};{estimatedWorstRoute};{estimatedWorstRouteCustomer}");
                                totalWriter.Flush();
                                csvWriter.Flush();
                            }
                        } 
                }
            }
        }
        public static async Task SolveAllAsync(string dir, string solDir, List<String> skip, Options opts)
        {
            if (!Directory.Exists(solDir))
                Directory.CreateDirectory(solDir);

            var solver = new Solver();
            Stopwatch watch = new Stopwatch();
            watch.Start();
            using (var totalWriter = new StreamWriter(Path.Join(solDir, $"allSolutions{opts.TestName}.txt")))
            using (var csvWriter = new StreamWriter(Path.Join(solDir, $"allSolutions{opts.TestName}.csv")))
            {
                csvWriter.WriteLine("SEP=;");
                csvWriter.WriteLine("Instance;Score;N;ILP time;LS time;LS score;ILP imp(%)");

                string fileNameStart = "";
                double totalValue = 0.0;
                int total = 0;
                for(int test= 0; test < opts.NumRepeats;test++)
                    foreach (var file in Directory.GetFiles(dir).OrderBy(f => f))
                    {
                        Console.WriteLine(file);
                        bool newInstance = false;
                        if (skip.Contains(Path.GetFileName(file)))
                            continue;

                        if (!Path.GetFileName(file).StartsWith(fileNameStart))
                        {
                            totalWriter.WriteLine($"AVERAGE {fileNameStart}: {totalValue / total}");
                            totalValue = 0;
                            total = 0;
                            fileNameStart = Path.GetFileName(file).Substring(0, 2);
                        }

                        (bool failed, List<RouteStore> ilpSol, double ilpVal, double ilpTime, double lsTime, double lsVal,string solutionJSON) = await solver.SolveSolomonInstanceAsync(file, numThreads: opts.NumThreads,numStarts:opts.NumStarts, numIterations: opts.Iterations, timeLimit: opts.TimeLimitLS * 1000);

                        csvWriter.WriteLine($"{Path.GetFileNameWithoutExtension(file)};{ilpVal};{ilpSol.Count};{ilpTime};{lsTime};{lsVal};{(ilpVal - lsVal) / lsVal * 100}");
                    
                        using (var writer = new StreamWriter(Path.Join(solDir, Path.GetFileNameWithoutExtension(file) + "_" + test  + ".txt")))
                        {
                            if (failed)
                                writer.Write("FAIL did not meet check");
                            writer.WriteLine($"Score: {ilpVal}, ilpTime: {ilpTime}, lsTime: {lsTime}");
                            foreach (var route in ilpSol)
                            {
                                writer.WriteLine($"{route}");
                            }
                            writer.WriteLine(solutionJSON);
                        }
                        if (failed)
                            totalWriter.Write("FAIL did not meet check");

                        totalValue += ilpVal;
                        total++;

                        totalWriter.WriteLine($"Instance: {Path.GetFileName(file)}, Score: {Math.Round(ilpVal, 3)}, Vehicles: {ilpSol.Count}, ilpTime: {Math.Round(ilpTime, 3)}, lsTime: {Math.Round(lsTime, 3)}, lsVal: {Math.Round(lsVal, 3)}, ilpImp: {Math.Round((lsVal - ilpVal) / lsVal * 100, 3)}%");
                        totalWriter.Flush();
                        csvWriter.Flush();
                    }
            }
        }
    }
}