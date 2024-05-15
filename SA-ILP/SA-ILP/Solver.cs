using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Gurobi;
using MathNet.Numerics;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.Interpolation;

namespace SA_ILP
{

    //Used for storing the columns during search
    class RouteStore
    {
        public List<int> Route { get; private set; }
        public double Value { get; private set; }


        public RouteStore(List<int> route, double value)
        {
            this.Route = route;
            Value = value;
        }

        //Routestore equaly depends only on the customers
        public override bool Equals(object? obj)
        {
            if (typeof(RouteStore) != obj.GetType())
                return false;
            var y = (RouteStore)obj;
            if (this.Route.Count != y.Route.Count)
                return false;
            for (int i = 0; i < Route.Count; i++)
            {
                if (Route[i] != y.Route[i])
                    return false;
            }
            return true;
        }

        public override string ToString()
        {
            return String.Join(' ', Route);
        }

        //Programmed to efficiently support up to 1000 customers. If more customers are in the problem this might integer overflows and more hash matches.
        public override int GetHashCode()
        {
            int total = 0;
            for (int i = 0; i < Route.Count; i++)
            {
                total += Route[i] * (i + 1001);
            }
            return total;
        }
    }

    internal class Solver
    {
        Random random;
        public static double CalcTotalDistance(List<Route> routes, List<Customer> removed, LocalSearch ls)
        {
            //This expclitly does not use the Score property of a route to force recalculation. This way bugs might be caught
            return routes.Sum(x => x.CalcObjective()) + CalcRemovedPenalty(removed.Count, ls);
        }
        public static double CalcTotalDistanceKDE(List<Route> routes, List<Customer> removed, LocalSearch ls)
        {
            //This expclitly does not use the Score property of a route to force recalculation. This way bugs might be caught
            return routes.Sum(x => x.CalcObjectiveKDE()) + CalcRemovedPenalty(removed.Count, ls);
        }

        public static double CalcRemovedPenalty(int removedCount, LocalSearch ls)
        {
            return Math.Pow(removedCount, ls.Config.BaseRemovedCustomerPenaltyPow) * (ls.Config.BaseRemovedCustomerPenalty / Math.Pow(ls.Temperature, ls.Config.RemovedCustomerTemperaturePow));
        }

        //Calculates the Solomon distance matrix
        private double[,,] CalculateDistanceMatrix(List<Customer> customers)
        {
            int size = customers.Max(x => x.Id) + 1;
            double[,,] matrix = new double[size, size, 1];
            for (int i = 0; i < customers.Count; i++)
                for (int j = 0; j < customers.Count; j++)
                    matrix[i, j, 0] = CalculateDistanceObjective(customers[i], customers[j]);//Math.Sqrt(Math.Pow(customers[i].X - customers[j].X,2) + Math.Pow(customers[i].Y - customers[j].Y,2));
            return matrix;
        }

        public static double CalculateDistanceObjective(Customer cust1, Customer cust2)
        {
            return Math.Sqrt(Math.Pow(cust1.X - cust2.X, 2) + Math.Pow(cust1.Y - cust2.Y, 2));
        }

        public Solver()
        {
            random = new Random();
        }

        public void SolveSolomonInstance(string fileName, int numIterations = 3000000, int timeLimit = 30000)
        {
            (string name, int numV, double capV, List<Customer> customers) = SolomonParser.ParseInstance(fileName);
            List<Task<(HashSet<RouteStore>, List<Route>, double)>> tasks = new List<Task<(HashSet<RouteStore>, List<Route>, double)>>();
            var distanceMatrix = CalculateDistanceMatrix(customers);
            var ls = new LocalSearch(LocalSearchConfigs.VRPTW, random.Next());
            (var colums, var sol, var value) = ls.LocalSearchInstance(0, name, numV, capV, customers.ConvertAll(i => new Customer(i)), distanceMatrix, new Gamma[distanceMatrix.GetLength(0), distanceMatrix.GetLength(1), distanceMatrix.GetLength(2)], new Gamma[distanceMatrix.GetLength(0), distanceMatrix.GetLength(1), distanceMatrix.GetLength(2)], numInterations: numIterations, timeLimit: timeLimit);
            foreach (Route route in sol)
                    route.CheckRouteValidity();

        }
        public async Task<(bool failed, List<Route> ilpSol, double ilpVal, double ilpTime, double lsTime, double lsVal, string solutionJSON)> SolveVRPLTTInstanceAsync(string fileName, int numIterations = 3000000, double bikeMinMass = 140, double bikeMaxMass = 290, int numLoadLevels = 10, double inputPower = 350, int timelimit = 30000, int numThreads = 4, int numStarts = 4, LocalSearchConfiguration? config = null, double ilpTimelimit = 3600)
        {
            LocalSearchConfiguration actualConfig;
            if (config == null)
                actualConfig = LocalSearchConfigs.VRPLTTFinal;
            else
                actualConfig = (LocalSearchConfiguration)config;

            (double[,] distances, List<Customer> customers) = VRPLTT.ParseVRPLTTInstance(fileName);
            Stopwatch w = new Stopwatch();
            w.Start();
            (double[,,] matrix, Gamma[,,] distributionMatrix, IContinuousDistribution[,,] approximationMatrix) = (null,null, null);
            Console.WriteLine("Calculating travel time matrix");
            (matrix, distributionMatrix, approximationMatrix, _) = VRPLTT.CalculateLoadDependentTimeMatrix(customers, distances, bikeMinMass, bikeMaxMass, numLoadLevels, inputPower, ((LocalSearchConfiguration)actualConfig).WindSpeed, ((LocalSearchConfiguration)actualConfig).WindDirection, actualConfig.DefaultDistribution is Normal);
            Console.WriteLine($"Created distance matrix in {((double)w.ElapsedMilliseconds / 1000).ToString("0.00")}s");
            (List<RouteStore> ilpSol, double ilpVal, double ilpTime, double lsTime, double lsVal, string solutionJSON) = await SolveInstanceAsync("", customers.Count, bikeMaxMass - bikeMinMass, customers, matrix, distributionMatrix, approximationMatrix, numThreads, numStarts, numIterations, (LocalSearchConfiguration)config, timeLimit: timelimit, ilpTimelimit: ilpTimelimit);

            double totalWaitingTime = 0;
            int numViolations = 0;
            double totalStartTime = 0;
            double checkValue = 0;
            double checkValueKDE = 0;
            double numCustomers = 0;
            double totalRouteLength = 0;

            HashSet<int> visitedCustomers = new HashSet<int>();
            bool failed = false;
            List<Route> Solution = new List<Route>();
            using (var sw = new StreamWriter("out.txt"))
                foreach (RouteStore rs in ilpSol)
                {
                    var ls = new LocalSearch(actualConfig, random.Next());
                    Route r = new Route(customers, rs, customers[0], matrix, distributionMatrix, approximationMatrix, bikeMaxMass - bikeMinMass, ls);
                    Solution.Add(r);
                    sw.WriteLine($"{r},");
                    checkValue += r.CalcObjective();
                    //checkValueKDE += r.CalcObjectiveKDE();
                    totalStartTime += r.arrival_times[0];
                    totalRouteLength += r.arrival_times[r.arrival_times.Count-1] - r.route.Sum(x => x.ServiceTime);
                    numCustomers += r.route.Count - 2;
                    var load = r.used_capacity;
                    if (r.route.Count != 2)
                        Console.WriteLine(r);
                    if (r.CheckRouteValidity())
                    {
                        failed = true;
                    }
                    for (int i = 0; i < r.route.Count - 1; i++)
                    {
                        if (i != 0 && visitedCustomers.Contains(r.route[i].Id))
                            failed = true;
                        visitedCustomers.Add(r.route[i].Id);
                        (var dist, IContinuousDistribution distribution) = r.CustomerDist(r.route[i], r.route[i + 1], load, false);
                        load -= r.route[i + 1].Demand;
                        if (r.arrival_times[i] + dist + r.route[i].ServiceTime < r.route[i + 1].TWStart)
                        {
                            Console.WriteLine($"Waiting for customer {r.route[i + 1].Id}");
                            totalWaitingTime += r.route[i + 1].TWStart - (r.arrival_times[i] + dist + r.route[i].ServiceTime);
                            numViolations += 1;
                        }
                    }
                }

            if (visitedCustomers.Count != customers.Count)
                failed = true;
            Console.WriteLine($"TotalRouteLength {totalRouteLength}");
            Console.WriteLine($"Num Customers {numCustomers}");
            Console.WriteLine($"Recalculated score: {checkValue}");
            //Console.WriteLine($"Recalculated KDE score: {checkValueKDE}");
            Console.WriteLine($"Total start time {totalStartTime}");
            Console.WriteLine($"Total waiting time: {totalWaitingTime} over {numViolations} customers");

            var parent = new LocalSearch(actualConfig, random.Next());
            return (failed, ilpSol.ConvertAll(x => new Route(customers, x, customers[0], matrix, distributionMatrix, approximationMatrix, bikeMaxMass - bikeMinMass, parent)), ilpVal, ilpTime, lsTime, lsVal, solutionJSON);
        }
        public async Task<(bool failed, List<Route> ilpSol, double ilpVal, double ilpTime, double lsTime, double lsVal, string solutionJSON)> SolveVRPLTTInstanceAsyncKDE(string CSVfileName, double[,,,] observations, int numIterations = 3000000, double bikeMinMass = 140, double bikeMaxMass = 290, int numLoadLevels = 10, double inputPower = 350, int timelimit = 30000, int numThreads = 4, int numStarts = 4, LocalSearchConfiguration? config = null, double ilpTimelimit = 3600, int repetition = 0)
        {
            LocalSearchConfiguration actualConfig;
            if (config == null)
                actualConfig = LocalSearchConfigs.VRPLTTFinal;
            else
                actualConfig = (LocalSearchConfiguration)config;

            (double[,] distances, List<Customer> customers) = VRPLTT.ParseVRPLTTInstance(CSVfileName);
            Stopwatch w = new Stopwatch();
            w.Start();
            (List<RouteStore> ilpSol, double ilpVal, double ilpTime, double lsTime, double lsVal, string solutionJSON) =  await SolveInstanceAsyncKDE("", customers.Count, bikeMaxMass - bikeMinMass, customers, observations, numThreads, numStarts, numIterations, (LocalSearchConfiguration)config, timeLimit: timelimit, ilpTimelimit: ilpTimelimit, repetition:repetition);

            double totalWaitingTime = 0;
            double totalTravelTime = 0;
            double totalLatePercentage = 0;
            double numViolations = 0;
            double checkValueKDE = 0;
            double checkValue2 = 0;
            int numCustomers = 0;
            //double totalRouteLength = 0;

            HashSet<int> visitedCustomers = new HashSet<int>();
            bool failed = false;
            List<Route> Solution = new List<Route>();
            using (var sw = new StreamWriter("out.txt"))
                foreach (RouteStore rs in ilpSol)
                {
                    var ls = new LocalSearch(actualConfig, random.Next());
                    Route r = new Route(customers, rs, customers[0], bikeMaxMass - bikeMinMass, observations, ls);
                    Solution.Add(r);
                    sw.WriteLine($"{r},");
                    
                    checkValueKDE += r.CalcObjectiveKDE();
                    //Console.WriteLine($"With total schedule score {Math.Round(checkValueKDE,2)}");
                    numCustomers += r.route.Count - 2;
                    if (r.route.Count != 2)
                        Console.WriteLine(r);
                   
                    if (r.CheckRouteValidityKDE())
                        {failed = true;}
                    else
                    {
                        for (int j = 0; j<observations.GetLength(3); j++)
                        {
                        double clock = 0;
                        double load = r.used_capacity;
                        
                        for (int i = 0; i < r.route.Count - 1; i++)
                            {
                                if (i != 0 && visitedCustomers.Contains(r.route[i].Id))
                                    failed = true;
                                visitedCustomers.Add(r.route[i].Id);
                                int ll = KDEdata.calculateKDELoadLevel(load);
                                double travelTime = 0;
                                try
                                {travelTime = observations[r.route[i].Id, r.route[i+1].Id, ll, j];}
                                catch {Console.WriteLine($"FAIL: cannot find {r.route[i].Id}, {r.route[i+1].Id}, {ll}, {j}");}

                                totalTravelTime += travelTime;
                                checkValue2 += travelTime;
                                clock += travelTime;
                                
                                if (clock < r.route[i + 1].TWStart)
                                {
                                    if (i!=0)
                                    {
                                        totalWaitingTime += r.route[i + 1].TWStart - clock;
                                        checkValue2 += (r.route[i + 1].TWStart - clock)*1.1;
                                    }  
                                    numViolations += 1;
                                    clock = r.route[i+1].TWStart;
                                }
                                else if (clock > r.route[i + 1].TWEnd)
                                {
                                    checkValue2 += 30;
                                    totalLatePercentage += 1;
                                }
                                load -= r.route[i + 1].Demand;
                                clock += r.route[i + 1].ServiceTime;
                            }
                        }
                    }
                }

            if (visitedCustomers.Count != customers.Count){failed = true;}
            Console.WriteLine($"Num Customers {numCustomers}");
            Console.WriteLine($"Total travel time {Math.Round(totalTravelTime/observations.GetLength(3),4)}");
            Console.WriteLine($"The service level of this route is {Math.Round(1 - totalLatePercentage/observations.GetLength(3)/numCustomers,4)}");
            Console.WriteLine($"The expected total waiting time: {Math.Round(totalWaitingTime/observations.GetLength(3),2)} over an average {numViolations/observations.GetLength(3)} customers");
            Console.WriteLine($"Recalculated KDE score with check score function: {Math.Round(checkValueKDE,2)}");
           
            var parent = new LocalSearch(actualConfig, random.Next());
            return (failed, ilpSol.ConvertAll(x => new Route(customers, x, customers[0], bikeMaxMass - bikeMinMass, observations, parent)), ilpVal, ilpTime, lsTime, lsVal, solutionJSON);
        }

        public double SolveVRPLTTInstance(string fileName, int numIterations = 3000000, double bikeMinMass = 150, double bikeMaxMass = 350, int numLoadLevels = 10, double inputPower = 400, int timelimit = 30000, LocalSearchConfiguration? config = null)
        {

            if (config == null)
                config = LocalSearchConfigs.VRPLTTDebug;

            (double[,] distances, List<Customer> customers) = VRPLTT.ParseVRPLTTInstance(fileName);

            (double[,,] matrix, Gamma[,,] distributionMatrix, IContinuousDistribution[,,] approximationMatrix, _) = VRPLTT.CalculateLoadDependentTimeMatrix(customers, distances, bikeMinMass, bikeMaxMass, numLoadLevels, inputPower, ((LocalSearchConfiguration)config).WindSpeed, ((LocalSearchConfiguration)config).WindDirection, ((LocalSearchConfiguration)config).DefaultDistribution is Normal);
            var ls = new LocalSearch((LocalSearchConfiguration)config, random.Next());
            (var colums, var sol, var value) = ls.LocalSearchInstance(0, "", customers.Count, bikeMaxMass - bikeMinMass, customers.ConvertAll(i => new Customer(i)), matrix, distributionMatrix, approximationMatrix, numInterations: numIterations, checkInitialSolution: false, timeLimit: timelimit);

            double totalDist = 0;
            double totalWait = 0;
            double totalOntimePercentage = 0;
            int numRoutes = 0;

            var simStopWatch = new Stopwatch();
            simStopWatch.Start();

            if (((LocalSearchConfiguration)config).UseStochasticFunctions)
            {
                foreach (var route in sol)
                {
                    if (route.route.Count != 2)
                    {
                        Console.WriteLine(route);
                        numRoutes++;
                        double avg = 0;
                        int total = 0;
                        double worst = double.MaxValue;
                        Customer? worstCust = null;
                        int worstIndex = -1;

                        for (int i = 0; i < route.route.Count; i++)
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

                        Console.WriteLine($"{route}: On time performance: {avg / total} worst: {worst} at {worstCust} at {worstIndex}");

                        var res = route.Simulate(2000000);
                        totalDist += res.AverageTravelTime;
                        totalWait += res.AverageWaitingTime;
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

                        Console.WriteLine($"Simmulated on time perfomance: {res.OnTimePercentage} worst: {min} at {minIndex}. {res.TotalToEarly} times to early and {res.TotalToLate} to late\n");
                    }

                }
            }
            simStopWatch.Stop();
            Console.WriteLine($"Simulating took {(double)simStopWatch.ElapsedMilliseconds / 1000} s");

            Console.WriteLine($"Average solution travel time: {totalDist} with OTP: {totalOntimePercentage / numRoutes}");
            Console.WriteLine($"Average solution waiting time: {totalWait}");

            VRPLTT.CalculateWindCyclingTime(fileName, bikeMinMass, bikeMaxMass, numLoadLevels, inputPower, ((LocalSearchConfiguration)config).WindDirection, sol);

            double totalWaitingTime = 0;
            int numViolations = 0;

            using (var sw = new StreamWriter("out.txt"))
                foreach (Route r in sol)
                {
                    if (r.route.Count > 2)
                        sw.WriteLine($"{r},");
                    var load = r.used_capacity;
                    for (int i = 0; i < r.route.Count - 1; i++)
                    {
                        (var dist, IContinuousDistribution distribution) = r.CustomerDist(r.route[i], r.route[i + 1], load, false);
                        load -= r.route[i + 1].Demand;
                        if (r.arrival_times[i] + dist + r.route[i].ServiceTime < r.route[i + 1].TWStart)
                        {
                            Console.WriteLine($"Waiting for customer {r.route[i + 1].Id}");

                            totalWaitingTime += r.route[i + 1].TWStart - (r.arrival_times[i] + dist + r.route[i].ServiceTime);
                            numViolations += 1;
                        }
                    }
                }
            Console.WriteLine($"Total waiting time: {totalWaitingTime} over {numViolations} customers");
            return value;
        }

        public double SolveVRPLTTInstanceKDE(string fileName, double[,,,] observations, int numIterations = 3000000, double bikeMinMass = 150, double bikeMaxMass = 290, int numLoadLevels = 10, double inputPower = 450, int timelimit = 30000, LocalSearchConfiguration? config = null)
        {
            if (config == null)
                config = LocalSearchConfigs.VRPLTTKDE;

            fileName += "/utrecht_full.csv";
            //Console.WriteLine($"the file name is {fileName}");
            (double[,] distances, List<Customer> customers) = VRPLTT.ParseVRPLTTInstance(fileName);

            var ls = new LocalSearch((LocalSearchConfiguration)config, random.Next());
            //numIterations = 100;
            (var columns, var sol, var value) = ls.LocalSearchInstanceKDE(0, "", customers.Count, bikeMaxMass - bikeMinMass, customers.ConvertAll(i => new Customer(i)), observations, numIterations: numIterations, checkInitialSolution: false, timeLimit: timelimit);

            double totalWaitingTime = 0;
            int numViolations = 0;

            using (var sw = new StreamWriter("out.txt"))
                foreach (Route r in sol)
                {
                    if (r.route.Count > 2)
                        sw.WriteLine($"{r},");
                    for (int j=0; j<observations.GetLength(3);j++)
                    {
                    var load = r.used_capacity;
                    double clock = 0;
                    
                    for (int i = 0; i < r.route.Count - 1; i++)
                    {
                        int ll = KDEdata.calculateKDELoadLevel(load);
                        double travelTime = observations[r.route[i].Id, r.route[i+1].Id, ll, j];
                        clock +=travelTime;
                        if (clock < r.route[i+1].TWStart)
                        {
                            totalWaitingTime += r.route[i].TWStart - clock;
                            clock = r.route[i].TWStart;
                            numViolations += 1;
                        }
                        clock += r.route[i].ServiceTime;
                        load -= r.route[i + 1].Demand;
                    }
                    }                
                }
            Console.WriteLine($"Total waiting time: {totalWaitingTime/observations.GetLength(3)} over {(int)numViolations/observations.GetLength(3)} customers");
            (List<RouteStore> bestSolution, double objectiveValue, double computationtime, string solutionJSON) = SolveILP(columns, customers, sol.Count, sol, 100);
            Console.WriteLine($"The objective value after the ILP is {Math.Round(objectiveValue,3)}, found in {Math.Round(computationtime,2)}.");
            
            return value;
        }

        public static void ErrorPrint(string toPrint)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine(toPrint);
            Console.ForegroundColor = ConsoleColor.White;
        }

        public static void PrintRoutes(List<Route> routes)
        {
            foreach (Route route in routes)
                if (route.route.Count != 2)
                    Console.WriteLine(route);
        }

        public async Task DoTest(string fileName, int numThreads = 1, int numIterations = 3000000, int timeLimit = 100000)
        {
            (string name, int numV, double capV, List<Customer> customers) = SolomonParser.ParseInstance(fileName);
            var distanceMatrix = CalculateDistanceMatrix(customers);
            await LocalSearchInstancAsync(name, numV, capV, customers, distanceMatrix, new Gamma[distanceMatrix.GetLength(0), distanceMatrix.GetLength(1), distanceMatrix.GetLength(2)], new Gamma[distanceMatrix.GetLength(0), distanceMatrix.GetLength(1), distanceMatrix.GetLength(2)], 4, 4, numIterations, timeLimit, LocalSearchConfigs.VRPTW);
        }

        public async Task<(bool failed, List<RouteStore> ilpSol, double ilpVal, double ilpTime, double lsTime, double lsVal, string solutionJSON)> SolveSolomonInstanceAsync(string fileName, int numThreads = 1, int numStarts = 4, int numIterations = 3000000, int timeLimit = 100000, LocalSearchConfiguration? config = null)
        {
            (string name, int numV, double capV, List<Customer> customers) = SolomonParser.ParseInstance(fileName);
            var distanceMatrix = CalculateDistanceMatrix(customers);


            if (config == null)
                config = LocalSearchConfigs.VRPTW;

            (List<RouteStore> ilpSol, double ilpVal, double ilpTime, double lsTime, double lsVal, string solutionJSON) = await SolveInstanceAsync(name, numV, capV, customers, distanceMatrix, new Gamma[distanceMatrix.GetLength(0), distanceMatrix.GetLength(1), distanceMatrix.GetLength(2)], new Gamma[distanceMatrix.GetLength(0), distanceMatrix.GetLength(1), distanceMatrix.GetLength(2)], numThreads, numStarts, numIterations, (LocalSearchConfiguration)config, timeLimit: timeLimit);
            bool failed = SolomonParser.CheckSolomonSolution(fileName, ilpSol, ilpVal);
            ilpSol.ForEach(x => Console.WriteLine(x));
            return (failed, ilpSol, ilpVal, ilpTime, lsTime, lsVal, solutionJSON);
        }

        public async Task<(HashSet<RouteStore> columns, List<Route> LSSolution, double LSTime, double LSVal)> LocalSearchInstancAsync(string name, int numV, double capV, List<Customer> customers, double[,,] distanceMatrix, Gamma[,,] distributionMatrix, IContinuousDistribution[,,] approximationMatrix, int numThreads, int numStarts, int numIterations, int timeLimit, LocalSearchConfiguration config)
        {
            List<Task<(HashSet<RouteStore>, List<Route>, double)>> tasks = new List<Task<(HashSet<RouteStore>, List<Route>, double)>>();

            Stopwatch watch = new Stopwatch();
            watch.Start();

            HashSet<RouteStore> allColumns = new HashSet<RouteStore>();
            List<Route> bestSolution = new List<Route>();
            int cnt = 0;
            double bestLSVal = double.MaxValue;

            for (int j = 0; j < numStarts; j += numThreads)
            {
                for (int i = 0; i < numThreads && i + j < numStarts; i++)
                {
                    var id = i;
                    var ls = new LocalSearch(config, random.Next());
                    tasks.Add(Task.Run(() => { return ls.LocalSearchInstance(id, name, numV, capV, customers.ConvertAll(i => new Customer(i)), distanceMatrix, distributionMatrix, approximationMatrix, numInterations: numIterations, timeLimit: timeLimit); }));
                }

                foreach (var task in tasks)
                {
                    (var columns, var solution, var value) = await task;
                    if (value < bestLSVal)
                    {
                        bestLSVal = value;
                        bestSolution = solution;

                    }
                    allColumns.UnionWith(columns);
                    cnt += columns.Count;
                }
            }
            watch.Stop();
            Console.WriteLine();
            Console.WriteLine($" Sum of unique columns found per start: {cnt}");
            Console.WriteLine($" Total amount of unique column: {allColumns.Count}");

            return (allColumns, bestSolution, (double)watch.ElapsedMilliseconds / 1000, bestLSVal);
        }
        public async Task<(HashSet<RouteStore> columns, List<Route> LSSolution, double LSTime, double LSVal)> LocalSearchInstancAsyncKDE(string name, int numV, double capV, List<Customer> customers, double[,,,] observations, int numThreads, int numStarts, int numInterations, int timeLimit, LocalSearchConfiguration config, int repetition = 0)
        {
            List<Task<(HashSet<RouteStore>, List<Route>, double)>> tasks = new List<Task<(HashSet<RouteStore>, List<Route>, double)>>();

            Stopwatch watch = new Stopwatch();
            watch.Start();

            HashSet<RouteStore> allColumns = new HashSet<RouteStore>();
            List<Route> bestSolution = new List<Route>();
            int cnt = 0;
            double bestLSVal = double.MaxValue;

            for (int j = 0; j < numStarts; j += numThreads)
            {
                for (int i = 0; i < numThreads && i + j < numStarts; i++)
                {
                    var id = i;
                    var ls = new LocalSearch(config, random.Next());
                    tasks.Add(Task.Run(() => { return ls.LocalSearchInstanceKDE(id, name, numV, capV, customers.ConvertAll(i => new Customer(i)), observations, numIterations: numInterations, timeLimit: timeLimit, repetition:repetition); }));
                }

                foreach (var task in tasks)
                {
                    (var columns, var solution, var value) = await task;
                    if (value < bestLSVal)
                    {
                        bestLSVal = value;
                        bestSolution = solution;
                    }
                    allColumns.UnionWith(columns);
                    cnt += columns.Count;
                }
            }
            watch.Stop();
            Console.WriteLine();
            Console.WriteLine($" Sum of unique columns found per start: {cnt}");
            Console.WriteLine($" Total amount of unique column: {allColumns.Count}");

            return (allColumns, bestSolution, (double)watch.ElapsedMilliseconds / 1000, bestLSVal);
        }

        public async Task<(List<RouteStore> ilpSol, double ilpVal, double ilpTime, double lsTime, double lsVal, string solutionJSON)> SolveInstanceAsync(string name, int numV, double capV, List<Customer> customers, double[,,] distanceMatrix, Gamma[,,] distributionMatrix, IContinuousDistribution[,,] approximationMatrix, int numThreads, int numStarts, int numIterations, LocalSearchConfiguration configuration, int timeLimit = 30000, double ilpTimelimit = 3600)
        {
            (var allColumns, var bestSolution, var LSTime, var LSSCore) = await LocalSearchInstancAsync(name, numV, capV, customers, distanceMatrix, distributionMatrix, approximationMatrix, numThreads, numStarts, numIterations, timeLimit, configuration);
            PrintRoutes(bestSolution);
            (var ilpSol, double ilpVal, double time, string solutionJSON) = SolveILP(allColumns, customers, numV, bestSolution, ilpTimelimit);

            return (ilpSol, ilpVal, time, LSTime, LSSCore, solutionJSON);
        }
        public async Task<(List<RouteStore> ilpSol, double ilpVal, double ilpTime, double lsTime, double lsVal, string solutionJSON)> SolveInstanceAsyncKDE(string name, int numV, double capV, List<Customer> customers, double[,,,] observations, int numThreads, int numStarts, int numIterations, LocalSearchConfiguration configuration, int timeLimit = 30000, double ilpTimelimit = 3600, int repetition=0)
        {
            (var allColumns, var bestSolution, var LSTime, var LSScore) = await LocalSearchInstancAsyncKDE(name, numV, capV, customers, observations, numThreads, numStarts, numIterations, timeLimit, configuration, repetition:repetition);
            PrintRoutes(bestSolution);
            (var ilpSol, double ilpVal, double time, string solutionJSON) = SolveILP(allColumns, customers, numV, bestSolution, ilpTimelimit);
            return (ilpSol, ilpVal, time, LSTime, LSScore, solutionJSON);
        }

        //Implements the VRPLTT ILP model to check the quality of the routes provided by the local search
        private void CheckRouteQualityVRPLTT(List<Route> routes, double[,,] distanceMatrix, double maxLoad)
        {


            foreach (var route in routes)
            {
                if (route.route.Count == 2)
                    continue;
                Console.WriteLine($"LS route score: {route.Score}");
                GRBEnv env = new GRBEnv();
                env.LogToConsole = 0;
                GRBModel model = new GRBModel(env);

                int numCust = route.route.Count - 1;

                GRBVar[,] edgeX = new GRBVar[route.route.Count - 1, route.route.Count - 1];
                GRBVar[] arrivalX = new GRBVar[numCust];
                GRBVar[,] vehicleWeight = new GRBVar[route.route.Count - 1, route.route.Count - 1];
                GRBVar[,,] loadLevelEdgeX = new GRBVar[route.route.Count - 1, route.route.Count - 1, distanceMatrix.GetLength(2)];

                double llWidth = maxLoad / distanceMatrix.GetLength(2);
                double load = route.route.Sum(x => x.Demand);

                for (int i = 0; i < route.route.Count - 1; i++)
                {
                    arrivalX[i] = model.AddVar(0, 1000, 0, GRB.CONTINUOUS, $"y_{i}");
                    arrivalX[i].Start = 0;//route.arrival_times[i];

                    int loadLevel = (int)((Math.Max(0, load - 0.000001) / maxLoad) * distanceMatrix.GetLength(2));
                    load -= route.route[i].Demand;


                    if (loadLevel == distanceMatrix.GetLength(2))
                        loadLevel--;

                    for (int j = 0; j < route.route.Count - 1; j++)
                    {
                        edgeX[i, j] = model.AddVar(0, 1, 0, GRB.BINARY, $"X{i}_{j}"); //(0, 1, GRB.BINARY, $"X{i}_{j}"
                        edgeX[i, j].Start = 0;
                        vehicleWeight[i, j] = model.AddVar(0, maxLoad, 0, GRB.CONTINUOUS, $"F{i}_{j}");
                        for (int l = 0; l < distanceMatrix.GetLength(2); l++)
                        {
                            loadLevelEdgeX[i, j, l] = model.AddVar(0, 1, distanceMatrix[route.route[i].Id, route.route[j].Id, l], GRB.BINARY, $"z{i}_{j}_{l}");
                            loadLevelEdgeX[i, j, l].Start = 0;
                        }
                    }

                    //Warm start
                    if (i != numCust - 1)
                    {
                        edgeX[i, i + 1].Start = 1;
                        loadLevelEdgeX[i, i + 1, loadLevel].Start = 1;
                        vehicleWeight[i, i + 1].Start = load;
                    }
                    else
                    {
                        edgeX[i, 0].Start = 1;
                        loadLevelEdgeX[i, 0, loadLevel].Start = 1;
                        vehicleWeight[i, 0].Start = 0;
                    }
                }

                for (int i = 0; i < numCust; i++)
                {


                    GRBLinExpr weightMatch = 0;
                    for (int j = 0; j < numCust; j++)
                    {

                        weightMatch.AddTerm(1, vehicleWeight[j, i]);
                        weightMatch.AddTerm(-1, vehicleWeight[i, j]);

                        GRBLinExpr totalLLToEdge = 0;

                        GRBLinExpr llLowerBound = 0;
                        GRBLinExpr llUpperBound = 0;

                        GRBLinExpr llTravelTime = 0;
                        for (int l = 0; l < distanceMatrix.GetLength(2); l++)
                        {
                            totalLLToEdge.AddTerm(1, loadLevelEdgeX[i, j, l]);


                            llLowerBound.AddTerm(l * llWidth, loadLevelEdgeX[i, j, l]);
                            llUpperBound.AddTerm((l + 1) * llWidth, loadLevelEdgeX[i, j, l]);

                            llTravelTime.AddTerm(distanceMatrix[route.route[i].Id, route.route[j].Id, l], loadLevelEdgeX[i, j, l]);

                        }


                        double Mij = Math.Max(0, route.route[i].TWEnd + route.route[i].ServiceTime + distanceMatrix[route.route[i].Id, route.route[j].Id, distanceMatrix.GetLength(2) - 1] - route.route[j].TWStart);

                        //Constraint 14
                        if (i != j && j != 0)
                        {
                            model.AddConstr(arrivalX[i] - arrivalX[j] + route.route[i].ServiceTime + llTravelTime <= Mij * (1 - edgeX[i, j]), $"Updating traveltimes ({i},{j})");
                        }

                        //Constraint 12
                        model.AddConstr(totalLLToEdge == edgeX[i, j], $"loadLevelXMatchesEdgeX ({i},{j})");

                        //Constraint 13
                        model.AddConstr(llLowerBound <= vehicleWeight[i, j], $"weight constraint l ({i},{j})");
                        model.AddConstr(llUpperBound >= vehicleWeight[i, j], $"weight constraint u ({i},{j})");

                        //Constraint 11
                        model.AddConstr(route.route[j].Demand * edgeX[i, j] <= vehicleWeight[i, j], $"VehiclecapConst ({i},{j})");
                        model.AddConstr(vehicleWeight[i, j] <= (maxLoad - route.route[i].Demand) * edgeX[i, j], $"VehiclecapConstUp ({i},{j})");
                    }
                    //Constraint 10
                    if (i != 0)
                        model.AddConstr(weightMatch == route.route[i].Demand, "Match weight");
                }

                //Outgoing edges
                for (int i = 0; i < numCust; i++)
                {
                    GRBLinExpr custTotalOut = 0;
                    GRBLinExpr custTotalIn = 0;
                    for (int j = 0; j < numCust; j++)
                    {
                        custTotalOut.AddTerm(1, edgeX[i, j]);
                        custTotalIn.AddTerm(1, edgeX[j, i]);
                    }

                    //Constraint 9
                    model.AddConstr(custTotalOut == 1, $"CustIn{route.route[i].Id}");
                    //Constraint 8
                    model.AddConstr(custTotalIn == 1, $"CustOut{route.route[i].Id}");

                }


                //timeWindows
                for (int i = 1; i < arrivalX.Length; i++)
                {
                    //Constraint 15
                    model.AddConstr(arrivalX[i] <= route.route[i].TWEnd, $"u{route.route[i].Id}");
                    model.AddConstr(route.route[i].TWStart <= arrivalX[i], $"l{route.route[i].Id}");
                }


                model.ModelSense = GRB.MINIMIZE;

                model.Optimize();

                List<(int, int, double, int)> r = new List<(int, int, double, int)>();

                for (int i = 0; i < numCust; i++)
                {
                    for (int j = 0; j < numCust; j++)
                    {
                        if (edgeX[i, j].X == 1)
                        {
                            int loadLevel = -1;
                            for (int l = 0; l < distanceMatrix.GetLength(2); l++)
                                if (loadLevelEdgeX[i, j, l].X == 1)
                                    loadLevel = l;
                            r.Add((route.route[i].Id, route.route[j].Id, vehicleWeight[i, j].X, loadLevel));
                        }

                    }
                }
                r.Sort((x, y) => y.Item3.CompareTo(x.Item3));

                Console.WriteLine($"ILp route: ({String.Join(',', r.ConvertAll(x => $"{x.Item1}"))})");
                Console.WriteLine($"ILP loadlevels: ({String.Join(',', r.ConvertAll(x => $"{x.Item4}"))})");
                Console.WriteLine($"ILP score: {model.ObjVal}");

                var test = edgeX.Cast<GRBVar>().ToArray();
                var res = test.ToList().Any(x => x.Start != x.X);
                Console.WriteLine($"ILP found different route: {res}");

            }
        }

        private (List<RouteStore>, double, double, string solutionJSON) SolveILP(HashSet<RouteStore> columns, List<Customer> customers, int numVehicles, List<Route> bestSolutionLS, double timelimit = 3600)
        {
            var columList = columns.ToArray();
            var bestSolStore = bestSolutionLS.ConvertAll(x => new RouteStore(x.CreateIdList(), x.Score));
            byte[,] custInRoute = new byte[customers.Count, columList.Length];
            for (int i = 0; i < columList.Length; i++)
            {
                for (int j = 0; j < columList[i].Route.Count - 1; j++)
                {
                    custInRoute[columList[i].Route[j], i] = 1;
                    custInRoute[columList[i].Route[j + 1], i] = 1;
                }
            }

            //Create model
            GRBEnv env = new GRBEnv();
            GRBModel model = new GRBModel(env);

            model.Parameters.TimeLimit = timelimit;
            model.Parameters.Threads = 16;

            //Create decision variables
            GRBVar[] columnDecisions = new GRBVar[columList.Length];
            for (int i = 0; i < columList.Length; i++)
            {
                columnDecisions[i] = model.AddVar(0, 1, columList[i].Value, GRB.BINARY, $"X{i}");
            }

            foreach (Customer cust in customers)
            {
                if (cust.Id == 0)
                    continue;
                GRBLinExpr custTotal = 0;
                for (int i = 0; i < columList.Length; i++)
                    custTotal.AddTerm(custInRoute[cust.Id, i], columnDecisions[i]);
                model.AddConstr(custTotal == 1, $"Cust{cust.Id}");
            }

            GRBLinExpr numV = 0;
            for (int i = 0; i < columList.Length; i++)
                numV.AddTerm(1, columnDecisions[i]);
            model.AddConstr(numV <= numVehicles, "Bound vehicles");
            model.ModelSense = GRB.MINIMIZE;

            //Add warm start
            for (int i = 0; i < columList.Length; i++)
            {
                if (bestSolStore.Contains(columList[i]))
                {
                    columnDecisions[i].Start = 1;
                }
            }

            model.Optimize();
            List<RouteStore> solution = new List<RouteStore>();
            try
            {
                for (int i = 0; i < columnDecisions.Length; i++)
                {
                    if (Math.Round(columnDecisions[i].X, 6) == 1)
                        solution.Add(columList[i]);
                }
            }
            catch
            {
                Console.WriteLine("FAIL: Gurobi could not find improved solution.");
                foreach (var route in bestSolutionLS)
                {
                    if (route.CreateIdList().Count() <= 2)
                        continue;
                    RouteStore routeStore = new RouteStore(route.CreateIdList(), route.CalcObjectiveKDE());
                    solution.Add(routeStore);
                }
            }
            string solutionJSON = model.GetJSONSolution();
            double val = model.ObjVal;
            double time = model.Runtime;
            model.Dispose();
            return (solution, val, time, solutionJSON);
        }
    }

    static class SolomonParser
    {
        public static (string, int, double, List<Customer>) ParseInstance(string fileName)
        {
            using (var reader = new StreamReader(fileName))
            {
                var name = reader.ReadLine().Replace("\n", "");

                reader.ReadLine();
                reader.ReadLine();
                reader.ReadLine();

                var reg = new Regex("  +");
                var line = reader.ReadLine().Replace("\n", "");
                var lineSplit = reg.Replace(line, " ").Split(' ');
                int numV = int.Parse(lineSplit[1]);
                int capV = int.Parse(lineSplit[2]);

                reader.ReadLine();
                reader.ReadLine();
                reader.ReadLine();
                reader.ReadLine();

                List<Customer> customers = new List<Customer>();
                while ((line = reader.ReadLine()) != null && line != "")
                {
                    line = line.Replace("\n", "");
                    lineSplit = reg.Replace(line, " ").Split(' ');
                    customers.Add(new Customer(int.Parse(lineSplit[1]), int.Parse(lineSplit[2]), int.Parse(lineSplit[3]), int.Parse(lineSplit[4]), int.Parse(lineSplit[5]), int.Parse(lineSplit[6]), int.Parse(lineSplit[7])));
                }
                return (name, numV, capV, customers);
            }
        }

        public static bool CheckSolomonSolution(string instance, List<RouteStore> solution, double value)
        {
            (string name, int numV, double capV, List<Customer> customers) = SolomonParser.ParseInstance(instance);

            double totalDist = 0;
            HashSet<int> CustomersVisited = new HashSet<int>();
            bool failedTotal = false;
            foreach (var route in solution)
            {
                double arrivalTime = 0;
                bool failed = false;
                double usedCapacity = 0;
                for (int i = 0; i < route.Route.Count; i++)
                {
                    var res = CustomersVisited.Add(route.Route[i]);
                    //If the customer is visited twice and it is not the depot. Fail the solution
                    if (route.Route[i] != 0 && !res)
                    {
                        failed = true;
                        Console.WriteLine($"FAIL. Visited customer {route.Route[i]} more than once");
                    }
                    //Check capacity
                    usedCapacity += customers[route.Route[i]].Demand;
                    if (usedCapacity > capV)
                    {
                        failed = true;
                        Console.WriteLine($"FAIL. Exceeded vehicle capacity");
                    }
                    if (arrivalTime > customers[route.Route[i]].TWStart)
                        arrivalTime = customers[route.Route[i]].TWStart;

                    //Update travel distances
                    if (i != route.Route.Count - 1)
                    {
                        double dist = Solver.CalculateDistanceObjective(customers[route.Route[i]], customers[route.Route[i + 1]]);
                        totalDist += dist;
                        arrivalTime += dist + customers[route.Route[i]].ServiceTime;
                    }


                }
                if (!failed)
                    Console.WriteLine("PASSED");
                failedTotal |= failed;
            }

            //Check if all customers were visited
            if (CustomersVisited.Count != customers.Count)
            {
                failedTotal = true;
                Console.WriteLine("Did not visit all customers");
            }
            //Check if the solution values match
            if (Math.Round(totalDist, 6) != Math.Round(value, 6))
            {
                failedTotal = true;
                Console.WriteLine($"Wrong objective value reported. Epected {value.ToString("0.000")} does not match found {totalDist.ToString("0.000")}");
            }
            return failedTotal;

        }

    }


}
