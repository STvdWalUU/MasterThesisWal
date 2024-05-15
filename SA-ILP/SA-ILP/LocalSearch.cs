using MathNet.Numerics.Distributions;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SA_ILP
{
    internal class LocalSearch
    {

        private static readonly object ConsoleWriterLock = new object();

        public double Temperature { get; private set; }


        public readonly LocalSearchConfiguration Config;

        private Random random;

        public OperatorSelector OS;

        public LocalSearch(LocalSearchConfiguration config, int seed, OperatorSelector os)
        {
            random = new Random(seed);
            Init(config, seed, os);
            Config = config;
        }


        private void Init(LocalSearchConfiguration config, int seed, OperatorSelector os)
        {
            Temperature = config.InitialTemperature;
            OS = os;
        }

        public LocalSearch(LocalSearchConfiguration config, int seed)
        {
            random = new Random(seed);

            Config = config;

            OperatorSelector os = new OperatorSelector(random);
            if (config.Operators == null)
            {
                throw new Exception("Please set the operators in the local search configuration");
            }
            //Add the operators to the operator selector
            else
                foreach (var op in config.Operators)
                {
                    os.Add(op.Item1, op.Item2, op.Item3, op.Item4);
                }


            Init(config, seed, os);

        }

        //Randomly shuffle a list of customers
        public static void EasyShuffle(List<Customer> customers, Random random, double timesShuffle = 1)
        {
            for (int i = 0; i < timesShuffle * customers.Count; i++)
            {
                int index1 = random.Next(1, customers.Count);
                int index2 = random.Next(1, customers.Count);

                var temp = customers[index1];
                customers[index1] = customers[index2];
                customers[index2] = temp;
            }


        }

        //Creates a nearest neighbor based initial solution
        private void CreateSmartInitialSolution(List<Route> routes, List<Customer> customers, List<Customer> removed)
        {
            //Creating a copy of the array such that customer can be removed
            customers = customers.ConvertAll(x => x);
            Customer depot = customers[0];
            customers.RemoveAt(0);

            Customer? seed = customers.MinBy(x => x.TWEnd);//customers[0];

            List<Customer> inserted = new List<Customer>();

            foreach (Route route in routes)
            {
                if (customers.Count == 0)
                    break;
                (double arrivalTime, IContinuousDistribution distribution) = route.CustomerDist(depot, seed, route.max_capacity, false);

                if (inserted.Contains(seed))
                    Console.WriteLine("Gaat fout buiten while");

                route.InsertCust(seed, route.route.Count - 1);
                inserted.Add(seed);
                customers.Remove(seed);
                //Add all customers
                while (customers.Count != 0)
                {
                    //Set the arrival time at the timewindow start of the first customer
                    if (arrivalTime < seed.TWStart)
                        arrivalTime = seed.TWStart;
                    arrivalTime += +seed.ServiceTime;

                    //Select the next customer by minimizing costs (with a random factor)
                    Customer? next = customers.MinBy(x =>
                    {
                        (double dist, IContinuousDistribution distribution) = route.CustomerDist(seed, x, route.max_capacity, false);

                        if (arrivalTime + dist < x.TWEnd)
                            if (arrivalTime + dist < x.TWStart)
                                return dist + route.CalculateEarlyPenaltyTerm(arrivalTime + dist, x.TWStart);
                            else
                                return dist + random.NextDouble() * 5;
                        else return double.MaxValue; //dist + route.CalculateLatePenaltyTerm(arrivalTime + dist, x.TWEnd);//

                    });
                    (double dist, IContinuousDistribution d) = route.CustomerDist(seed, next, route.max_capacity, false);
                    if (arrivalTime + dist > next.TWEnd || route.used_capacity + next.Demand > route.max_capacity)
                    {
                        seed = customers.MinBy(x => x.TWEnd);
                        break;
                    }
                    arrivalTime += dist;// + next.ServiceTime;

                    seed = next;



                    route.InsertCust(seed, route.route.Count - 1);
                    customers.Remove(seed);
                    inserted.Add(seed);
                    if (customers.Count == 0)
                        seed = null;

                }
                //route.InsertCust(next, route.route.Count-1);

            }

            //Not all customers could be aded using the neirest neighbor method. Insert them greedily
            if (customers.Count != 0)
            {
                foreach (Customer cust in customers)
                {
                    Route best = null;
                    double bestVal = Double.PositiveInfinity;
                    int bestPos = 0;
                    foreach (Route route in routes)
                    {
                        (int pos, double val) = route.BestPossibleInsert(cust);
                        if (val < bestVal)
                        {
                            bestVal = val;
                            best = route;
                            bestPos = pos;
                        }
                    }
                    best.InsertCust(cust, bestPos);
                }
            }
        }
        private void CreateSmartInitialSolutionKDE(List<Route> routes, List<Customer> customers, List<Customer> removed, double[,,,] observations)
        {
            //Creating a copy of the array such that customers can be removed
            customers = customers.ConvertAll(x => x);
            Customer depot = customers[0];
            customers.RemoveAt(0);

            Customer? seed = customers.MinBy(x => x.TWEnd);//customers[0];
            List<Customer> inserted = new List<Customer>();

            foreach (Route route in routes)
            {
                if (customers.Count == 0)
                    break;
                
                double clock = observations[depot.Id, seed.Id, observations.GetLength(2)-1, 0];
                
                if (inserted.Contains(seed))
                    Console.WriteLine("Gaat fout buiten while");

                route.InsertCust(seed, route.route.Count - 1);
                inserted.Add(seed);
                customers.Remove(seed);
                //Add all customers
                while (customers.Count != 0)
                {
                    //Set the arrival time at the timewindow start of the first customer
                    if (clock < seed.TWStart)
                        clock = seed.TWStart;
                    clock += seed.ServiceTime;

                    //Select the next customer by minimizing costs (with a random factor)
                    Customer? next = customers.MinBy(x =>
                    {
                        //(double dist, IContinuousDistribution distribution) = route.CustomerDist(seed, x, route.max_capacity, false);
                        double travelTime = route.observationMatrix[seed.Id, x.Id, route.observationMatrix.GetLength(2)-1, 0];

                        if (clock + travelTime < x.TWEnd)
                            if (clock + travelTime< x.TWStart)
                                return travelTime + route.CalculateEarlyPenaltyTermKDE(clock + travelTime, x.TWStart, Config.WaitingPenalty);
                            else
                                return travelTime + random.NextDouble() * 5;
                        else return double.MaxValue; //dist + route.CalculateLatePenaltyTerm(arrivalTime + dist, x.TWEnd);//

                    });
                    double travelTime = route.observationMatrix[seed.Id, next.Id, route.observationMatrix.GetLength(2)-1, 0];

                    //(double dist, IContinuousDistribution d) = route.CustomerDist(seed, next, route.max_capacity, false);
                    if (clock + travelTime > next.TWEnd || route.used_capacity + next.Demand > route.max_capacity)
                    {
                        seed = customers.MinBy(x => x.TWEnd);
                        break;
                    }
                    clock += travelTime;
                    seed = next;
                    route.InsertCust(seed, route.route.Count - 1);
                    customers.Remove(seed);
                    inserted.Add(seed);
                    if (customers.Count == 0)
                        seed = null;
                }
            }

            //Not all customers could be aded using the neirest neighbor method. Insert them greedily
            if (customers.Count != 0)
            {
                foreach (Customer cust in customers)
                {
                    Route best = null;
                    double bestVal = Double.PositiveInfinity;
                    int bestPos = 0;
                    foreach (Route route in routes)
                    {
                        (int pos, double val) = route.BestPossibleInsert(cust);
                        if (val < bestVal)
                        {
                            bestVal = val;
                            best = route;
                            bestPos = pos;
                        }
                    }
                    best.InsertCust(cust, bestPos);
                }
            }
        }

        private bool IsValidSolution(List<Route> routes, List<Customer> removed)
        {
            //Always require all customers to be served for a route to be valid
            if (removed.Count != 0)
                return false;
            bool upperViolations = routes.Exists(x => x.ViolatesUpperTimeWindow);
            bool lowerViolations = routes.Exists(x => x.ViolatesLowerTimeWindow);

            return (!upperViolations || Config.AllowLateArrival) && (!lowerViolations || Config.AllowDeterministicEarlyArrival);
        }

        private bool IsValidRoute(Route route)
        {
            return (!route.ViolatesUpperTimeWindow || Config.AllowLateArrival) && (!route.ViolatesLowerTimeWindow || Config.AllowDeterministicEarlyArrival);
        }

        //Executes operator and check the score if this check is enabled
        private void RunAndCheckOperator(int id, List<Route> routes, List<Customer> removed, double imp, Action op)
        {
            if (Config.useKDE && Config.CheckOperatorScores)
                {
                    RunAndCheckOperatorKDE(id, routes, removed, imp, op);
                }
            double expectedVal = 0;
            List<Route>? beforeCopy = null;
            if (Config.CheckOperatorScores)
            {    
                expectedVal = Solver.CalcTotalDistance(routes, removed, this) - imp;
            }
            
            if (Config.SaveRoutesBeforeOperator)
            {
                beforeCopy = routes.ConvertAll(i => i.CreateDeepCopy());
            }

            op();
            if (Config.CheckOperatorScores)
                if (Math.Round(Solver.CalcTotalDistance(routes, removed, this), 6) != Math.Round(expectedVal, 6))
                    Solver.ErrorPrint($"{id}: ERROR expected {expectedVal} not equal to {Solver.CalcTotalDistance(routes, removed, this)} with imp: {imp}. Diff:{expectedVal - Solver.CalcTotalDistance(routes, removed, this)} , OP: {OS.LastOperator}");
        }
        private void RunAndCheckOperatorKDE(int id, List<Route> routes, List<Customer> removed, double imp, Action op)
        {
            double expectedVal = 0;
            List<Route>? beforeCopy = null;
            if (Config.CheckOperatorScores)
            {    
                expectedVal = Solver.CalcTotalDistanceKDE(routes, removed, this) - imp;
            }
            
            if (Config.SaveRoutesBeforeOperator)
            {
                beforeCopy = routes.ConvertAll(i => i.CreateDeepCopyKDE());
            }

            op();
            
            if (Config.CheckOperatorScores)
                if (Math.Round(Solver.CalcTotalDistanceKDE(routes, removed, this), 6) != Math.Round(expectedVal, 6))
                    {   
                        Solver.ErrorPrint($"{id}: ERROR expected {expectedVal} not equal to {Solver.CalcTotalDistanceKDE(routes, removed, this)} with imp: {imp}. Diff:{expectedVal - Solver.CalcTotalDistanceKDE(routes, removed, this)} , OP: {OS.LastOperator}");
                        Console.WriteLine("Route before:");
                        Solver.PrintRoutes(beforeCopy);
                        double loadBefore = 0;
                        foreach (Route route in beforeCopy)
                            loadBefore+= route.used_capacity;
                        //Console.WriteLine($"Total load before operator: {loadBefore}");
                        Console.WriteLine("Route after:");
                        Solver.PrintRoutes(routes);
                        double loadAfter = 0;
                        foreach (Route route in routes)
                            loadAfter+= route.used_capacity;
                        //Console.WriteLine($"Total load before operator: {loadAfter}");
                    } 
        }

        //Actualy perform the local search
        public (HashSet<RouteStore>, List<Route>, double) LocalSearchInstance(int id, string name, int numVehicles, double vehicleCapacity, List<Customer> customers, double[,,] distanceMatrix, Gamma[,,] distributionMatrix, IContinuousDistribution[,,] approximationMatrix, int numInterations = 3000000, int timeLimit = 30000, bool checkInitialSolution = false)
        {
            Console.WriteLine("Starting local search");

            EasyShuffle(customers, random);

            List<Route> routes = new List<Route>();
            List<Customer> removed = new List<Customer>();

            //Generate routes
            for (int i = 0; i < numVehicles; i++)
                routes.Add(new Route(customers[0], distanceMatrix, distributionMatrix, approximationMatrix, vehicleCapacity, seed: random.Next(), this));

            CreateSmartInitialSolution(routes, customers, removed);
            //CreateStupidInitialSolution(routes, customers, removed);

            if (checkInitialSolution)
                //Validate intial solution
                foreach (Route route in routes)
                    if (route.CheckRouteValidity())
                    {
                        Console.WriteLine("Initiele oplossing niet valid");
                        throw new Exception();
                    }

            Console.WriteLine($"Finished making initial solution with value {Solver.CalcTotalDistance(routes, removed, this).ToString("0.000")}");
            //Console.WriteLine($"With KDE value {Solver.CalcTotalDistanceKDE(routes, removed, this).ToString("0.000")}");

            int amtImp = 0, amtWorse = 0, amtNotDone = 0;
            //double temp = initialTemp;
            double totalP = 0;
            double countP = 0;
            Stopwatch timer = new Stopwatch();
            timer.Start();
            Stopwatch printTimer = new Stopwatch();
            printTimer.Start();

            //Used to keep track of the routes which contain customers. This speeds up execution as this only needs to be updated when a change is applied
            List<int> viableRoutes = Enumerable.Range(0, routes.Count).Where(i => routes[i].route.Count > 2).ToList();

            int lastChangeAcceptedOnIt = 0;

            //All saved columns
            HashSet<RouteStore> Columns = new HashSet<RouteStore>();

            //The best found solution so far
            List<Route> BestSolution = routes.ConvertAll(i => i.CreateDeepCopy());
            List<Customer> BestSolutionRemoved = removed.ConvertAll(x => x);

            List<(int, double)> SearchScores = new List<(int, double)>();
            List<(int, double)> BestSolutionScores = new List<(int, double)>();


            //Dictionaries used for tracking the performance of the different operators. These are only used if the option is enabled in the configuration
            Dictionary<string, int> OPImprovementCount = new Dictionary<string, int>();
            Dictionary<string, double> OPImprovementTotal = new Dictionary<string, double>();
            Dictionary<string, double> OPBestImprovement = new Dictionary<string, double>();

            Dictionary<string, int> OPAcceptedWorseCount = new Dictionary<string, int>();
            Dictionary<string, double> OPAcceptedWorseTotal = new Dictionary<string, double>();
            Dictionary<string, int> OPNotPossible = new Dictionary<string, int>();

            foreach (string op in OS.OperatorList)
            {
                OPImprovementCount[op] = 0;
                OPImprovementTotal[op] = 0;
                OPBestImprovement[op] = 0;
                OPAcceptedWorseTotal[op] = 0;
                OPAcceptedWorseCount[op] = 0;
                OPNotPossible[op] = 0;
            }

            double bestSolValue = Solver.CalcTotalDistance(BestSolution, removed, this);
            //double bestSolValueKDE = Solver.CalcTotalDistanceKDE(BestSolution, removed, this);
            double currentValue = bestSolValue;
            int bestImprovedIteration = 0;
            int restartPreventionIteration = 0;
            int numRestarts = 0;
            int previousUpdateIteration = 0;
            int iteration = 0;
            for (; iteration < numInterations && timer.ElapsedMilliseconds <= timeLimit; iteration++)
            {
                    
                double imp = 0;
                Action? act = null;
                var nextOperator = OS.Next();
                (imp, act) = nextOperator(routes, viableRoutes, random, removed, this);

                if (act != null)
                {
                    if (currentValue > double.MaxValue - 1000000)
                        Console.WriteLine("OVERFLOW");
                    //Accept all improvements
                    if (imp > 0)
                    {

                        OPImprovementCount[OS.LastOperator] += 1;
                        OPImprovementTotal[OS.LastOperator] += imp;

                        if (imp > OPBestImprovement[OS.LastOperator])
                            OPBestImprovement[OS.LastOperator] = imp;

                        //Apply the changes of the neighbor
                        RunAndCheckOperator(id, routes, removed, imp, act);

                        //Update viable routes cache
                        viableRoutes = Enumerable.Range(0, routes.Count).Where(i => routes[i].route.Count > 2).ToList();

                        currentValue -= imp;
                        if (id == 0 && Config.SaveScoreDevelopment)
                            SearchScores.Add((iteration, currentValue));

                        //Check wheter it improves the best known solution
                        if (Math.Round(currentValue, 6) < Math.Round(bestSolValue, 6) && removed.Count == 0 && IsValidSolution(routes, removed))
                        {
                            //New best solution found
                            bestSolValue = currentValue;
                            bestImprovedIteration = iteration;
                            restartPreventionIteration = iteration;
                            //Create a copy
                            BestSolution = routes.ConvertAll(i => i.CreateDeepCopy());

                            //Score development can be saved if enabled
                            if (id == 0 && Config.SaveScoreDevelopment)
                                BestSolutionScores.Add((iteration, bestSolValue));

                            //Now we know all customers are used
                            BestSolutionRemoved = new List<Customer>();
                            foreach (Route route in routes)
                            {
                                if (IsValidRoute(route))
                                    Columns.Add(new RouteStore(route.CreateIdList(), route.Score));

                            }

                        }

                        //Add columns to the column store
                        if (Config.SaveColumnsAfterAllImprovements && Temperature < Config.InitialTemperature * Config.SaveColumnThreshold)
                            foreach (Route route in routes)
                            {
                                if (IsValidRoute(route))
                                    Columns.Add(new RouteStore(route.CreateIdList(), route.Score));

                            }


                        amtImp += 1;
                        lastChangeAcceptedOnIt = iteration;

                    }

                    else
                    {
                        double acceptP = Math.Exp(imp / Temperature);
                        totalP += acceptP;
                        countP += 1;
                        if (random.NextDouble() <= acceptP)
                        {
                            //Worse solution accepted
                            amtWorse += 1;


                            OPAcceptedWorseCount[OS.LastOperator] += 1;
                            OPAcceptedWorseTotal[OS.LastOperator] += imp;

                            RunAndCheckOperator(id, routes, removed, imp, act);


                            if (Config.SaveColumnsAfterWorse && Temperature < Config.InitialTemperature * Config.SaveColumnThreshold)
                                foreach (Route route in routes)
                                {
                                    if (IsValidRoute(route))
                                        Columns.Add(new RouteStore(route.CreateIdList(), route.Score));

                                }

                            viableRoutes = Enumerable.Range(0, routes.Count).Where(i => routes[i].route.Count > 2).ToList();
                            currentValue -= imp;
                            if (id == 0 && Config.SaveScoreDevelopment)
                                SearchScores.Add((iteration, currentValue));

                            lastChangeAcceptedOnIt = iteration;
                        }

                    }
                }
                else
                {
                    OPNotPossible[OS.LastOperator] += 1;
                    amtNotDone += 1;
                }
                if (iteration % Config.IterationsPerAlphaChange == 0 && iteration != 0)
                {
                    //Update temperature using Alpha
                    Temperature *= Config.Alpha;

                    //Update caches
                    routes.ForEach(x => x.ResetCache());
                    currentValue = Solver.CalcTotalDistance(routes, removed, this);
                    bestSolValue = Solver.CalcTotalDistance(BestSolution, BestSolutionRemoved, this);
                }
                //Check if the restart conditions are met
                if (iteration - restartPreventionIteration > Config.NumIterationsOfNoChangeBeforeRestarting && Temperature < Config.RestartTemperatureBound && numRestarts < Config.NumRestarts) //&& iteration - lastChangeExceptedOnIt > 1000
                {
                    numRestarts += 1;
                    restartPreventionIteration = iteration;
                    //Restart
                    var oldTemp = Temperature;
                    Temperature += Config.InitialTemperature / 3;
                    routes.ForEach(x => x.ResetCache());
                    routes = BestSolution.ConvertAll(i => i.CreateDeepCopy());
                    viableRoutes = Enumerable.Range(0, routes.Count).Where(i => routes[i].route.Count > 2).ToList();
                    removed = BestSolutionRemoved.ConvertAll(x => x);

                    //Reset the scores
                    currentValue = Solver.CalcTotalDistance(routes, removed, this);
                    bestSolValue = Solver.CalcTotalDistance(BestSolution, new List<Customer>(), this);
                    Console.WriteLine($"{id}:Best solution changed to long ago a T: {oldTemp}. Restarting from best solution with T: {Temperature}");
                }
                else if (Temperature < 0.02 && numRestarts >= Config.NumRestarts)
                    break;
                if (printTimer.ElapsedMilliseconds > 3 * 1000)
                {
                    var elapsed = printTimer.ElapsedMilliseconds;
                    printTimer.Restart();
                    double itsPerSecond = (iteration - previousUpdateIteration) / ((double)elapsed / 1000);
                    previousUpdateIteration = iteration;
                    int cnt = routes.Count(x => x.route.Count > 2);
                    Console.WriteLine($"{id}: T: {Temperature.ToString("0.000")}, S: {Solver.CalcTotalDistance(routes, removed, this).ToString("0.000")}, TS: {currentValue.ToString("0.000")}, N: {cnt}, IT: {iteration}, LA {iteration - lastChangeAcceptedOnIt}, B: {bestSolValue},{Solver.CalcTotalDistance(BestSolution, BestSolutionRemoved, this)}, BI: {bestImprovedIteration}, IT/s: {itsPerSecond.ToString("0.00")}/s");

                }
            }

            //Saving the columns of the best solutions
            foreach (Route route in BestSolution)
            {
                if (IsValidRoute(route))
                    Columns.Add(new RouteStore(route.CreateIdList(), route.Score));

            }
            if (id == 0 && Config.SaveScoreDevelopment)
                BestSolutionScores.Add((iteration, bestSolValue));

            Console.WriteLine($"DONE {id}: {name}, Score: {Solver.CalcTotalDistance(BestSolution, removed, this)}, Columns: {Columns.Count}. Completed {iteration} iterations in {Math.Round((double)timer.ElapsedMilliseconds / 1000, 3)}s");

            if (Config.PrintExtendedInfo)
            {
                lock (ConsoleWriterLock)
                {
                    Console.WriteLine($"  {id}: Total: {amtNotDone + amtImp + amtWorse}, improvements: {amtImp}, worse: {amtWorse}, not done: {amtNotDone}");
                    Console.WriteLine($" {id}: OPERATOR PERFORMANCE OVERVIEW");
                    foreach (string op in OS.OperatorList)
                    {
                        string spaces = "";
                        if (op.Length < 20)
                            spaces = new string(' ', 20 - op.Length);
                        Console.WriteLine($"\t{op}:{spaces} ICnt: {OPImprovementCount[op]}, AI: {(OPImprovementTotal[op] / OPImprovementCount[op]).ToString("0.00000")}, B: {(OPBestImprovement[op]).ToString("0.00000")}, WCnt: {OPAcceptedWorseCount[op]}, AW: {(OPAcceptedWorseTotal[op] / OPAcceptedWorseCount[op]).ToString("0.00000")}, NP : {OPNotPossible[op]} ");
                    }
                }

            }
            if (id == 0 && Config.SaveScoreDevelopment)
            {
                Console.WriteLine("Saving scores");
                System.IO.File.WriteAllLines("SearchScores.txt", SearchScores.ConvertAll(x => $"{x.Item1};{x.Item2}"));
                File.WriteAllLines("BestScores.txt", BestSolutionScores.ConvertAll(x => $"{x.Item1};{x.Item2}"));
                Console.WriteLine("Done saving scores");
            }
            return (Columns, BestSolution, Solver.CalcTotalDistance(BestSolution, removed, this));
        }

        public (HashSet<RouteStore>, List<Route>, double) LocalSearchInstanceKDE(int id, string name, int numVehicles, double vehicleCapacity, List<Customer> customers, double[,,,] observations, int numIterations = 3000000, int timeLimit = 30000, bool checkInitialSolution = false, int repetition = 0)
        {
            Console.WriteLine("Starting local search");
            //Console.WriteLine($"Total used capacity is {customers.Sum(x => x.Demand)}");

            EasyShuffle(customers, random);

            List<Route> routes = new List<Route>();
            List<Customer> removed = new List<Customer>();

            //Generate routes
            for (int i = 0; i < numVehicles; i++)
                routes.Add(new Route(customers[0], vehicleCapacity, observations, seed: random.Next(), this));

            CreateSmartInitialSolutionKDE(routes, customers, removed, observations);
            //CreateSmartInitialSolution(routes, customers, removed);
            foreach (Route route in routes)
                {
                    Console.WriteLine($"{route}");
                }

            if (checkInitialSolution)
                //Validate intial solution
                foreach (Route route in routes)
                    if (!route.CheckRouteValidityKDE())
                    {
                        Console.WriteLine("FAIL: Initial solution not valid");
                        throw new Exception();
                    }

            Console.WriteLine($"Finished making initial solution with KDE value {Solver.CalcTotalDistanceKDE(routes, removed, this).ToString("0.00")}");

            int amtImp = 0, amtWorse = 0, amtNotDone = 0;
            //double temp = initialTemp;
            double totalP = 0;
            double countP = 0;
            
            Stopwatch printTimer = new Stopwatch();
            printTimer.Start();

            //Used to keep track of the routes which contain customers. This speeds up execution as this only needs to be updated when a change is applied
            List<int> viableRoutes = Enumerable.Range(0, routes.Count).Where(i => routes[i].route.Count > 2).ToList();

            int lastChangeAcceptedOnIt = 0;

            //All saved columns
            HashSet<RouteStore> Columns = new HashSet<RouteStore>();

            //The best found solution so far
            List<Route> BestSolution = routes.ConvertAll(i => i.CreateDeepCopyKDE());
            List<Customer> BestSolutionRemoved = removed.ConvertAll(x => x);

            List<(int, double)> SearchScores = new List<(int, double)>();
            List<(int, double)> BestSolutionScores = new List<(int, double)>();

            //Dictionaries used for tracking the performance of the different operators. These are only used if the option is enabled in the configuration
            Dictionary<string, int> OPImprovementCount = new Dictionary<string, int>();
            Dictionary<string, double> OPImprovementTotal = new Dictionary<string, double>();
            Dictionary<string, double> OPBestImprovement = new Dictionary<string, double>();

            Dictionary<string, int> OPAcceptedWorseCount = new Dictionary<string, int>();
            Dictionary<string, double> OPAcceptedWorseTotal = new Dictionary<string, double>();
            Dictionary<string, int> OPNotPossible = new Dictionary<string, int>();

            foreach (string op in OS.OperatorList)
            {
                OPImprovementCount[op] = 0;
                OPImprovementTotal[op] = 0;
                OPBestImprovement[op] = 0;
                OPAcceptedWorseTotal[op] = 0;
                OPAcceptedWorseCount[op] = 0;
                OPNotPossible[op] = 0;
                //Console.WriteLine($"Our operators include {op}");
            }
            double bestSolValue = Solver.CalcTotalDistanceKDE(BestSolution, removed, this);
            double currentValue = bestSolValue;
            int bestImprovedIteration = 0;
            int restartPreventionIteration = 0;
            int numRestarts = 0;
            int previousUpdateIteration = 0;
            int iteration = 0;
            Stopwatch timer = new Stopwatch();
            timer.Start();
            //numIterations = 1000;
            //int RootnrObservations = (int)Math.Sqrt(observations.GetLength(3));
            for (; iteration < numIterations && timer.ElapsedMilliseconds <= timeLimit; iteration++)
            { 
                double imp = 0;
                Action? act = null;
                var nextOperator = OS.Next();
                //Solver.PrintRoutes(routes);
                (imp, act) = nextOperator(routes, viableRoutes, random, removed, this);
                //foreach (Route route in routes)
                 //   route.used_capacity = 0;

                if (act != null)
                {
                    if (currentValue > double.MaxValue - 1000000)
                        Console.WriteLine("OVERFLOW");
                    
                    if (imp > currentValue)
                        Console.WriteLine($"Impossible improvement {Math.Round(imp,2)} to {Math.Round(currentValue,2)}from {OS.LastOperator}");

                    //Accept all improvements
                    if (imp > 0)
                    {
                        OPImprovementCount[OS.LastOperator] += 1;
                        OPImprovementTotal[OS.LastOperator] += imp;

                        if (imp > OPBestImprovement[OS.LastOperator])
                            OPBestImprovement[OS.LastOperator] = imp;

                        //Apply the changes of the neighbor
                        RunAndCheckOperatorKDE(id, routes, removed, imp, act);

                        //Update viable routes cache
                        viableRoutes = Enumerable.Range(0, routes.Count).Where(i => routes[i].route.Count > 2).ToList();

                        currentValue -= imp;
                        if (id == 0 && Config.SaveScoreDevelopment)
                            SearchScores.Add((iteration, currentValue));

                        //Check wheter it improves the best known solution
                        if (Math.Round(currentValue, 6) < Math.Round(bestSolValue, 6) && removed.Count == 0 && IsValidSolution(routes, removed))
                        {
                            //New best solution found
                            bestSolValue = currentValue;
                            bestImprovedIteration = iteration;
                            restartPreventionIteration = iteration;
                            //Create a copy
                            BestSolution = routes.ConvertAll(i => i.CreateDeepCopyKDE());

                            //Score development can be saved if enabled
                            if (id == 0 && Config.SaveScoreDevelopment)
                                BestSolutionScores.Add((iteration, bestSolValue));

                            //Now we know all customers are used
                            BestSolutionRemoved = new List<Customer>();
                            foreach (Route route in routes)
                            {
                                if (IsValidRoute(route))
                                    Columns.Add(new RouteStore(route.CreateIdList(), route.Score));
                            }
                        }

                        //Add columns to the column store
                        if (Config.SaveColumnsAfterAllImprovements && Temperature < Config.InitialTemperature * Config.SaveColumnThreshold)
                            foreach (Route route in routes)
                            {
                                if (IsValidRoute(route))
                                    Columns.Add(new RouteStore(route.CreateIdList(), route.Score));
                            }
                        amtImp += 1;
                        lastChangeAcceptedOnIt = iteration;
                    }
                    else
                    {
                        // imp !> 0, thus acceptP is indeed between 0 and 1
                        double acceptP = Math.Exp(imp / Temperature);
                        totalP += acceptP;
                        countP += 1;
                        if (random.NextDouble() <= acceptP)
                        {
                            //Worse solution accepted
                            amtWorse += 1;

                            OPAcceptedWorseCount[OS.LastOperator] += 1;
                            OPAcceptedWorseTotal[OS.LastOperator] += imp; 

                            RunAndCheckOperatorKDE(id, routes, removed, imp, act);

                            if (Config.SaveColumnsAfterWorse && Temperature < Config.InitialTemperature * Config.SaveColumnThreshold)
                                foreach (Route route in routes)
                                {
                                    if (IsValidRoute(route))
                                        Columns.Add(new RouteStore(route.CreateIdList(), route.CalcObjectiveKDE(finalCalculation:true)));
                                }
                            viableRoutes = Enumerable.Range(0, routes.Count).Where(i => routes[i].route.Count > 2).ToList();
                            currentValue -= imp;
                            if (id == 0 && Config.SaveScoreDevelopment)
                                SearchScores.Add((iteration, currentValue));

                            lastChangeAcceptedOnIt = iteration;
                        }
                    }
                }
                else
                {
                    OPNotPossible[OS.LastOperator] += 1;
                    amtNotDone += 1;
                }
                if (iteration % Config.IterationsPerAlphaChange == 0 && iteration != 0)
                {
                    //Update temperature using Alpha
                    Temperature *= Config.Alpha;

                    //Update caches
                    routes.ForEach(x => x.ResetCache());
                    currentValue = Solver.CalcTotalDistanceKDE(routes, removed, this);
                    bestSolValue = Solver.CalcTotalDistanceKDE(BestSolution, BestSolutionRemoved, this);
                }
                //Check if the restart conditions are met
                if (iteration - restartPreventionIteration > Config.NumIterationsOfNoChangeBeforeRestarting && Temperature < Config.RestartTemperatureBound && numRestarts < Config.NumRestarts) //&& iteration - lastChangeExceptedOnIt > 1000
                {
                    numRestarts += 1;
                    restartPreventionIteration = iteration;
                    //Restart
                    var oldTemp = Temperature;
                    Temperature += Config.InitialTemperature / 3;
                    routes.ForEach(x => x.ResetCache());
                    routes = BestSolution.ConvertAll(i => i.CreateDeepCopyKDE());
                    viableRoutes = Enumerable.Range(0, routes.Count).Where(i => routes[i].route.Count > 2).ToList();
                    removed = BestSolutionRemoved.ConvertAll(x => x);

                    //Reset the scores
                    currentValue = Solver.CalcTotalDistanceKDE(routes, removed, this);
                    bestSolValue = Solver.CalcTotalDistanceKDE(BestSolution, new List<Customer>(), this);
                    Console.WriteLine($"{id}:Best solution changed too long ago a T: {oldTemp}. Restarting from best solution with T: {Temperature}");
                }
                else if (Temperature < 0.02 && numRestarts >= Config.NumRestarts)
                    break;
                if (printTimer.ElapsedMilliseconds > 3 * 1000)
                {
                    var elapsed = printTimer.ElapsedMilliseconds;
                    printTimer.Restart();
                    double itsPerSecond = (iteration - previousUpdateIteration) / ((double)elapsed / 1000);
                    previousUpdateIteration = iteration;
                    int cnt = routes.Count(x => x.route.Count > 2);
                    Console.WriteLine($"{id}: T: {Temperature.ToString("0.000")}, S: {Solver.CalcTotalDistanceKDE(routes, removed, this).ToString("0.00")}, TS: {currentValue.ToString("0.00")}, N: {cnt}, IT: {iteration}, LA {iteration - lastChangeAcceptedOnIt}, B: {bestSolValue.ToString("0.00")},{Solver.CalcTotalDistanceKDE(BestSolution, BestSolutionRemoved, this).ToString("0.00")}, BI: {bestImprovedIteration}, IT/s: {itsPerSecond.ToString("0.00")}/s");
                }
            }

            //Saving the columns of the best solutions
            foreach (Route route in BestSolution)
            {
                if (IsValidRoute(route))
                    Columns.Add(new RouteStore(route.CreateIdList(), route.CalcObjectiveKDE(finalCalculation:true)));
            }
            if (id == 0 && Config.SaveScoreDevelopment)
                {BestSolutionScores.Add((iteration, bestSolValue));
                SearchScores.Add((iteration, bestSolValue));}

            Console.WriteLine($"DONE {id}: {name}, Score: {Solver.CalcTotalDistanceKDE(BestSolution, removed, this)}, Columns: {Columns.Count}. Completed {iteration} iterations in {Math.Round((double)timer.ElapsedMilliseconds / 1000, 3)}s");
            if (Config.PrintExtendedInfo)
            {
                lock (ConsoleWriterLock)
                {
                    Console.WriteLine($"  {id}: Total: {amtNotDone + amtImp + amtWorse}, improvements: {amtImp}, worse: {amtWorse}, not done: {amtNotDone}");
                    Console.WriteLine($" {id}: OPERATOR PERFORMANCE OVERVIEW");
                    foreach (string op in OS.OperatorList)
                    {
                        string spaces = "";
                        if (op.Length < 20)
                            spaces = new string(' ', 20 - op.Length);
                        Console.WriteLine($"\t{op}:{spaces} ICnt: {OPImprovementCount[op]}, AI: {(OPImprovementTotal[op] / OPImprovementCount[op]).ToString("0.00000")}, B: {(OPBestImprovement[op]).ToString("0.00000")}, WCnt: {OPAcceptedWorseCount[op]}, AW: {(OPAcceptedWorseTotal[op] / OPAcceptedWorseCount[op]).ToString("0.00000")}, NP : {OPNotPossible[op]} ");
                    }
                }

            }
            {
                Console.WriteLine("Saving scores");
                System.IO.File.WriteAllLines("SearchScores.txt", SearchScores.ConvertAll(x => $"{x.Item1};{x.Item2}"));
                var timeNow = DateTime.Now;
                string directoryName = $"ScoreDevs_{timeNow}";
                string directoryPath = Path.Combine(Environment.CurrentDirectory, directoryName);
                if (!Directory.Exists(directoryPath))
                    Directory.CreateDirectory(directoryPath);
                    
                string fileName = $"BestScores_{Config.txtFileName}_{Config.WaitingPenalty}_{Config.LatenessPenalty}_{Config.InitialTemperature}_{Config.IterationsPerAlphaChange}.txt";
                File.WriteAllLines(Path.Combine(directoryPath, fileName), BestSolutionScores.ConvertAll(x => $"{x.Item1};{x.Item2}"));
                Console.WriteLine("Done saving scores");
            }
            return (Columns, BestSolution, Solver.CalcTotalDistanceKDE(BestSolution, removed, this));
        }
    }


}
