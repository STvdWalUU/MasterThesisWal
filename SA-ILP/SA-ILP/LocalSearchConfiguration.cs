using MathNet.Numerics.Distributions;
using System;
using System.Collections.Generic;
using System.Dynamic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SA_ILP
{
    //Configurations struct used for setting up the algorithm
    public struct LocalSearchConfiguration
    {
        public bool useKDE {get; set;}
        public bool useKernel {get; set;}
        public string kernelChoice {get; set;}
        public string bandwidth {get; set;}
        public bool useTardiness {get;set;}
        public double WaitingPenalty {get; set;}
        public double LatenessPenalty {get; set;}
        public bool useWeights {get; set;}
        public string txtFileName {get;set;}
        public double InitialTemperature { get; set; }
        public bool AllowLateArrivalDuringSearch { get; set; }
        public bool AllowEarlyArrivalDuringSearch { get; set; }
        public bool AllowLateArrival { get; set; }
        public bool AllowDeterministicEarlyArrival { get; set; }
        public double BaseRemovedCustomerPenalty { get; set; }
        public double BaseRemovedCustomerPenaltyPow { get; set; }
        public double BaseEarlyArrivalPenalty { get; set; }
        public double BaseLateArrivalPenalty { get; set; }
        public double Alpha { get; set; }
        public bool SaveColumnsAfterAllImprovements { get; set; }
        public bool PenalizeDeterministicEarlyArrival { get; set; }
        public bool PenalizeLateArrival { get; set; }
        public bool AdjustDeterministicEarlyArrivalToTWStart { get; set; }
        //Only used in simmulations so far
        public bool AdjustEarlyArrivalToTWStart { get; set; }
        public bool CheckOperatorScores { get; set; }
        public bool SaveRoutesBeforeOperator { get; set; }
        public bool SaveColumnsAfterWorse { get; set; }
        public double SaveColumnThreshold { get; set; }
        public bool PrintExtendedInfo { get; set; }
        public bool SaveScoreDevelopment { get; set; }
        public double ExpectedEarlinessPenalty { get; set; }
        public double ExpectedLatenessPenalty { get; set; }
        public bool UseMeanOfDistributionForTravelTime { get; set; }
        public bool ScaleEarlinessPenaltyWithTemperature { get; set; }
        public bool ScaleLatenessPenaltyWithTemperature { get; set; }
        public bool UseMeanOfDistributionForScore { get; set; }
        public bool IgnoreWaitingDuringDistributionAddition { get; set; }
        public double WindSpeed { get; set; }
        public double[] WindDirection { get; set; }
        public int NumRestarts { get; set; }
        public int IterationsPerAlphaChange { get; set; }
        public int NumIterationsOfNoChangeBeforeRestarting { get; set; }
        public double RestartTemperatureBound { get; set; }
        public IContinuousDistribution DefaultDistribution { get; set; }
        public bool AllowEarlyArrivalInSimulation { get; set; }
        internal List<(Operator, Double, String, int)> Operators { get; set; }
        public double RemovedCustomerTemperaturePow { get; set; }
        public bool CutProbabilityDistributionAt0 { get; set; }
        public bool UseStochasticFunctions { get; set; }
        public bool UseNormalMaximizationForAllSteps { get; set; }
        public override string ToString()
        {
            LocalSearchConfiguration obj = this;
            return GetType().GetProperties()

        .Select(info =>
        {

            return (info.Name, Value: info.GetValue(obj, null) ?? "(null)");
        })
        .Aggregate(
            new StringBuilder(),
            (sb, pair) => sb.AppendLine($"{pair.Name}: {pair.Value}"),
            sb => sb.ToString());
        }
    }
    public static class LocalSearchConfigs
    {
        public static LocalSearchConfiguration VRPLTTFinal => new LocalSearchConfiguration
        {
            InitialTemperature = 1,

            //THESE OPTIONS ARE NOT YET CORRECTLY IMPLEMENTED
            AllowEarlyArrivalDuringSearch = true,
            AllowLateArrivalDuringSearch = true,

            useKDE = false,
            useTardiness = false,
            WaitingPenalty = 1,
            LatenessPenalty = 5,

            AllowDeterministicEarlyArrival = true,
            AllowEarlyArrivalInSimulation = true,
            AllowLateArrival = false,
            BaseEarlyArrivalPenalty = 2,
            BaseLateArrivalPenalty = 2,

            BaseRemovedCustomerPenalty = 0.1,
            BaseRemovedCustomerPenaltyPow = 2,
            RemovedCustomerTemperaturePow = 2,
            Alpha = 0.99,

            SaveColumnsAfterAllImprovements = true,
            SaveColumnsAfterWorse = true,
            SaveColumnThreshold = 0.2,

            PenalizeDeterministicEarlyArrival = true,
            PenalizeLateArrival = true,
            AdjustDeterministicEarlyArrivalToTWStart = true,
            AdjustEarlyArrivalToTWStart = true,
            CheckOperatorScores = false,
            SaveRoutesBeforeOperator = false,

            PrintExtendedInfo = false,
            SaveScoreDevelopment = false,
            ExpectedEarlinessPenalty = 0,
            ExpectedLatenessPenalty = 0,
            UseMeanOfDistributionForTravelTime = false,
            ScaleEarlinessPenaltyWithTemperature = true,
            ScaleLatenessPenaltyWithTemperature = true,
            IgnoreWaitingDuringDistributionAddition = true,
            IterationsPerAlphaChange = 10000,
            NumIterationsOfNoChangeBeforeRestarting = 600000,
            RestartTemperatureBound = 0.02,
            NumRestarts = 7,
            WindSpeed = 0, //6.75,
            WindDirection = [0,-1], //[1/2, Math.Sqrt(3)/2],
            //WindDirection = [0,-1], // north // [1,0] is west wind, [0,1] is south wind, i.e. [1,0] wind blows from west to east.

            Operators = new List<(Operator, double, string, int)>()
            {
                ((routes, viableRoutes, random, removed, temp) => Operators.MoveRandomCustomerToRandomCustomer(routes, viableRoutes, random), 1, "move", 1),//repeated 1 time
                ((x, y, z, w, v) => Operators.SwapRandomCustomers(x, y, z), 1, "swap", 4), //repeated 4 times
                //((x, y, z, w, v) => Operators.SwapInsideRoute(x, y, z), 1, "swap_inside_route", 4), //repeated 4 times
                ((x, y, z, w, v) => Operators.ReverseOperator(x, y, z), 1, "reverse",1), //repeated 1 time
                ((x, y, z, w, v) => Operators.ScrambleSubRoute(x, y, z), 1, "scramble",1), //Repeated 1 time
                //((x, y, z, w, v) =>Operators.RemoveRandomCustomer(x, y, z, w, v), 1, "remove", 1), //repeated 1 time
                //((x, y, z, w, v) =>Operators.AddRandomRemovedCustomer(x, y, z, w, v), 1, "add", 1), //repeated 1 time
                ((x, y, z, w, v) => Operators.GreedilyMoveRandomCustomer(x, y, z), 0.1, "move_to_best",4), //repeated 4 time
                ((x, y, z, w, v) => Operators.MoveRandomCustomerToRandomRoute(x, y, z), 1, "move_to_random_route", 1), //repeated 1 times
                //((x, y, z, w, v) => Operators.SwapRandomTails(x, y, z), 1, "swap_tails",1), //Repeated 1 time
            }
        };
        public static LocalSearchConfiguration VRPLTTOriginal => new LocalSearchConfiguration
        {
            InitialTemperature = 1,

            //THESE OPTIONS ARE NOT YET CORRECTLY IMPLEMENTED
            AllowEarlyArrivalDuringSearch = true,
            AllowLateArrivalDuringSearch = true,


            AllowDeterministicEarlyArrival = true,
            AllowEarlyArrivalInSimulation = true,
            AllowLateArrival = false,
            BaseEarlyArrivalPenalty = 2,
            BaseLateArrivalPenalty = 2,

            BaseRemovedCustomerPenalty = 0.1,
            BaseRemovedCustomerPenaltyPow = 2,
            RemovedCustomerTemperaturePow = 2,
            Alpha = 0.99,

            SaveColumnsAfterAllImprovements = true,
            SaveColumnsAfterWorse = true,
            SaveColumnThreshold = 0.2,


            PenalizeDeterministicEarlyArrival = false,
            PenalizeLateArrival = true,
            AdjustDeterministicEarlyArrivalToTWStart = true,
            AdjustEarlyArrivalToTWStart = true,
            CheckOperatorScores = false,
            SaveRoutesBeforeOperator = false,

            PrintExtendedInfo = false,
            SaveScoreDevelopment = false,
            ExpectedEarlinessPenalty = 0,
            ExpectedLatenessPenalty = 0,
            UseMeanOfDistributionForTravelTime = false,
            ScaleEarlinessPenaltyWithTemperature = true,
            ScaleLatenessPenaltyWithTemperature = true,
            IgnoreWaitingDuringDistributionAddition = true,
            IterationsPerAlphaChange = 10000,
            NumIterationsOfNoChangeBeforeRestarting = 600000,
            RestartTemperatureBound = 0.02,
            NumRestarts = 7,
            WindSpeed = 0,

            Operators = new List<(Operator, double, string, int)>()
            {
                ((x, y, z, w, v) =>Operators.AddRandomRemovedCustomer(x, y, z, w, v), 1, "add", 1), //repeated 1 time
                ((x, y, z, w, v) =>Operators.RemoveRandomCustomer(x, y, z, w, v), 1, "remove", 1), //repeated 1 time
                ((routes, viableRoutes, random, removed, temp) => Operators.MoveRandomCustomerToRandomCustomer(routes, viableRoutes, random), 1, "move", 1),//repeated 1 time
               ((x, y, z, w, v) => Operators.GreedilyMoveRandomCustomer(x, y, z), 0.1, "move_to_best",1), //repeated 1 time
                ((x, y, z, w, v) => Operators.MoveRandomCustomerToRandomRoute(x, y, z), 1, "move_to_random_route", 4), //repeated 4 times
                ((x, y, z, w, v) => Operators.SwapRandomCustomers(x, y, z), 1, "swap", 4), //repeated 4 times
                ((x, y, z, w, v) => Operators.SwapInsideRoute(x, y, z), 1, "swap_inside_route", 4), //repeated 4 times
                ((x, y, z, w, v) => Operators.ReverseOperator(x, y, z), 1, "reverse",1), //repeated 1 time
                ((x, y, z, w, v) => Operators.ScrambleSubRoute(x, y, z), 1, "scramble",1), //Repeated 1 time
                ((x, y, z, w, v) => Operators.SwapRandomTails(x, y, z), 1, "swap_tails",1), //Repeated 1 time
            }

        };
        public static LocalSearchConfiguration VRPLTTWithoutWaiting
        {
            get
            {
                var config = VRPLTTFinal;
                config.AllowDeterministicEarlyArrival = false;
                config.AllowEarlyArrivalInSimulation = false;
                config.PenalizeDeterministicEarlyArrival = true;
                return config;
            }
        }
        public static LocalSearchConfiguration VRPLTTNoGreedyOperators
        {
            get
            {
                var config = VRPLTTFinal;
                config.Operators = new List<(Operator, double, string, int)>()
            {
                ((x, y, z, w, v) =>Operators.AddRandomRemovedCustomer(x, y, z, w, v), 1, "add", 1), //repeated 1 time
                ((x, y, z, w, v) =>Operators.RemoveRandomCustomer(x, y, z, w, v), 1, "remove", 1), //repeated 1 time
                ((routes, viableRoutes, random, removed, temp) => Operators.MoveRandomCustomerToRandomCustomer(routes, viableRoutes, random), 1, "move", 1),//repeated 1 time
               //((x, y, z, w, v) => Operators.GreedilyMoveRandomCustomer(x, y, z), 0.1, "move_to_best",1), //repeated 1 time
                //((x, y, z, w, v) => Operators.MoveRandomCustomerToRandomRoute(x, y, z), 1, "move_to_random_route", 4), //repeated 4 times
                ((x, y, z, w, v) => Operators.SwapRandomCustomers(x, y, z), 1, "swap", 4), //repeated 4 times
                ((x, y, z, w, v) => Operators.SwapInsideRoute(x, y, z), 1, "swap_inside_route", 4), //repeated 4 times
                ((x, y, z, w, v) => Operators.ReverseOperator(x, y, z), 1, "reverse",1), //repeated 1 time
                ((x, y, z, w, v) => Operators.ScrambleSubRoute(x, y, z), 1, "scramble",1), //Repeated 1 time
                ((x, y, z, w, v) => Operators.SwapRandomTails(x, y, z), 1, "swap_tails",1), //Repeated 1 time
            };
            config.WindSpeed = 3.75;

                return config;
            }
        }
        /// <summary>
        /// The basic configuration for using the KDE method to solve the VRPLTT problem.
        /// </summary>
        public static LocalSearchConfiguration VRPLTTFinalKDE => new LocalSearchConfiguration
        {
            InitialTemperature = 10,
            IterationsPerAlphaChange = 10000,
            NumIterationsOfNoChangeBeforeRestarting = 600000,
            RestartTemperatureBound = 0.02,
            NumRestarts = 7,
            Alpha = 0.99,
            useWeights = false,

            //THESE OPTIONS ARE NOT YET CORRECTLY IMPLEMENTED
            AllowEarlyArrivalDuringSearch = true,
            AllowLateArrivalDuringSearch = true,

            useKDE = true,
            useKernel = true,
            kernelChoice = "Epanechnikov",
            bandwidth = "AMISE",
            useTardiness = false,
            WaitingPenalty = 1,
            LatenessPenalty = 5,

            WindSpeed = 0,
            PenalizeDeterministicEarlyArrival = true,
            AdjustEarlyArrivalToTWStart = true,
            AdjustDeterministicEarlyArrivalToTWStart = true,
            BaseEarlyArrivalPenalty = 2,
            BaseLateArrivalPenalty = 2,

            CheckOperatorScores = true,

            AllowDeterministicEarlyArrival = true,
            AllowEarlyArrivalInSimulation = true,
            AllowLateArrival = false,

            BaseRemovedCustomerPenalty = 0.1,
            BaseRemovedCustomerPenaltyPow = 2,
            RemovedCustomerTemperaturePow = 2,

            SaveColumnsAfterAllImprovements = true,
            SaveColumnsAfterWorse = true,
            SaveColumnThreshold = 0.2,

            PenalizeLateArrival = true,
            
            SaveRoutesBeforeOperator = true,

            PrintExtendedInfo = true,
            SaveScoreDevelopment = true,
            ScaleEarlinessPenaltyWithTemperature = true,
            ScaleLatenessPenaltyWithTemperature = true,
            
            ExpectedEarlinessPenalty = 0,
            ExpectedLatenessPenalty = 0,
            UseMeanOfDistributionForTravelTime = false,
            IgnoreWaitingDuringDistributionAddition = true,

            Operators = new List<(Operator, double, string, int)>()
            {
                ((routes, viableRoutes, random, removed, temp) => Operators.MoveRandomCustomerToRandomCustomer(routes, viableRoutes, random), 1, "move", 1),//repeated 1 time
                ((x, y, z, w, v) => Operators.SwapRandomCustomers(x, y, z), 1, "swap", 4), //repeated 4 times
                //((x, y, z, w, v) => Operators.SwapInsideRoute(x, y, z), 1, "swap_inside_route", 4), //repeated 4 times
                ((x, y, z, w, v) => Operators.ReverseOperator(x, y, z), 1, "reverse",1), //repeated 1 time
                ((x, y, z, w, v) => Operators.ScrambleSubRoute(x, y, z), 1, "scramble",1), //Repeated 1 time
                //((x, y, z, w, v) =>Operators.RemoveRandomCustomer(x, y, z, w, v), 1, "remove", 1), //repeated 1 time
                //((x, y, z, w, v) =>Operators.AddRandomRemovedCustomer(x, y, z, w, v), 1, "add", 1), //repeated 1 time
                ((x, y, z, w, v) => Operators.GreedilyMoveRandomCustomer(x, y, z), 0.1, "move_to_best",4), //repeated 4 time
                ((x, y, z, w, v) => Operators.MoveRandomCustomerToRandomRoute(x, y, z), 1, "move_to_random_route", 1), //repeated 1 times
                //((x, y, z, w, v) => Operators.SwapRandomTails(x, y, z), 1, "swap_tails",1), //Repeated 1 time
            }
        };
        public static LocalSearchConfiguration VRPLTTKDE
        {
            get
            {
                var config = VRPLTTFinal;
                config.Operators = new List<(Operator, double, string, int)>()
            {    
                ((routes, viableRoutes, random, removed, temp) => Operators.MoveRandomCustomerToRandomCustomer(routes, viableRoutes, random), 1, "move", 1),//repeated 1 time
                ((x, y, z, w, v) => Operators.SwapRandomCustomers(x, y, z), 1, "swap", 4), //repeated 4 times
                //((x, y, z, w, v) => Operators.SwapInsideRoute(x, y, z), 1, "swap_inside_route", 4), //repeated 4 times
                ((x, y, z, w, v) => Operators.ReverseOperator(x, y, z), 1, "reverse",1), //repeated 1 time
                ((x, y, z, w, v) => Operators.ScrambleSubRoute(x, y, z), 1, "scramble",1), //Repeated 1 time
                //((x, y, z, w, v) =>Operators.RemoveRandomCustomer(x, y, z, w, v), 1, "remove", 1), //repeated 1 time
                //((x, y, z, w, v) =>Operators.AddRandomRemovedCustomer(x, y, z, w, v), 1, "add", 1), //repeated 1 time
                ((x, y, z, w, v) => Operators.GreedilyMoveRandomCustomer(x, y, z), 0.1, "move_to_best",4), //repeated 4 time
                ((x, y, z, w, v) => Operators.MoveRandomCustomerToRandomRoute(x, y, z), 1, "move_to_random_route", 1), //repeated 1 times
                //((x, y, z, w, v) => Operators.SwapRandomTails(x, y, z), 1, "swap_tails",1), //Repeated 1 time
            };
            
            config.useKDE = true;
            config.useKernel = true;
            //config.kernelChoice = "None";
            //config.bandwidth = "AMISE";
            config.useTardiness = false;
            config.WaitingPenalty = 1.1;
            config.LatenessPenalty = 10;

            config.NumRestarts = 7;

            config.PenalizeDeterministicEarlyArrival = true;
            config.PenalizeLateArrival = true;
            config.AllowEarlyArrivalInSimulation = true;
            config.CheckOperatorScores = false;
            config.ScaleLatenessPenaltyWithTemperature = false;
            config.ScaleEarlinessPenaltyWithTemperature = false;

            return config;
            }
        }
        public static LocalSearchConfiguration VRPLTTNoExtraInternalSwap
        {
            get
            {
                var config = VRPLTTFinal;
                config.Operators = new List<(Operator, double, string, int)>()
            {
                ((x, y, z, w, v) =>Operators.AddRandomRemovedCustomer(x, y, z, w, v), 1, "add", 1), //repeated 1 time
                ((x, y, z, w, v) =>Operators.RemoveRandomCustomer(x, y, z, w, v), 1, "remove", 1), //repeated 1 time
                ((routes, viableRoutes, random, removed, temp) => Operators.MoveRandomCustomerToRandomCustomer(routes, viableRoutes, random), 1, "move", 1),//repeated 1 time
               ((x, y, z, w, v) => Operators.GreedilyMoveRandomCustomer(x, y, z), 0.1, "move_to_best",1), //repeated 1 time
                ((x, y, z, w, v) => Operators.MoveRandomCustomerToRandomRoute(x, y, z), 1, "move_to_random_route", 4), //repeated 4 times
                ((x, y, z, w, v) => Operators.SwapRandomCustomers(x, y, z), 1, "swap", 4), //repeated 4 times
                //((x, y, z, w, v) => Operators.SwapInsideRoute(x, y, z), 1, "swap_inside_route", 4), //repeated 4 times
                ((x, y, z, w, v) => Operators.ReverseOperator(x, y, z), 1, "reverse",1), //repeated 1 time
                ((x, y, z, w, v) => Operators.ScrambleSubRoute(x, y, z), 1, "scramble",1), //Repeated 1 time
                ((x, y, z, w, v) => Operators.SwapRandomTails(x, y, z), 1, "swap_tails",1), //Repeated 1 time
            };

                return config;
            }
        }
        public static LocalSearchConfiguration VRPLTTLinearHigherAddRemovePenaltyLinearTempPow
        {
            get
            {
                var config = LocalSearchConfigs.VRPLTTFinal;
                config.BaseRemovedCustomerPenalty = 8;
                config.BaseRemovedCustomerPenaltyPow = 1;
                config.RemovedCustomerTemperaturePow = 1;
                return config;
            }
        }
        public static LocalSearchConfiguration VRPLTTLinearHigherAddRemovePenalty
        {
            get
            {
                var config = LocalSearchConfigs.VRPLTTFinal;
                config.BaseRemovedCustomerPenalty = 4;
                config.BaseRemovedCustomerPenaltyPow = 1;
                return config;
            }
        }
        public static LocalSearchConfiguration VRPSLTTWithoutWaiting
        {
            get
            {
                var config = VRPLTTFinal;

                //Enable the usage of the stochastic implementation
                config.UseStochasticFunctions = true;

                config.ExpectedEarlinessPenalty = 10;
                config.ExpectedLatenessPenalty = 10;

                config.AllowDeterministicEarlyArrival = true;
                config.PenalizeDeterministicEarlyArrival = false;
                config.AllowEarlyArrivalInSimulation = false;

                config.CheckOperatorScores = false;
                config.SaveRoutesBeforeOperator = false;
                config.ScaleEarlinessPenaltyWithTemperature = true;
                config.ScaleLatenessPenaltyWithTemperature = true;

                //This does not work with the checks currently!
                config.UseMeanOfDistributionForTravelTime = false;
                config.UseMeanOfDistributionForScore = false;

                config.IgnoreWaitingDuringDistributionAddition = true;


                config.AdjustDeterministicEarlyArrivalToTWStart = false;

                //Always adhere to the customers timewindows
                config.AdjustEarlyArrivalToTWStart = true;



                config.CutProbabilityDistributionAt0 = true;

                config.DefaultDistribution = new Gamma(0, 10);//new Normal(0, 0);
                //config.DefaultDistribution = new Normal(0, 0);

                config.NumRestarts = 3;


                //Need to change the operator configuration, such that move and add do not have the ability to create new routes. This ends up creating way to many new routes
                config.Operators = new List<(Operator, double, string, int)>()
            {
                //((x, y, z, w, v) =>Operators.AddRandomRemovedCustomer(x, y, z, w, v,false), 1, "add", 1), //repeated 1 time
                //((x, y, z, w, v) =>Operators.RemoveRandomCustomer(x, y, z, w, v), 1, "remove", 1), //repeated 1 time
                ((routes, viableRoutes, random, removed, temp) => Operators.MoveRandomCustomerToRandomCustomer(routes, viableRoutes, random,false), 1, "move", 1),//repeated 1 time
               ((x, y, z, w, v) => Operators.GreedilyMoveRandomCustomer(x, y, z), 0.1, "move_to_best",1), //repeated 1 time
                ((x, y, z, w, v) => Operators.MoveRandomCustomerToRandomRoute(x, y, z), 1, "move_to_random_route", 4), //repeated 4 times
                ((x, y, z, w, v) => Operators.SwapRandomCustomers(x, y, z), 1, "swap", 4), //repeated 4 times
                //((x, y, z, w, v) => Operators.SwapInsideRoute(x, y, z), 1, "swap_inside_route", 4), //repeated 4 times
                ((x, y, z, w, v) => Operators.ReverseOperator(x, y, z), 1, "reverse",1), //repeated 1 time
                ((x, y, z, w, v) => Operators.ScrambleSubRoute(x, y, z), 1, "scramble",1), //Repeated 1 time
                ((x, y, z, w, v) => Operators.SwapRandomTails(x, y, z), 1, "swap_tails",1), //Repeated 1 time
            };



                return config;
            }
        }
        public static LocalSearchConfiguration VRPSLTTWithoutWaitingNormal
        {
            get { var config = VRPSLTTWithoutWaiting; config.CutProbabilityDistributionAt0 = false; config.DefaultDistribution = new Normal(0, 0); return config; }
        }
        public static LocalSearchConfiguration VRPSLTTWithoutWaitingCutNormal
        {
            get { var config = VRPSLTTWithoutWaitingNormal; config.CutProbabilityDistributionAt0 = true; ; return config; }
        }
        public static LocalSearchConfiguration VRPSLTTWithoutWaitingJustMean
        {
            get { var config = VRPSLTTWithoutWaiting; config.AllowDeterministicEarlyArrival = false; config.PenalizeDeterministicEarlyArrival = true; config.ExpectedEarlinessPenalty = 0; config.ExpectedLatenessPenalty = 0; config.UseMeanOfDistributionForScore = true; config.UseMeanOfDistributionForTravelTime = true; return config; }
        }

        public static LocalSearchConfiguration VRPSLTTWithWaitingNormal
        {
            get
            {
                var config = VRPSLTTWithoutWaiting;

                //Need to use the normal approximation when waiting is taken into account
                config.DefaultDistribution = new Normal(0, 0);

                config.AllowDeterministicEarlyArrival = true;
                config.PenalizeDeterministicEarlyArrival = false;
                config.AdjustEarlyArrivalToTWStart = true;
                config.ExpectedEarlinessPenalty = 0;
                config.AllowEarlyArrivalInSimulation = true;
                config.CutProbabilityDistributionAt0 = false;


                //Use distributions to estimate maximization of distribution and deterministic value
                config.IgnoreWaitingDuringDistributionAddition = false;

                return config;
            }
        }

        public static LocalSearchConfiguration VRPSLTTWithWaitingCutNormal
        {
            get { var config = VRPSLTTWithWaitingNormal; config.CutProbabilityDistributionAt0 = true; return config; }
        }
        public static LocalSearchConfiguration VRPSLTTWithWaitingStupidGamma
        {
            get { var config = VRPSLTTWithWaitingNormal; config.DefaultDistribution = new Gamma(0, 10); config.AdjustDeterministicEarlyArrivalToTWStart = true; config.IgnoreWaitingDuringDistributionAddition = true; return config; }
        }
        public static LocalSearchConfiguration VRPSLTTWithWaitingJustMean
        {
            get { var config = VRPSLTTWithWaitingNormal; config.AllowDeterministicEarlyArrival = true; config.PenalizeDeterministicEarlyArrival = false; config.ExpectedEarlinessPenalty = 0; config.ExpectedLatenessPenalty = 0; config.UseMeanOfDistributionForScore = true; config.UseMeanOfDistributionForTravelTime = true; return config; }
        }
        public static LocalSearchConfiguration VRPSLTTWithoutWaitingNormalInBetweenMaximizaton
        {
            get { var config = VRPSLTTWithoutWaitingCutNormal; config.UseNormalMaximizationForAllSteps = true; return config; }
        }
        public static LocalSearchConfiguration VRPSLTTWithWaitingNormalInBetweenMaximizaton
        {
            get { var config = VRPSLTTWithWaitingCutNormal; config.UseNormalMaximizationForAllSteps = true; return config; }
        }
        public static LocalSearchConfiguration VRPLTTWithWind
        {
            get
            {
                var config = VRPLTTFinal;

                config.WindDirection = new double[] {0,-1} ;//{ 1/2, Math.Sqrt(3)/2 }; // {0,-1} this corresponds with wind from the north, i.e. 0 deg
                config.WindSpeed = 6.75; //3.75 corresponds with values used in the KDE testing setting

                return config;
            }
        }
        public static LocalSearchConfiguration VRPLTTDebug { get { var config = VRPLTTFinal; config.PrintExtendedInfo = true; return config; } }
        public static LocalSearchConfiguration VRPTW => new LocalSearchConfiguration
        {
            InitialTemperature = 1,
            AllowEarlyArrivalDuringSearch = true,
            AllowLateArrivalDuringSearch = true,
            AllowDeterministicEarlyArrival = true,
            AllowEarlyArrivalInSimulation = true,
            AllowLateArrival = false,
            BaseEarlyArrivalPenalty = 10,
            BaseLateArrivalPenalty = 10,

            BaseRemovedCustomerPenalty = 100,

            BaseRemovedCustomerPenaltyPow = 2,
            Alpha = 0.99,
            SaveColumnsAfterAllImprovements = true,
            PenalizeDeterministicEarlyArrival = false,
            PenalizeLateArrival = true,
            AdjustDeterministicEarlyArrivalToTWStart = true,
            CheckOperatorScores = false,
            SaveRoutesBeforeOperator = false,
            SaveColumnsAfterWorse = true,

            SaveColumnThreshold = 0.2,
            SaveScoreDevelopment = false,
            ExpectedEarlinessPenalty = 0,
            ExpectedLatenessPenalty = 0,
            UseMeanOfDistributionForTravelTime = false,
            ScaleEarlinessPenaltyWithTemperature = true,
            ScaleLatenessPenaltyWithTemperature = true,
            IgnoreWaitingDuringDistributionAddition = true,
            IterationsPerAlphaChange = 10000,
            NumIterationsOfNoChangeBeforeRestarting = 600000,
            RestartTemperatureBound = 0.02,
            NumRestarts = 7,
            WindSpeed = 0,

            Operators = new List<(Operator, double, string, int)>()
            {
                //((x, y, z, w, v) =>Operators.AddRandomRemovedCustomer(x, y, z, w, v), 1, "add", 1), //repeated 1 time
                //((x, y, z, w, v) =>Operators.RemoveRandomCustomer(x, y, z, w, v), 1, "remove", 1), //repeated 1 time
                ((routes, viableRoutes, random, removed, temp) => Operators.MoveRandomCustomerToRandomCustomer(routes, viableRoutes, random), 1, "move", 1),//repeated 1 time
               ((x, y, z, w, v) => Operators.GreedilyMoveRandomCustomer(x, y, z), 0.1, "move_to_best",1), //repeated 1 time
                ((x, y, z, w, v) => Operators.MoveRandomCustomerToRandomRoute(x, y, z), 1, "move_to_random_route", 4), //repeated 4 times
                ((x, y, z, w, v) => Operators.SwapRandomCustomers(x, y, z), 1, "swap", 4), //repeated 4 times
                //((x, y, z, w, v) => Operators.SwapInsideRoute(x, y, z), 1, "swap_inside_route", 4), //repeated 4 times
                ((x, y, z, w, v) => Operators.ReverseOperator(x, y, z), 1, "reverse",1), //repeated 1 time
                ((x, y, z, w, v) => Operators.ScrambleSubRoute(x, y, z), 1, "scramble",1), //Repeated 1 time
                ((x, y, z, w, v) => Operators.SwapRandomTails(x, y, z), 1, "swap_tails",1), //Repeated 1 time
            }

        };
        //public static List<Operator> SimpleOperators => new List<Func<List<Route>, List<int>, Random, List<Customer>, double, (double, Action?)>>() {Operators.AddRandomRemovedCustomer,Operators. };

        //public static (List<Operator>, List<double>) SimpleOperators
        //{
        //    get
        //    {
        //        return CreateSimpleOperators();
        //    }
        //}

        //private static (List<Operator>, List<double>) CreateSimpleOperators()
        //{

        //}

    }
}
