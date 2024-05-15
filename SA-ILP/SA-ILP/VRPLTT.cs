using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SA_ILP
{
    public static class VRPLTT
    {
        private static double CalculateSpeed(double heightDiff, double length, double vehicleMass, double powerInput, Vector<double> wind, Vector<double> td)
        {
            double speed = 25/3.6;
            double slope = Math.Asin(heightDiff / length);
            double requiredPow = CalcRequiredForce(speed, vehicleMass, slope, wind, td);
            double orignalPow = requiredPow;
            //length = Math.Sqrt((length*length) + heightDiff*heightDiff); // This line is needed to account for a longer travel distance when also traveling uphill.

            //this part is not necessary and also not correct
            //if (slope <= 0 && wind.L2Norm() == 0){return speed;}

            if (powerInput >= requiredPow)
            {
                return speed;
            }
            while (speed > 0)
            {
                speed -= 0.01;
                requiredPow = CalcRequiredForce(speed, vehicleMass, slope, wind, td);
                if (powerInput > requiredPow)
                {
                    if (orignalPow + requiredPow - 2 * powerInput < 0)
                        speed += 0.01;
                    return speed;
                }
                else
                {
                    orignalPow = requiredPow;
                }
            }
            return 0;
        }
        /// <summary>
        /// Binary searches the space of travel speeds in the range [0,25] km h^-1 for the maximum achievable speed.
        /// </summary>
        /// <param name="heightDiff"> the height difference</param>
        /// <param name="length"> the length of the road to be traveled </param>
        /// <param name="vehicleMass"> the total mass of the vehicle, including load and driver, in kg</param>
        /// <param name="powerInput"> the maximum input power</param>
        /// <param name="wind"> the wind as vector </param>
        /// <param name="td"> the travel direction as vector with length one</param>
        /// <returns> speed in ms^-1</returns>
        private static double CalculateSpeedBinSearch(double heightDiff, double length, double vehicleMass, double powerInput, Vector<double> wind, Vector<double> td)
        {
            double tolerance = 1e-5;
            int maxIterations = 1000;
            int iteration = 0;
            double speedLimit = 25;
            double lowerBound = 0;
            double upperBound = speedLimit;
            double midpoint = (double)(speedLimit-lowerBound)/2;
            double slope = Math.Asin(heightDiff / length);
            double requiredPower = CalcRequiredForce(speedLimit / 3.6, vehicleMass, slope, wind, td);
            double orignalPower = requiredPower;
            //length = Math.Sqrt((length*length) + heightDiff*heightDiff); // This line is needed to account for a longer travel distance when also traveling uphill.

            //this part is not necessary and also not correct
            //if (slope <= 0 && wind.L2Norm() == 0){return speed;}

            if (powerInput >= requiredPower)
            {
                // this is now in km/h
                return speedLimit/3.6;
            }
            while (speedLimit-lowerBound >= tolerance && iteration < maxIterations)
            {
                iteration ++;
                midpoint = (double)(upperBound+lowerBound)/2;
                if (CalcRequiredForce(speedLimit/3.6, vehicleMass, slope, wind, td)< powerInput)
                {
                    lowerBound = midpoint;
                }
                else if (CalcRequiredForce(speedLimit/3.6, vehicleMass, slope, wind, td)> powerInput)
                {
                    speedLimit = midpoint;
                } else
                {
                    return midpoint/3.6;
                }
                
            }
            return (double)(upperBound+lowerBound)/2;
        }
        /// <summary>
        /// This function computes the travel time for a given heightdifference, travel distance, mass, power, direction and wind.
        /// </summary>
        /// <param name="heightDiff"> difference in height over the arc</param>
        /// <param name="length"> length of the road to be traveled</param>
        /// <param name="vehicleMass"> total mass of the vehicle, including load and driver</param>
        /// <param name="powerInput"> the maximum power that can be put in </param>
        /// <param name="wind"> the wind as vector</param>
        /// <param name="td"> the travel direction as vector with length one</param>
        /// <returns> travel time in minutes </returns>
        public static double CalculateTravelTime(double heightDiff, double length, double vehicleMass, double powerInput, Vector<double> wind, Vector<double> td)
        {
            if (length == 0)
                return 0;
            length *= 1000;

            // double speed = CalculateSpeed(heightDiff, length, vehicleMass, powerInput, wind, td)/3.6;
            double speed = CalculateSpeedBinSearch(heightDiff, length, vehicleMass, powerInput, wind, td);

            //Return travel time in minutes
            return length / speed / 60;
        }
        public static Gamma CreateTravelTimeDistribution(double weight, double traveltime)
        {
            double shape = 1.5 + traveltime * 0.5 + weight / (290);
            double rate = 10;
            var gamma = new Gamma(shape, rate);

            return gamma;
        }

        //
        //https://stackoverflow.com/questions/16266809/convert-from-latitude-longitude-to-x-y
        public static (double, double) EquirectangularProjection(double latitude, double longitude, double centralLatitude, double centralLongitude)
        {
            double X = (longitude / 180 * Math.PI - centralLongitude / 180 * Math.PI) * Math.Cos(centralLatitude / 180 * Math.PI);
            double Y = (latitude / 180 * Math.PI - centralLatitude / 180 * Math.PI);
            return (X, Y);
        }

        //Calculate the load dependent time matrix, the gamma distribtued delay variable and some approximation of this gamma distribution to use during execution. It also returns the component of the windvector along the travel direction.
        public static (double[,,], Gamma[,,], IContinuousDistribution[,,], double[,] partOfWindTaken) CalculateLoadDependentTimeMatrix(List<Customer> customers, double[,] distanceMatrix, double minWeight, double maxWeight, int numLoadLevels, double powerInput, double windSpeed = 0, double[] windVec = null, bool UseNormalApproximation = false)
        {
            double[,,] matrix = new double[customers.Count, customers.Count, numLoadLevels];
            double[,] partOfWindTaken = new double[customers.Count, customers.Count];
            Gamma[,,] distributionMatrix = new Gamma[customers.Count, customers.Count, numLoadLevels];
            IContinuousDistribution[,,] approximationMatrix = new IContinuousDistribution[customers.Count, customers.Count, numLoadLevels];

            //Create the wind vector
            var V = Vector<double>.Build;
            if (windVec == null)
                windVec = new double[] { 1, 0 };
            var wd = V.DenseOfArray(windVec);
            wd = wd.Divide(wd.L2Norm());
            var wind = wd * windSpeed;

            //Calculate the average latitude and longitude for the projection
            double minLatitude = double.MaxValue;
            double maxLatitude = double.MinValue;
            double minLongtitude = double.MaxValue;
            double maxLongitude = double.MinValue;

            foreach (var c in customers)
            {
                if (c.X < minLatitude)
                    minLatitude = c.X;
                if (c.X > maxLatitude)
                    maxLatitude = c.X;
                if (c.Y < minLongtitude)
                    minLongtitude = c.Y;
                if (c.Y > maxLongitude)
                    maxLongitude = c.Y;
            }
            double centralLatitude = (minLatitude + maxLatitude) / 2;
            double centralLongitude = (minLongtitude + maxLongitude) / 2;

            //Calculate the various matrices.
            Parallel.For(0, customers.Count, new ParallelOptions { MaxDegreeOfParallelism = 16 }, i =>
                 {
                     for (int j = 0; j < customers.Count; j++)
                     {
                         double dist;
                         if (i < j)
                             dist = distanceMatrix[i, j];
                         else
                             dist = distanceMatrix[j, i];
                         double heightDiff = customers[j].Elevation - customers[i].Elevation;

                         //Convert the latitute and longitude coordinates of the customers to carthesian coordinates using equirectangular projection
                         (double X1, double Y1) = EquirectangularProjection(customers[j].X, customers[j].Y, centralLatitude, centralLongitude);
                         (double X2, double Y2) = EquirectangularProjection(customers[i].X, customers[i].Y, centralLatitude, centralLongitude);

                         double xDirection = X1 - X2;
                         double yDirection = Y1 - Y2;

                         //Create and normalize vector from the indivial values
                         double[] custVec = { xDirection, yDirection };
                         var td = V.DenseOfArray(custVec);
                         td = td.Divide(td.L2Norm());

                         //https://math.stackexchange.com/questions/286391/find-the-component-of-veca-along-vecb
                         double windComponentAlongTravelDirection = (wd * -td) / td.L2Norm();
                         partOfWindTaken[i, j] = windComponentAlongTravelDirection;

                         for (int l = 0; l < numLoadLevels; l++)
                         {
                             double loadLevelWeight = minWeight + ((maxWeight - minWeight) / numLoadLevels) * l + ((maxWeight - minWeight) / numLoadLevels) / 2;

                             matrix[i, j, l] = VRPLTT.CalculateTravelTime(heightDiff, dist, loadLevelWeight, powerInput, wind, td);
                             //Console.WriteLine($"Deze dingetje is afstand: {dist} tijd: {matrix[i,j,l]}");
                             distributionMatrix[i, j, l] = CreateTravelTimeDistribution(loadLevelWeight, matrix[i, j, l]);

                             //Set the approximate distribution. 
                             if (UseNormalApproximation)
                                 approximationMatrix[i, j, l] = new Normal(distributionMatrix[i, j, l].Mean, distributionMatrix[i, j, l].StdDev); // //
                             else
                                 approximationMatrix[i, j, l] = distributionMatrix[i, j, l];
                         }
                     }
                 });

            Gamma gam = null;
            double longest = 0;

            using (StreamWriter w = new StreamWriter("data.txt"))
                for (int i = 0; i < customers.Count; i++)
                    for (int j = 0; j < customers.Count; j++)
                        for (int l = 0; l < numLoadLevels; l++)
                        {
                            if (matrix[i, j, l] > longest)
                            {
                                longest = matrix[i, j, l];
                                gam = distributionMatrix[i, j, l];
                            }
                            w.WriteLine($"{matrix[i, j, l]};{distributionMatrix[i, j, l].Mean};{distributionMatrix[i, j, l].Mode}");
                        }
            Console.WriteLine($"{gam.Shape};{gam.Rate}");
            return (matrix, distributionMatrix, approximationMatrix, partOfWindTaken);
        }
        public static (double[,] distances, List<Customer> customers) ParseVRPLTTInstance(string file)
        {
            List<Customer> customers = new List<Customer>();

            using (StreamReader sr = new StreamReader(file))
            {
                //Console.WriteLine($"The filename of the CSV file is :{file}");
                var len = sr.ReadLine().Split(',').Length - 8;
                Console.WriteLine(file);
                Console.WriteLine(len);

                double[,] distances = new double[len, len];
                string line = "";
                while ((line = sr.ReadLine()) != null)
                {
                    var lineSplit = line.Split(',');
                    int id = int.Parse(lineSplit[0]);
                    double x = double.Parse(lineSplit[1], NumberStyles.Any, CultureInfo.InvariantCulture);
                    double y = double.Parse(lineSplit[2], NumberStyles.Any, CultureInfo.InvariantCulture);
                    double elevation = double.Parse(lineSplit[3], NumberStyles.Any, CultureInfo.InvariantCulture);
                    double demand, twstart, twend, serviceTime;
                    if (lineSplit[4] != "")
                        demand = double.Parse(lineSplit[4], NumberStyles.Any, CultureInfo.InvariantCulture);
                    else
                        demand = 0;

                    if (lineSplit[5] != "")
                        twstart = double.Parse(lineSplit[5], NumberStyles.Any, CultureInfo.InvariantCulture);
                    else
                        twstart = 0;
                    if (lineSplit[6] != "")
                        twend = double.Parse(lineSplit[6], NumberStyles.Any, CultureInfo.InvariantCulture);
                    else
                        twend = 0;
                    if (lineSplit[7] != "")
                        serviceTime = double.Parse(lineSplit[7], NumberStyles.Any, CultureInfo.InvariantCulture);
                    else
                        serviceTime = 0;

                    var customer = new Customer(id, x, y, demand, twstart, twend, serviceTime, elevation);
                    for (int i = 8; i < lineSplit.Length; i++)
                    {
                        distances[id, i - 8] = double.Parse(lineSplit[i], NumberStyles.Any, CultureInfo.InvariantCulture);
                    }
                    customers.Add(customer);
                }
                return (distances, customers);
            }
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="cycleSpeed"> the intended cycling speed in m/s</param>
        /// <param name="mass"> the total mass of the vehicle + load + driver</param>
        /// <param name="slope">the slope of the road in the travel direction in percentage</param>
        /// <param name="trueWind">the true wind as vector with units m/s</param>
        /// <param name="td">the travel direction, vector with length one</param>
        /// <returns></returns>
        public static double CalcRequiredForce(double cycleSpeed, double mass, double slope, Vector<double> trueWind, Vector<double> td)
        {
            //Constant for the equations of resistance
            double Cd = 1.18; //air drag coefficient
            double A = 0.83; // frontal area of bike
            double Ro = 1.18; // air density
            double Cr = 0.01; //roll resistance coefficient
            double g = 9.81; // gravitational acceleration

            double dragWindSpeedSquared = Math.Pow(cycleSpeed, 2); // Default dragwind speed squared

            //This loop changes the dragWindSpeed if the trueWind is not zero
            if (trueWind.L1Norm() != 0)
            {
                //Next line downplays the effect of wind when coming sideways (due to buildings,trees etc.)
                // if ((Math.Abs(Math.Acos(-td[0] * wind.Normalize(2)[0] + -td[1] * wind.Normalize(2)[1]))) >= 15 / 180 * Math.PI) { wind = wind * 0.6; }

                //Apparant wind(actual wind) is the vector sum of true wind and drag wind
                var apparantWind = -td * cycleSpeed + trueWind;

                //Creates a vector with the direction of the apparent wind, with squared length
                //var actualWindSquared = apparantWind.Divide(apparantWind.L2Norm()) * Math.Pow(apparantWind.L2Norm(), 2); //TotalFd;
                var actualWindSquared = apparantWind * apparantWind.L2Norm(); //TotalFd;

                //takes the inproduct with travel direction
                dragWindSpeedSquared = -td * actualWindSquared;
            }
            //Calculate the part of the air resistance in the direction of cycling
            double Fd = Cd * A * Ro * dragWindSpeedSquared / 2;
            double Fr = Cr * mass * g * Math.Cos(Math.Atan(slope));
            double Fg = mass * g * Math.Sin(Math.Atan(slope));

            //returns needed power to drive speed 'cycleSpeed' for given input of slope, wind and mass.
            return (Fd + Fr + Fg) * cycleSpeed / 0.95;
        }

        internal static double CalculateWindCyclingTime(string file, double bikeMinWeight, double bikeMaxWeight, int numLoadlevels, double bikePower, double[] windDirection, List<Route> solution, List<List<int>> custs = null)
        {
            //Calculate the load dependent time matrix with no wind to analyze the mount of time is spend cycling against the wind.
            var parsed = VRPLTT.ParseVRPLTTInstance(file);
            (double[,,] matrix, var dists, var approx, double[,] windpart) = VRPLTT.CalculateLoadDependentTimeMatrix(parsed.customers, parsed.distances, bikeMinWeight, bikeMaxWeight, numLoadlevels, bikePower, 0, windDirection);

            if (custs != null)
            {
                var ls = new LocalSearch(LocalSearchConfigs.VRPLTTFinal, 0);
                foreach (var r in custs)
                {
                    var newRoute = new Route(parsed.customers[0], matrix, dists, approx, bikeMaxWeight - bikeMinWeight, 1, ls);
                    foreach (var c in r)
                    {
                        if (c != 0)
                            newRoute.InsertCust(parsed.customers[c], newRoute.route.Count - 1);
                    }
                    solution.Add(newRoute);
                }
            }

            double timeCyclingAgainstWind = 0;
            foreach (var route in solution)
                if (route.route.Count > 2)
                {
                    double weight = route.used_capacity;
                    for (int i = 0; i < route.route.Count - 1; i++)
                    {
                        double ll = (weight / route.max_capacity) * numLoadlevels;
                        int loadLevel = (int)ll;

                        //The upperbound is inclusive
                        if (ll == loadLevel && weight != 0)
                            loadLevel--;
                        timeCyclingAgainstWind += matrix[route.route[i].Id, route.route[i + 1].Id, loadLevel] * windpart[route.route[i].Id, route.route[i + 1].Id];
                    }
                }

            return timeCyclingAgainstWind;
        }
    }
}