using MathNet.Numerics.Distributions;
using MathNet.Numerics;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.ComponentModel;
using System.IO.Enumeration;

namespace SA_ILP
{
    /// <summary>
    /// Class that is used to do the basis manipulations needed for using the KDE framework.
    /// </summary>
    public static class KDEdata
    {
        static string FileName ;
        private static double[,,,] observationsMatrix;
        private static bool matrixGenerated = false;
        private static readonly object _lock = new object();
        /// <summary>
        /// Stores the observation matrix corresponding to the txt file as variable.
        /// </summary>
        public static double[,,,] ObservationsMatrix
        {
            get
            {
                if (!matrixGenerated)
                {
                    lock (_lock)
                    {
                        if (!matrixGenerated)
                        {
                            observationsMatrix = ConvertTXTtoMatrix(FileName);
                            matrixGenerated = true;
                        }
                    }
                }
                return observationsMatrix;
            }
        }
        /// <summary>
        /// Computes the loadlevel corresponding to a certain load, 
        /// assuming the differenct between the maximum and minimum load is 150kg
        /// </summary>
        /// <param name="load"></param>
        /// <returns>loadleavel</returns>
        public static int calculateKDELoadLevel(double load)
        { 
            int loadlevel;
            loadlevel = (int)Math.Min(Math.Max(Math.Ceiling(10 * (load/ 150.0) - 1) , 0), 9);
            //if (load<0)
                //Console.WriteLine($"{load} returns {loadlevel}");
            return loadlevel;
        }
        public static double calculateKDEObjective(double totalTravelTime, double totalWaitingTime, double WaitingPenalty ,double TotalTardiness, double LatenessPenalty, double numberTooLate)
        {
            double objectiveValue = totalTravelTime + totalWaitingTime * WaitingPenalty + TotalTardiness*LatenessPenalty;
            return objectiveValue/observationsMatrix.GetLength(3);
        }
        /// <summary>
        /// Converts a 2D txt file into a 4D-array with at position [i,j,l,k] 
        /// the traveltime coresponding to arc (i,j), loadlevel l and observation k.
        /// </summary>
        /// <param name="fileName"></param>
        /// <returns>travelTimeMatrix</returns>
        public static double[,,,] ConvertTXTtoMatrix(string fileName)
        {
            FileName = fileName;
            Console.WriteLine($"Reading observations file {fileName}");
            StreamReader Textfile = new StreamReader(fileName);
            // Store each line in array of strings 
            string line="";
            try
            {
                line = Textfile.ReadLine();
            }
            catch (System.Exception e)
            {
                Console.WriteLine("THIS FAILS");
                Console.WriteLine(e.Message);
            }
            //string[] variables = new string[3];
            string[] variables = line.Split(",");
            int nrCust = int.Parse(variables[0]);
            int nrLoadlevels = int.Parse(variables[1]);
            int nrObservations = 1;
            if (variables[2] == " ETT")
                nrObservations = 1;
            else
                nrObservations = int.Parse(variables[2]);
            
            //Console.WriteLine(string.Format("nrCust = {0} and nrLoadlevels = {1} and nrObservations = {2}",nrCust,nrLoadlevels, nrObservations));
            
            string row = Textfile.ReadLine();
            int rowCounter = 0;
            int columnCounter;
            double[,,,] travelTimeMatrix = new double[nrCust,nrCust,nrLoadlevels,nrObservations];
            
            while(row != null){
                //Console.WriteLine(row);
                columnCounter = 0;
                foreach(string number in row.Split(",")){

                    int a = rowCounter/nrCust;
                    int b = rowCounter%nrCust;
                    int c = columnCounter/nrObservations;
                    int d = columnCounter%nrObservations;
                    //Console.WriteLine("a=" + a.ToString() + " b=" +b.ToString() +" c="+c.ToString() + " d=" + d.ToString());
                    try
                    {
                        travelTimeMatrix[a,b,c,d] = double.Parse(number,System.Globalization.CultureInfo.InvariantCulture);
                    }
                    catch (System.Exception e)
                    {
                        Console.WriteLine("FAILED");
                        Console.WriteLine(e.Message);
                    }
                    
                    columnCounter ++;
                }
                rowCounter++;
                row = Textfile.ReadLine();       
            }
        return travelTimeMatrix;    
    }

    }
}
