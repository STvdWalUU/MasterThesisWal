global using Operator = System.Func<System.Collections.Generic.List<SA_ILP.Route>, System.Collections.Generic.List<int>, System.Random, System.Collections.Generic.List<SA_ILP.Customer>, SA_ILP.LocalSearch, (double, System.Action?)>;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SA_ILP
{
    internal class OperatorSelector
    {
        //Manages the selection of operators/neighborhoods 

        Random random;
        List<Operator> operators;
        List<double> weights;
        List<String> labels;
        List<double> threshHolds;
        List<int> repeats;
        private int last = -1;

        List<String> operatorHistory;

        public List<String> OperatorList => labels.ConvertAll(x => x);

        public String LastOperator { get; private set; }
        public OperatorSelector(Random random)
        {
            this.random = random;
            operators = new List<Operator>();
            weights = new List<double>();
            threshHolds = new List<double>();
            labels = new List<string>();
            LastOperator = "none";
            operatorHistory = new List<string>();
            repeats = new List<int>();
        }
        public void Add(Operator op, double weight, String label = "unnamed-operator", int numRepeats = -1)
        {
            if (numRepeats == -1)
            {
                operators.Add(op);
                repeats.Add(1);
            }
            else
            {
                operators.Add((x, y, z, w, v) => Operators.RepeatNTimes(numRepeats, op, x, y, z, w, v));
                repeats.Add(numRepeats);
            }
            weights.Add(weight);
            labels.Add(label);

            // the thresholds list is redefined with adding each new operator
            threshHolds = new List<double>();
            double totalWeight = weights.Sum();

            double cumulative = 0;
            foreach (double w in weights)
            {
                cumulative += w;
                threshHolds.Add(cumulative / totalWeight);
            }

        }


        public void Add(List<Operator> operators, List<double> weights)
        {
            if (operators.Count != weights.Count)
                throw new Exception("List counts must match");

            for (int i = 0; i < operators.Count; i++)
                Add(operators[i], weights[i]);
        }

        public Operator Next()
        {
            var p = random.NextDouble();
            for (int i = 0; i < threshHolds.Count; i++)
            {
                // thresholds is a list of doubles that split up the interval [0.1] into subintervals 
                //with relativel length equal to their relative weight
                if (p <= threshHolds[i])
                {
                    LastOperator = labels[i];
                    //operatorHistory.Add(labels[i]);
                    return operators[i];
                }
            }

            throw new Exception("Threshold error");
        }
        public override string ToString()
        {
            string res = "";
            for (int i = 0; i < operators.Count; i++)
            {
                res += $"OP: {labels[i]} RP: {weights[i]} Repeats: {repeats[i]}\n";
            }

            return res;
            //return base.ToString();
        }
    }
}
