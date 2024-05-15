using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SA_ILP
{
    public class Customer
    {
        public int Id { get; private set; }
        public double X { get; private set; }
        public double Y { get; private set; }
        public double Demand { get; private set; }
        public double TWStart { get; private set; }
        public double TWEnd { get; private set; }

        public double ServiceTime { get; private set; }

        public double Elevation { get; private set; }

        public Customer(int id, double x, double y, double demand, double twstart, double twend, double serviceTime, double elevation = 0)
        {
            if (twend == 0)
                twend = double.MaxValue;

            this.Id = id;
            this.X = x;
            this.Y = y;
            this.Demand = demand;
            this.TWStart = twstart;
            this.TWEnd = twend;
            this.ServiceTime = serviceTime;
            this.Elevation = elevation;
        }

        public Customer(Customer cust)
        {
            this.Id = cust.Id;
            this.X = cust.X;
            this.Y = cust.Y;
            this.Demand = cust.Demand;
            this.TWEnd = cust.TWEnd;
            this.TWStart = cust.TWStart;
            this.ServiceTime = cust.ServiceTime;
        }

        public override bool Equals(object? obj)
        {
            if (obj == null)
                return false;
            if (obj.GetType() != typeof(Customer))
                return false;

            return ((Customer)obj).Id.Equals(this.Id);
        }

        public static bool operator ==(Customer? cust1, Customer? cust2)
        {
            if (cust1 is null)
            {
                if (cust2 is null)
                    return true;
                return false;
            }
            return cust1.Equals(cust2);
        }

        public static bool operator !=(Customer? cust1, Customer? cust2) => !(cust1 == cust2);

        public override string ToString()
        {
            return Id.ToString();
        }
    }
}
