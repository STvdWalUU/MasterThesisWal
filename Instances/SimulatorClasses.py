#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 12:13:50 2023

@author: stijnvanderwal
"""

import numpy as np

class customer:
    totalCustomers = 0
    
    def __init__(self, number, x,y,h,demand, TWearly, TWlate, serviceTime = 5 ):
        self.number = number
        self.TWearly = TWearly
        self.TWlate = TWlate
        self.demand = demand
        self.x = x
        self.y = y
        self.h = h
        self.serviceTime = serviceTime
        customer.totalCustomers += 1
    
    def __str__(self):
        if self.number == 0:
            return f"Depot at [{round(self.x,3)},{round(self.y,3)},{round(self.h,3)}] with TW [{round(self.TWearly,2)},{round(self.TWlate,2)}] and demand {self.demand}"
        return f"Customer {self.number} at [{round(self.x,3)},{round(self.y,3)},{round(self.h,3)}] with TW [{round(self.TWearly,2)},{round(self.TWlate,2)}] and demand {self.demand}"
        
class customerCollection:
    def __init__(self, customers):
        self.customers = []
        self.customers += customers
    
    def emptyRoute(self, nrCustomers):
        customs = []
        for i in range(nrCustomers-1):
            cust = customer(i+1, 0, 480, 5, 0,0,0)
            customs.append(cust)
        return customerCollection(customs)
    
    def __str__(self):
        string = ""
        for cust in self.customers:
            string += str(cust.number) + ", "
        return string
        
class route(customerCollection):
    depot = customer(0, 50, 500, 0 , 0, 0, 10e10, serviceTime=0)
    def __init__(self, nrRoute, customers, starttime):
        self.nrRoute = nrRoute
        self.startTime = starttime
        load = 0
        for cust in customers:
            load += cust.demand
        self.totalLoad = load
        self.customers = []
        self.customers += customers
        
    def randomRoute(nrRoute, lengthRoute):
        customers = []
        for i in range(lengthRoute):
            dx = np.random.normal(50,10)
            dy = np.random.normal(500,10)
            h = np.random.normal(0,1)
            cust = customer(i+1, i*10+50,i*15 + 200,5,dx,dy,h)
            customers.append(cust)
        return route(nrRoute, customers,0)
    
    def testRoute(nrRoute):
        customers = []
        for i in range(13):
            dx = 45+i
            dy = 450 + 10*i
            h = 0 
            cust = customer(i+1, (1+i)*145,(i+1)*150,5,dx,dy,h)
            if i >= 8:
                cust.TWearly += 50*i
                cust.TWlate += (100*i)
            customers.append(cust)
        return route(nrRoute, customers,0)
    
    def write(self):
        custs = self.customers
        for customer in custs:
            print(f"{customer.number}")
        print("\n")
   
    def writeLocations(self):
        custs = self.customers
        print(f"Locations for route {self.nrRoute}")
        for customer in custs:
            print(f"{customer.x},{customer.y}")

class schedule:
    def __init__(self, routes):
        if not isinstance(routes, list):
            raise Exception("input must be a list")
        if not isinstance(routes[0], customerCollection):
            raise Exception("items of list not routes")
        self.routes = routes
    
    def __str__(self):
        string = ""
        for route in self.routes:
            for cust in route.customers:
                string += str(cust.number) + ", "
            string += "\n"
        return string
    
    def write(self):
        for route in self.routes:
            route.write()