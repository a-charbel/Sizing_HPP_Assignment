#! /usr/bin/env python
from __future__ import print_function, division, absolute_import, unicode_literals
# Copyright (C) 2019 DTU Wind

import os
import logging
import sys

import pandas as pd
import numpy as np
import openpyxl
import math
from statistics import mean

import matplotlib.pyplot as plt
from datetime import date, timedelta, datetime
import numpy_financial as npf

#from __future__ import division
import pyomo.environ as pyo

#from landbosse.excelio import landbosse_input_dir


class HPP(object):

    """
    A class used to represent a hybrid power plant

    ...

    Attributes
    ----------

    hpp_grid_connection : float
        Grid connection capacity in MW
    hpp_land_area_available : float
        Total land area available for the HPP in sq. km.
    wind_rating_WT : float
        Rating of the wind turbine in MW
    wind_nWT_per_string : int
        Number of wind turbines connected in series in each string
    wind_lifetime_WT : int/float
        Lifetime of wind turbines in years
    wind_rotor_diameter : float
        Rotor diameter of the wind turbine in m
    wind_hub_height : float
        Hub height of the wind turbine in m
    wind_turbine_spacing : int
        Spacing between 2 wind turbines as a multiple of rotor diameter
    wind_turbine_row_spacing : float
        Spacing between 2 rows of WT strings in m
    solar_lifetime_PV : int/float
        Lifetime of the solar PV in years

    Methods
    -------
    load_Exogenous_Data_Wind()
        Load external data for wind power
        Exogenous Data involve wind power timeseries
    load_Exogenous_Data_Solar()
        Load external data for solar power
        Exogenous Data involve solar power timeseries
    load_ExogenousData_Price()
        Load external data for market price
        Exogenous Data involve spot market price timeseries
    load_Exogenous_Data_Sizing()
        Load all the external data for sizing optimization
        Exogenous Data involve wind power, solar power and spot market timeseries
    load_Exogenous_Data_EMS()
        Load all the external data for sizing optimization
        Exogenous Data involve wind power, solar power, spot market, balancing market data, power forecast and spot market forecast
    calculate_Capacity_Factor(ts, capacity)
        Calculate the capacity factor of any power plant with input power time series in MW, ts and capacity in MW
    sizing(wind_ts, solar_ts, price_ts)
        calls sizing optimization method for either wind/solar or wind/solar/battery for sizing optimization depending on input from parameters_Simulation
    sizing_Wind_Solar(wind_ts, solar_ts, price_ts)
        performs sizing optimization for wind and solar sizing
    sizing_Wind_Solar_Battery(wind_ts, solar_ts, price_ts)
        performs sizing optimization for wind and solar sizing
    calculate_LCOE(investment_cost, maintenance_cost_per_year, discount_rate, lifetime, AEP_per_year)
        calculate the LCOE for given investment cost, maintenance cost per year, discount rate, lifetime, AEP per year
    calculate_IRR(P_HPP_t, price_t, investment_cost, maintenance_cost_per_year)
        calculate internal rate of return for given input of power generation from HPP, price timeseries, investment cost, maintenance cost per year
    calculate_NPV(P_HPP_t, price_t, investment_cost, maintenance_cost_per_year, discount_rate)
        calculate net present value for given input of power generation from HPP, price timeseries, investment cost, maintenance cost per year and discount rate
    """

    def __init__(
            self,
            parameter_dict,
            simulation_dict={},
    ):

        # add all parameters in parameter_dict into the hpp object
        parameter_list = []
        for key, value in parameter_dict.items():
            # This line define properties in the class
            # it is equivalent to define them one-by-one
            # self.hpp_grid_connection=hpp_grid_connection
            self.__dict__[key] = value
            parameter_list += [key]
        self.parameter_list = parameter_list

        simulation_list = []
        for key, value in simulation_dict.items():
            # This line define properties in the class
            # it is equivalent to define them one-by-one
            # self.hpp_grid_connection=hpp_grid_connection
            self.__dict__[key] = value
            simulation_list += [key]

    # Define methods of hpp class
    """
    Load exogenous data for wind solar and market prices
    """

    def load_Exogenous_Data_Sizing(
        self,
        input_dir_ts,
        input_wind_ts_filename,
        input_solar_ts_filename,
        input_price_ts_filename,
        timename,
        timeFormat_wind,
        timeFormat_solar,
        timeFormat_price,
        timeZone_wind,
        timeZone_solar,
        timeZone_price,
        timeZone_analysis,
    ):
        """
        Method for reading wind power, solar power and price timeseries csv files;
        convert the timeseries into specified time format;

        Output
        ------
        data: Formatted timeseries

        """
        if wind_as_component == 1:
            # wind_ts=self.load_Exogenous_Data_Wind()
            wind_ts = read_csv_Data(
                input_dir_ts + input_wind_ts_filename,
                timename,
                timeFormat_wind,
                timeZone_wind,
                timeZone_analysis,
            )
        else:
            wind_ts = self.pd.Series([])

        if solar_as_component == 1:
            # solar_ts=self.load_Exogenous_Data_Solar()
            solar_ts = read_csv_Data(
                input_dir_ts + input_solar_ts_filename,
                timename,
                timeFormat_solar,
                timeZone_solar,
                timeZone_analysis,
            )
        else:
            solar_ts = self.pd.Series([])

        # price_ts=self.load_Exogenous_Data_Price()
        price_ts = read_csv_Data(
            input_dir_ts + input_price_ts_filename,
            timename,
            timeFormat_price,
            timeZone_price,
            timeZone_analysis,
        )

        return wind_ts, solar_ts, price_ts

    def load_Exogenous_Data_EMS(self):
        """
        Method for reading wind power, solar power and price timeseries csv files;
        convert the timeseries into specified time format;

        Output
        ------
        data: Formatted timeseries

        """
        wind_ts = self.load_Exogenous_Data_Wind()
        solar_ts = self.load_Exogenous_Data_Solar()
        price_ts = self.load_Exogenous_Data_Price()

        return wind_ts, solar_ts, price_ts

    """
    Derived methods of hpp class
    """

    def calculate_Capacity_Factor(self, ts, capacity):
        """
        Method for calculating capacity factor for a considered location

        Input Parameters
        ----------------
        ts: timeseries data

        Output Parameters
        -----------------
        cap_fact: capacity factor for a considered location
        """
        # cap_fact = sum(ts.values)/len(ts.values)/capacity
        try:
            cap_fact = ts.mean() / capacity
        except:
            cap_fact = mean(ts)/capacity

        return cap_fact

    def sizing(self, wind_ts, solar_ts, price_ts):
        """
        DO NOT USE FOR ASSIGNMENT
        High level method to calculate sizing of HPP (wind and solar),
        with/without battery storage - 

        Returns
        -------
        Capacity of Wind Power
        Capacity of Solar Power
        HPP power output timeseries
        HPP power curtailment timeseries
        HPP total CAPEX
        HPP total OPEX
        Levelised cost of energy
        """
        # extract parameters into the variable space
        globals().update(self.__dict__)

        if (wind_as_component == 1) and \
           (solar_as_component == 1) and \
           (battery_as_component == 0):
            [hpp_wind_capacity, hpp_solar_capacity, P_HPP_t,
             P_curtailment_t, hpp_investment_cost, hpp_maintenance_cost,
             LCOE, NPV, IRR] = self.sizing_Wind_Solar(
                wind_ts, solar_ts, price_ts)
            return hpp_wind_capacity, hpp_solar_capacity, P_HPP_t, P_curtailment_t, hpp_investment_cost, hpp_maintenance_cost, LCOE, NPV, IRR

        if (wind_as_component == 1) and \
           (solar_as_component == 1) and \
           (battery_as_component == 1):
            [hpp_wind_capacity, hpp_solar_capacity, hpp_battery_power_rating,
             hpp_battery_energy_capacity, P_RES_available_t,
             P_HPP_t, P_curtailment_t, P_charge_discharge_t,
             E_SOC_t, hpp_investment_cost, hpp_maintenance_cost,
             LCOE, NPV, IRR] = self.sizing_Wind_Solar_Battery(
                wind_ts, solar_ts, price_ts)
            return hpp_wind_capacity, hpp_solar_capacity, hpp_battery_power_rating, hpp_battery_energy_capacity, P_RES_available_t, P_HPP_t, P_curtailment_t, P_charge_discharge_t, E_SOC_t, hpp_investment_cost, hpp_maintenance_cost, LCOE, NPV, IRR


    def sizing_Wind_Solar_Pyomo(self, wind_ts, solar_ts, price_ts):
        """
        Method to calculate sizing of wind and solar

        Returns
        -------
        Capacity of Wind Power
        Capacity of Solar Power
        HPP power output timeseries
        HPP power curtailment timeseries
        HPP total CAPEX
        HPP total OPEX
        Levelised cost of energy
        """
        
        # extract parameters into the variable space
        globals().update(self.__dict__)
        
        time = price_ts.index
        
        model = pyo.ConcreteModel()
        
        ## Variables ##
        model.IDX1 = range(len(time))
        model.IDX2 = range(1)
        model.IDX3 = range(hpp_lifetime)

        model.P_HPP_t = pyo.Var(model.IDX1, domain=pyo.NonNegativeReals, bounds=(0,hpp_grid_connection))
        model.P_curtailment_t = pyo.Var(model.IDX1, domain=pyo.NonNegativeReals)
        model.Wind_MW = pyo.Var(model.IDX2, domain=pyo.NonNegativeIntegers)
        model.Solar_MW = pyo.Var(model.IDX2, domain=pyo.NonNegativeIntegers)
        
        ## Constraints ##
        model.curtailment_constraint = pyo.ConstraintList()
        model.power_constraint = pyo.ConstraintList()
        
        for t in range(0,len(time)):
            model.curtailment_constraint.add(model.P_curtailment_t[t] >= wind_ts[t] * model.Wind_MW[0] +
                solar_ts[t] * model.Solar_MW[0] -
                hpp_grid_connection)
            model.power_constraint.add(model.P_HPP_t[t] == wind_ts[t] * model.Wind_MW[0] +
                solar_ts[t] * model.Solar_MW[0] -
                model.P_curtailment_t[t])
        
        
        # Objective Function ##
        model.OBJ = pyo.Objective( expr = 
            # CAPEX
            -(wind_turbine_cost * model.Wind_MW[0] + \
              solar_PV_cost * model.Solar_MW[0] + \
              hpp_BOS_soft_cost * (model.Wind_MW[0] + model.Solar_MW[0]) + \
              hpp_grid_connection_cost * hpp_grid_connection + \
              solar_hardware_installation_cost * model.Solar_MW[0] + \
              wind_civil_works_cost * model.Wind_MW[0]) + \
                # revenues and OPEX
                sum((sum(price_ts[t] * (wind_ts[t] * model.Wind_MW[0] + \
                    solar_ts[t] * model.Solar_MW[0] - \
                    model.P_curtailment_t[t]) for t in model.IDX1) - \
                    (wind_fixed_onm_cost * model.Wind_MW[0] + \
                    solar_fixed_onm_cost * model.Solar_MW[0])) / \
                    np.power(1 + hpp_discount_factor,(i+1)) \
                    for i in model.IDX3) \
                    , sense=pyo.maximize \
            )
            
        opt = pyo.SolverFactory('glpk')
        results = opt.solve(model, tee=True)
        results.write()
        
        ## Return calculated results ##
        P_curtailment_ts = []
        P_HPP_ts = []
        
        for count in range(0, len(time)):
            P_curtailment_ts.append(model.P_curtailment_t[count]())
            P_HPP_ts.append(model.P_HPP_t[count]())
                    
        AEP = sum(P_HPP_ts)
        AEP_per_year = np.ones(hpp_lifetime) * AEP

        # Investment cost
        hpp_investment_cost = \
            wind_turbine_cost * model.Wind_MW[0]() + \
            solar_PV_cost * model.Solar_MW[0]() + \
            hpp_grid_connection_cost * hpp_grid_connection + \
            hpp_BOS_soft_cost * (model.Wind_MW[0]() + model.Solar_MW[0]()) + \
            wind_civil_works_cost * model.Wind_MW[0]() + \
            solar_hardware_installation_cost * model.Solar_MW[0]()
        
        # Maintenance cost
        hpp_maintenance_cost = np.ones(hpp_lifetime) * (wind_fixed_onm_cost *\
            model.Wind_MW[0]() + solar_fixed_onm_cost * model.Solar_MW[0]())

        LCOE = self.calculate_LCOE(
            hpp_investment_cost,
            hpp_maintenance_cost,
            hpp_discount_factor,
            hpp_lifetime,
            AEP_per_year)

        
        IRR = self.calculate_IRR(
            P_HPP_ts,
            price_ts,
            hpp_investment_cost,
            hpp_maintenance_cost)
        
        NPV = self.calculate_NPV(
            P_HPP_ts,
            price_ts,
            hpp_investment_cost,
            hpp_maintenance_cost,
            hpp_discount_factor)
        
        return model.Wind_MW[0](), model.Solar_MW[0](), P_HPP_ts, P_curtailment_ts, hpp_investment_cost, hpp_maintenance_cost, LCOE, NPV, IRR
    
    def sizing_Wind_Pyomo(self, wind_ts, solar_ts, price_ts):
        """
        Method to calculate sizing of wind and solar

        Returns
        -------
        Capacity of Wind Power
        Capacity of Solar Power
        HPP power output timeseries
        HPP power curtailment timeseries
        HPP total CAPEX
        HPP total OPEX
        Levelised cost of energy
        """
        
        # extract parameters into the variable space
        globals().update(self.__dict__)
        
        time = price_ts.index
        
        model = pyo.ConcreteModel()
        
        ## Variables ##
        model.IDX1 = range(len(time))
        model.IDX2 = range(1)
        model.IDX3 = range(hpp_lifetime)

        model.P_HPP_t = pyo.Var(model.IDX1, domain=pyo.NonNegativeReals, bounds=(0,hpp_grid_connection))
        model.P_curtailment_t = pyo.Var(model.IDX1, domain=pyo.NonNegativeReals)
        model.Wind_MW = pyo.Var(model.IDX2, domain=pyo.NonNegativeIntegers)
        model.Solar_MW = pyo.Var(model.IDX2, domain=pyo.NonNegativeIntegers)
        
        ## Constraints ##
        model.curtailment_constraint = pyo.ConstraintList()
        model.power_constraint = pyo.ConstraintList()
        model.solar_constraint = pyo.ConstraintList()
        
        model.solar_constraint.add(model.Solar_MW[0] == 0)

        for t in range(0,len(time)):
            model.curtailment_constraint.add(model.P_curtailment_t[t] >= wind_ts[t] * model.Wind_MW[0] +
                solar_ts[t] * model.Solar_MW[0] -
                hpp_grid_connection)
            model.power_constraint.add(model.P_HPP_t[t] == wind_ts[t] * model.Wind_MW[0] +
                solar_ts[t] * model.Solar_MW[0] -
                model.P_curtailment_t[t])
        
        
        # Objective Function ##
        model.OBJ = pyo.Objective( expr = 
            # CAPEX
            -(wind_turbine_cost * model.Wind_MW[0] + \
              solar_PV_cost * model.Solar_MW[0] + \
              hpp_BOS_soft_cost * (model.Wind_MW[0] + model.Solar_MW[0]) + \
              hpp_grid_connection_cost * hpp_grid_connection + \
              solar_hardware_installation_cost * model.Solar_MW[0] + \
              wind_civil_works_cost * model.Wind_MW[0]) + \
                # revenues and OPEX
                sum((sum(price_ts[t] * (wind_ts[t] * model.Wind_MW[0] + \
                    solar_ts[t] * model.Solar_MW[0] - \
                    model.P_curtailment_t[t]) for t in model.IDX1) - \
                    (wind_fixed_onm_cost * model.Wind_MW[0] + \
                    solar_fixed_onm_cost * model.Solar_MW[0])) / \
                    np.power(1 + hpp_discount_factor,(i+1)) \
                    for i in model.IDX3) \
                    , sense=pyo.maximize \
            )
            
        opt = pyo.SolverFactory('glpk')
        results = opt.solve(model, tee=True)
        results.write()
        
        ## Return calculated results ##
        P_curtailment_ts = []
        P_HPP_ts = []
        
        for count in range(0, len(time)):
            P_curtailment_ts.append(model.P_curtailment_t[count]())
            P_HPP_ts.append(model.P_HPP_t[count]())
                    
        AEP = sum(P_HPP_ts)
        AEP_per_year = np.ones(hpp_lifetime) * AEP

        # Investment cost
        hpp_investment_cost = \
            wind_turbine_cost * model.Wind_MW[0]() + \
            solar_PV_cost * model.Solar_MW[0]() + \
            hpp_grid_connection_cost * hpp_grid_connection + \
            hpp_BOS_soft_cost * (model.Wind_MW[0]() + model.Solar_MW[0]()) + \
            wind_civil_works_cost * model.Wind_MW[0]() + \
            solar_hardware_installation_cost * model.Solar_MW[0]()
        
        # Maintenance cost
        hpp_maintenance_cost = np.ones(hpp_lifetime) * (wind_fixed_onm_cost *\
            model.Wind_MW[0]() + solar_fixed_onm_cost * model.Solar_MW[0]())

        LCOE = self.calculate_LCOE(
            hpp_investment_cost,
            hpp_maintenance_cost,
            hpp_discount_factor,
            hpp_lifetime,
            AEP_per_year)

        
        IRR = self.calculate_IRR(
            P_HPP_ts,
            price_ts,
            hpp_investment_cost,
            hpp_maintenance_cost)
        
        NPV = self.calculate_NPV(
            P_HPP_ts,
            price_ts,
            hpp_investment_cost,
            hpp_maintenance_cost,
            hpp_discount_factor)
        
        return model.Wind_MW[0](), model.Solar_MW[0](), P_HPP_ts, P_curtailment_ts, hpp_investment_cost, hpp_maintenance_cost, LCOE, NPV, IRR
    
    
    def sizing_Solar_Pyomo(self, wind_ts, solar_ts, price_ts):
        """
        Method to calculate sizing of wind and solar

        Returns
        -------
        Capacity of Wind Power
        Capacity of Solar Power
        HPP power output timeseries
        HPP power curtailment timeseries
        HPP total CAPEX
        HPP total OPEX
        Levelised cost of energy
        """
        
        # extract parameters into the variable space
        globals().update(self.__dict__)
        
        time = price_ts.index
        
        model = pyo.ConcreteModel()
        
        ## Variables ##
        model.IDX1 = range(len(time))
        model.IDX2 = range(1)
        model.IDX3 = range(hpp_lifetime)

        model.P_HPP_t = pyo.Var(model.IDX1, domain=pyo.NonNegativeReals, bounds=(0,hpp_grid_connection))
        model.P_curtailment_t = pyo.Var(model.IDX1, domain=pyo.NonNegativeReals)
        model.Wind_MW = pyo.Var(model.IDX2, domain=pyo.NonNegativeIntegers)
        model.Solar_MW = pyo.Var(model.IDX2, domain=pyo.NonNegativeIntegers)
        
        ## Constraints ##
        model.curtailment_constraint = pyo.ConstraintList()
        model.power_constraint = pyo.ConstraintList()
        model.wind_constraint = pyo.ConstraintList()
        
        model.wind_constraint.add(model.Wind_MW[0] == 0)

        for t in range(0,len(time)):
            model.curtailment_constraint.add(model.P_curtailment_t[t] >= wind_ts[t] * model.Wind_MW[0] +
                solar_ts[t] * model.Solar_MW[0] -
                hpp_grid_connection)
            model.power_constraint.add(model.P_HPP_t[t] == wind_ts[t] * model.Wind_MW[0] +
                solar_ts[t] * model.Solar_MW[0] -
                model.P_curtailment_t[t])
        
        
        # Objective Function ##
        model.OBJ = pyo.Objective( expr = 
            # CAPEX
            -(wind_turbine_cost * model.Wind_MW[0] + \
              solar_PV_cost * model.Solar_MW[0] + \
              hpp_BOS_soft_cost * (model.Wind_MW[0] + model.Solar_MW[0]) + \
              hpp_grid_connection_cost * hpp_grid_connection + \
              solar_hardware_installation_cost * model.Solar_MW[0] + \
              wind_civil_works_cost * model.Wind_MW[0]) + \
                # revenues and OPEX
                sum((sum(price_ts[t] * (wind_ts[t] * model.Wind_MW[0] + \
                    solar_ts[t] * model.Solar_MW[0] - \
                    model.P_curtailment_t[t]) for t in model.IDX1) - \
                    (wind_fixed_onm_cost * model.Wind_MW[0] + \
                    solar_fixed_onm_cost * model.Solar_MW[0])) / \
                    np.power(1 + hpp_discount_factor,(i+1)) \
                    for i in model.IDX3) \
                    , sense=pyo.maximize \
            )
            
        opt = pyo.SolverFactory('glpk')
        results = opt.solve(model, tee=True)
        results.write()
        
        ## Return calculated results ##
        P_curtailment_ts = []
        P_HPP_ts = []
        
        for count in range(0, len(time)):
            P_curtailment_ts.append(model.P_curtailment_t[count]())
            P_HPP_ts.append(model.P_HPP_t[count]())
                    
        AEP = sum(P_HPP_ts)
        AEP_per_year = np.ones(hpp_lifetime) * AEP

        # Investment cost
        hpp_investment_cost = \
            wind_turbine_cost * model.Wind_MW[0]() + \
            solar_PV_cost * model.Solar_MW[0]() + \
            hpp_grid_connection_cost * hpp_grid_connection + \
            hpp_BOS_soft_cost * (model.Wind_MW[0]() + model.Solar_MW[0]()) + \
            wind_civil_works_cost * model.Wind_MW[0]() + \
            solar_hardware_installation_cost * model.Solar_MW[0]()
        
        # Maintenance cost
        hpp_maintenance_cost = np.ones(hpp_lifetime) * (wind_fixed_onm_cost *\
            model.Wind_MW[0]() + solar_fixed_onm_cost * model.Solar_MW[0]())

        LCOE = self.calculate_LCOE(
            hpp_investment_cost,
            hpp_maintenance_cost,
            hpp_discount_factor,
            hpp_lifetime,
            AEP_per_year)

        
        IRR = self.calculate_IRR(
            P_HPP_ts,
            price_ts,
            hpp_investment_cost,
            hpp_maintenance_cost)
        
        NPV = self.calculate_NPV(
            P_HPP_ts,
            price_ts,
            hpp_investment_cost,
            hpp_maintenance_cost,
            hpp_discount_factor)
        
        return model.Wind_MW[0](), model.Solar_MW[0](), P_HPP_ts, P_curtailment_ts, hpp_investment_cost, hpp_maintenance_cost, LCOE, NPV, IRR
    
    def sizing_Wind_Solar_Battery_Pyomo(self, wind_ts, solar_ts, price_ts):
        """
        Method to calculate sizing of wind and solar and battery

        Returns
        -------
        Capacity of Wind Power
        Capacity of Solar Power
        Capacity of Batteru
        HPP power output timeseries
        HPP power curtailment timeseries
        HPP total CAPEX
        HPP total OPEX
        Levelised cost of energy
        """

        # extract parameters into the variable space
        globals().update(self.__dict__)
        
        time = price_ts.index

        # time set with an additional time slot for the last soc
        SOCtime = time.append(pd.Index([time[-1] + pd.Timedelta('1hour')]))

        
        model = pyo.ConcreteModel()
        
        ## Variables ##
        model.IDX1 = range(len(time))
        model.IDX2 = range(1)
        model.IDX3 = range(hpp_lifetime)
        model.IDX4 = range(len(SOCtime))
        
        model.P_HPP_t = pyo.Var(model.IDX1, domain=pyo.NonNegativeReals, bounds=(0,hpp_grid_connection))
        model.P_curtailment_t = pyo.Var(model.IDX1, domain=pyo.NonNegativeReals)
        model.Wind_MW = pyo.Var(model.IDX2, domain=pyo.NonNegativeIntegers)
        model.Solar_MW = pyo.Var(model.IDX2, domain=pyo.NonNegativeIntegers)
        # Power charge/discharge from battery 
        # Lower bound as large negative number in order to allow the variable to
        # have either positive or negative values
        model.P_charge_discharge_t = pyo.Var(model.IDX1, bounds=(-10000,10000))
        # Battery rated energy capacity
        model.E_batt_MWh = pyo.Var(model.IDX2, domain=pyo.NonNegativeIntegers, bounds=(0, 10000))
        # Battery rated power capacity
        model.P_batt_MW = pyo.Var(model.IDX2, domain=pyo.NonNegativeIntegers, bounds=(0, 10000))
        # Battery energy level
        model.E_SOC_t = pyo.Var(model.IDX4, domain=pyo.NonNegativeReals, bounds=(0,10000))
        
        
        ## Constraints ##
        model.curtailment_constraint = pyo.ConstraintList()
        model.power_constraint = pyo.ConstraintList()
        model.charge_discharge_constraint = pyo.ConstraintList()
        model.battery_energy_constraint = pyo.ConstraintList()
        model.battery_energy_min_constraint = pyo.ConstraintList()
        model.battery_dynamics_constraint = pyo.ConstraintList()
        
        
        model.SOC_initial_condition = pyo.Constraint(expr = model.E_SOC_t[0] == 0.5 * model.E_batt_MWh[0])
        
        # SOC at the end of the year has to be equal to SOC at the beginning of the year
        model.SOC_final = pyo.Constraint(expr = model.E_SOC_t[len(time) - 1] == 0.5 * model.E_batt_MWh[0])
        
        # Delta_t of 1 hour
        dt = 1
                
        for t in range(0,len(time)):
            # Battery charge/discharge within its power rating
            model.charge_discharge_constraint.add(model.P_charge_discharge_t[t] <= model.P_batt_MW[0])
            model.charge_discharge_constraint.add(model.P_charge_discharge_t[t] >= -model.P_batt_MW[0])
            
            # Constraining battery energy level to maximum battery level
            model.battery_energy_constraint.add(model.E_SOC_t[t] <= model.E_batt_MWh[0])
            # Constraining battery energy level to minimum battery level
            model.battery_energy_min_constraint.add(model.E_SOC_t[t] >= \
                    (1 - battery_depth_of_discharge) * model.E_batt_MWh[0])
            # print(battery_depth_of_discharge)
            
            # Power constraint
            model.power_constraint.add(model.P_HPP_t[t] == wind_ts[t] * model.Wind_MW[0] +\
                solar_ts[t] * model.Solar_MW[0] -\
                model.P_curtailment_t[t] + model.P_charge_discharge_t[t])
        
        # Battery dynamics
        for t in range(1,len(time)):
            model.battery_dynamics_constraint.add(model.E_SOC_t[t] == model.E_SOC_t[t-1] -\
                 model.P_charge_discharge_t[t] * dt)
         
        # Objective Function ##
        model.OBJ = pyo.Objective( expr = 
            # CAPEX
            -(wind_turbine_cost * model.Wind_MW[0] + \
              solar_PV_cost * model.Solar_MW[0] + \
              hpp_BOS_soft_cost * (model.Wind_MW[0] + model.Solar_MW[0]) + \
              hpp_grid_connection_cost * hpp_grid_connection + \
              solar_hardware_installation_cost * model.Solar_MW[0] + \
              wind_civil_works_cost * model.Wind_MW[0] + \
              battery_energy_cost * model.E_batt_MWh[0] + \
              (battery_power_cost + battery_BOP_installation_commissioning_cost + \
               battery_control_system_cost) * model.P_batt_MW[0]) + \
                # revenues and OPEX
                sum((sum(price_ts[t] * model.P_HPP_t[t] for t in model.IDX1) - \
                    (wind_fixed_onm_cost * model.Wind_MW[0] + \
                    solar_fixed_onm_cost * model.Solar_MW[0] + \
                    battery_energy_onm_cost * model.E_batt_MWh[0])) / \
                    np.power(1 + hpp_discount_factor,(i+1)) \
                    for i in model.IDX3) \
                    , sense=pyo.maximize \
            )
            
        
        opt = pyo.SolverFactory('glpk')
        results = opt.solve(model, tee=False)
        results.write()
        
        ## Return calculated results ##
        P_curtailment_ts = []
        P_HPP_ts = []    
        P_charge_discharge_ts = []
        E_SOC_ts = []
        
        for count in range(0, len(time)):
            P_curtailment_ts.append(model.P_curtailment_t[count]())
            P_HPP_ts.append(model.P_HPP_t[count]())
            P_charge_discharge_ts.append(model.P_charge_discharge_t[count]())
            E_SOC_ts.append(model.E_SOC_t[count]())
        
        

        P_RES_available_ts = wind_ts * \
            model.Wind_MW[0]() + solar_ts * model.Solar_MW[0]()

       
        AEP = sum(P_HPP_ts)
        AEP_per_year = np.ones(hpp_lifetime) * AEP

        # Investment cost
        hpp_investment_cost = \
            wind_turbine_cost * model.Wind_MW[0]() + \
            solar_PV_cost * model.Solar_MW[0]() + \
            battery_energy_cost * model.E_batt_MWh[0]() + \
            (battery_power_cost +
              battery_BOP_installation_commissioning_cost +
              battery_control_system_cost) * model.P_batt_MW[0]() + \
            hpp_grid_connection_cost * hpp_grid_connection + \
            hpp_BOS_soft_cost * (model.Wind_MW[0]() +
                                  model.Solar_MW[0]()) + \
            solar_hardware_installation_cost * model.Solar_MW[0]() + \
            wind_civil_works_cost * model.Wind_MW[0]()

        # Maintenance cost
        hpp_maintenance_cost = \
            np.ones(hpp_lifetime) * (
                wind_fixed_onm_cost * model.Wind_MW[0]() +
                solar_fixed_onm_cost * model.Solar_MW[0]() +
                battery_energy_onm_cost * model.E_batt_MWh[0]())

        LCOE = self.calculate_LCOE(
            hpp_investment_cost, hpp_maintenance_cost,
            hpp_discount_factor, hpp_lifetime, AEP_per_year)

       
        
        IRR = self.calculate_IRR(
            P_HPP_ts, price_ts, hpp_investment_cost,hpp_maintenance_cost)
        
        NPV = self.calculate_NPV(
            P_HPP_ts, price_ts, hpp_investment_cost,hpp_maintenance_cost, \
                hpp_discount_factor)

        return model.Wind_MW[0](), model.Solar_MW[0](), model.P_batt_MW[0](), \
            model.E_batt_MWh[0](), P_RES_available_ts, P_HPP_ts, \
                P_curtailment_ts, P_charge_discharge_ts, E_SOC_ts, \
                hpp_investment_cost, hpp_maintenance_cost, LCOE, NPV, IRR


    def sizing_Wind_Battery_Pyomo(self, wind_ts, solar_ts, price_ts):
        """
        Method to calculate sizing of wind and solar and battery

        Returns
        -------
        Capacity of Wind Power
        Capacity of Solar Power
        Capacity of Batteru
        HPP power output timeseries
        HPP power curtailment timeseries
        HPP total CAPEX
        HPP total OPEX
        Levelised cost of energy
        """

        # extract parameters into the variable space
        globals().update(self.__dict__)
        
        time = price_ts.index

        # time set with an additional time slot for the last soc
        SOCtime = time.append(pd.Index([time[-1] + pd.Timedelta('1hour')]))

        
        model = pyo.ConcreteModel()
        
        ## Variables ##
        model.IDX1 = range(len(time))
        model.IDX2 = range(1)
        model.IDX3 = range(hpp_lifetime)
        model.IDX4 = range(len(SOCtime))
        
        model.P_HPP_t = pyo.Var(model.IDX1, domain=pyo.NonNegativeReals, bounds=(0,hpp_grid_connection))
        model.P_curtailment_t = pyo.Var(model.IDX1, domain=pyo.NonNegativeReals)
        model.Wind_MW = pyo.Var(model.IDX2, domain=pyo.NonNegativeIntegers)
        model.Solar_MW = pyo.Var(model.IDX2, domain=pyo.NonNegativeIntegers)
        # Power charge/discharge from battery 
        # Lower bound as large negative number in order to allow the variable to
        # have either positive or negative values
        model.P_charge_discharge_t = pyo.Var(model.IDX1, bounds=(-10000,10000))
        # Battery rated energy capacity
        model.E_batt_MWh = pyo.Var(model.IDX2, domain=pyo.NonNegativeIntegers, bounds=(0, 10000))
        # Battery rated power capacity
        model.P_batt_MW = pyo.Var(model.IDX2, domain=pyo.NonNegativeIntegers, bounds=(0, 10000))
        # Battery energy level
        model.E_SOC_t = pyo.Var(model.IDX4, domain=pyo.NonNegativeReals, bounds=(0,10000))
        
        
        ## Constraints ##

        #Frocing solar to 0 
        model.solar_constraint = pyo.ConstraintList()
        model.solar_constraint.add(model.Solar_MW[0] == 0)

        model.curtailment_constraint = pyo.ConstraintList()
        model.power_constraint = pyo.ConstraintList()
        model.charge_discharge_constraint = pyo.ConstraintList()
        model.battery_energy_constraint = pyo.ConstraintList()
        model.battery_energy_min_constraint = pyo.ConstraintList()
        model.battery_dynamics_constraint = pyo.ConstraintList()
        

        
        model.SOC_initial_condition = pyo.Constraint(expr = model.E_SOC_t[0] == 0.5 * model.E_batt_MWh[0])
        
        # SOC at the end of the year has to be equal to SOC at the beginning of the year
        model.SOC_final = pyo.Constraint(expr = model.E_SOC_t[len(time) - 1] == 0.5 * model.E_batt_MWh[0])
        
        # Delta_t of 1 hour
        dt = 1
                
        for t in range(0,len(time)):
            # Battery charge/discharge within its power rating
            model.charge_discharge_constraint.add(model.P_charge_discharge_t[t] <= model.P_batt_MW[0])
            model.charge_discharge_constraint.add(model.P_charge_discharge_t[t] >= -model.P_batt_MW[0])
            
            # Constraining battery energy level to maximum battery level
            model.battery_energy_constraint.add(model.E_SOC_t[t] <= model.E_batt_MWh[0])
            # Constraining battery energy level to minimum battery level
            model.battery_energy_min_constraint.add(model.E_SOC_t[t] >= \
                    (1 - battery_depth_of_discharge) * model.E_batt_MWh[0])
            # print(battery_depth_of_discharge)
            
            # Power constraint
            model.power_constraint.add(model.P_HPP_t[t] == wind_ts[t] * model.Wind_MW[0] +\
                solar_ts[t] * model.Solar_MW[0] -\
                model.P_curtailment_t[t] + model.P_charge_discharge_t[t])
        
        # Battery dynamics
        for t in range(1,len(time)):
            model.battery_dynamics_constraint.add(model.E_SOC_t[t] == model.E_SOC_t[t-1] -\
                 model.P_charge_discharge_t[t] * dt)
         
        # Objective Function ##
        model.OBJ = pyo.Objective( expr = 
            # CAPEX
            -(wind_turbine_cost * model.Wind_MW[0] + \
              solar_PV_cost * model.Solar_MW[0] + \
              hpp_BOS_soft_cost * (model.Wind_MW[0] + model.Solar_MW[0]) + \
              hpp_grid_connection_cost * hpp_grid_connection + \
              solar_hardware_installation_cost * model.Solar_MW[0] + \
              wind_civil_works_cost * model.Wind_MW[0] + \
              battery_energy_cost * model.E_batt_MWh[0] + \
              (battery_power_cost + battery_BOP_installation_commissioning_cost + \
               battery_control_system_cost) * model.P_batt_MW[0]) + \
                # revenues and OPEX
                sum((sum(price_ts[t] * model.P_HPP_t[t] for t in model.IDX1) - \
                    (wind_fixed_onm_cost * model.Wind_MW[0] + \
                    solar_fixed_onm_cost * model.Solar_MW[0] + \
                    battery_energy_onm_cost * model.E_batt_MWh[0])) / \
                    np.power(1 + hpp_discount_factor,(i+1)) \
                    for i in model.IDX3) \
                    , sense=pyo.maximize \
            )
            
        
        opt = pyo.SolverFactory('glpk')
        results = opt.solve(model, tee=False)
        results.write()
        
        ## Return calculated results ##
        P_curtailment_ts = []
        P_HPP_ts = []    
        P_charge_discharge_ts = []
        E_SOC_ts = []
        
        for count in range(0, len(time)):
            P_curtailment_ts.append(model.P_curtailment_t[count]())
            P_HPP_ts.append(model.P_HPP_t[count]())
            P_charge_discharge_ts.append(model.P_charge_discharge_t[count]())
            E_SOC_ts.append(model.E_SOC_t[count]())
        
        

        P_RES_available_ts = wind_ts * \
            model.Wind_MW[0]() + solar_ts * model.Solar_MW[0]()

       
        AEP = sum(P_HPP_ts)
        AEP_per_year = np.ones(hpp_lifetime) * AEP

        # Investment cost
        hpp_investment_cost = \
            wind_turbine_cost * model.Wind_MW[0]() + \
            solar_PV_cost * model.Solar_MW[0]() + \
            battery_energy_cost * model.E_batt_MWh[0]() + \
            (battery_power_cost +
              battery_BOP_installation_commissioning_cost +
              battery_control_system_cost) * model.P_batt_MW[0]() + \
            hpp_grid_connection_cost * hpp_grid_connection + \
            hpp_BOS_soft_cost * (model.Wind_MW[0]() +
                                  model.Solar_MW[0]()) + \
            solar_hardware_installation_cost * model.Solar_MW[0]() + \
            wind_civil_works_cost * model.Wind_MW[0]()

        # Maintenance cost
        hpp_maintenance_cost = \
            np.ones(hpp_lifetime) * (
                wind_fixed_onm_cost * model.Wind_MW[0]() +
                solar_fixed_onm_cost * model.Solar_MW[0]() +
                battery_energy_onm_cost * model.E_batt_MWh[0]())

        LCOE = self.calculate_LCOE(
            hpp_investment_cost, hpp_maintenance_cost,
            hpp_discount_factor, hpp_lifetime, AEP_per_year)

       
        
        IRR = self.calculate_IRR(
            P_HPP_ts, price_ts, hpp_investment_cost,hpp_maintenance_cost)
        
        NPV = self.calculate_NPV(
            P_HPP_ts, price_ts, hpp_investment_cost,hpp_maintenance_cost, \
                hpp_discount_factor)

        return model.Wind_MW[0](), model.Solar_MW[0](), model.P_batt_MW[0](), \
            model.E_batt_MWh[0](), P_RES_available_ts, P_HPP_ts, \
                P_curtailment_ts, P_charge_discharge_ts, E_SOC_ts, \
                hpp_investment_cost, hpp_maintenance_cost, LCOE, NPV, IRR

    def sizing_Solar_Battery_Pyomo(self, wind_ts, solar_ts, price_ts):
        """
        Method to calculate sizing of wind and solar and battery

        Returns
        -------
        Capacity of Wind Power
        Capacity of Solar Power
        Capacity of Batteru
        HPP power output timeseries
        HPP power curtailment timeseries
        HPP total CAPEX
        HPP total OPEX
        Levelised cost of energy
        """

        # extract parameters into the variable space
        globals().update(self.__dict__)
        
        time = price_ts.index

        # time set with an additional time slot for the last soc
        SOCtime = time.append(pd.Index([time[-1] + pd.Timedelta('1hour')]))

        
        model = pyo.ConcreteModel()
        
        ## Variables ##
        model.IDX1 = range(len(time))
        model.IDX2 = range(1)
        model.IDX3 = range(hpp_lifetime)
        model.IDX4 = range(len(SOCtime))
        
        model.P_HPP_t = pyo.Var(model.IDX1, domain=pyo.NonNegativeReals, bounds=(0,hpp_grid_connection))
        model.P_curtailment_t = pyo.Var(model.IDX1, domain=pyo.NonNegativeReals)
        model.Wind_MW = pyo.Var(model.IDX2, domain=pyo.NonNegativeIntegers)
        model.Solar_MW = pyo.Var(model.IDX2, domain=pyo.NonNegativeIntegers)
        # Power charge/discharge from battery 
        # Lower bound as large negative number in order to allow the variable to
        # have either positive or negative values
        model.P_charge_discharge_t = pyo.Var(model.IDX1, bounds=(-10000,10000))
        # Battery rated energy capacity
        model.E_batt_MWh = pyo.Var(model.IDX2, domain=pyo.NonNegativeIntegers, bounds=(0, 10000))
        # Battery rated power capacity
        model.P_batt_MW = pyo.Var(model.IDX2, domain=pyo.NonNegativeIntegers, bounds=(0, 10000))
        # Battery energy level
        model.E_SOC_t = pyo.Var(model.IDX4, domain=pyo.NonNegativeReals, bounds=(0,10000))
        
        
        ## Constraints ##

        #Frocing wind to 0 
        model.wind_constraint = pyo.ConstraintList()
        model.wind_constraint.add(model.Wind_MW[0] == 0)

        model.curtailment_constraint = pyo.ConstraintList()
        model.power_constraint = pyo.ConstraintList()
        model.charge_discharge_constraint = pyo.ConstraintList()
        model.battery_energy_constraint = pyo.ConstraintList()
        model.battery_energy_min_constraint = pyo.ConstraintList()
        model.battery_dynamics_constraint = pyo.ConstraintList()
        

        
        model.SOC_initial_condition = pyo.Constraint(expr = model.E_SOC_t[0] == 0.5 * model.E_batt_MWh[0])
        
        # SOC at the end of the year has to be equal to SOC at the beginning of the year
        model.SOC_final = pyo.Constraint(expr = model.E_SOC_t[len(time) - 1] == 0.5 * model.E_batt_MWh[0])
        
        # Delta_t of 1 hour
        dt = 1
                
        for t in range(0,len(time)):
            # Battery charge/discharge within its power rating
            model.charge_discharge_constraint.add(model.P_charge_discharge_t[t] <= model.P_batt_MW[0])
            model.charge_discharge_constraint.add(model.P_charge_discharge_t[t] >= -model.P_batt_MW[0])
            
            # Constraining battery energy level to maximum battery level
            model.battery_energy_constraint.add(model.E_SOC_t[t] <= model.E_batt_MWh[0])
            # Constraining battery energy level to minimum battery level
            model.battery_energy_min_constraint.add(model.E_SOC_t[t] >= \
                    (1 - battery_depth_of_discharge) * model.E_batt_MWh[0])
            # print(battery_depth_of_discharge)
            
            # Power constraint
            model.power_constraint.add(model.P_HPP_t[t] == wind_ts[t] * model.Wind_MW[0] +\
                solar_ts[t] * model.Solar_MW[0] -\
                model.P_curtailment_t[t] + model.P_charge_discharge_t[t])
        
        # Battery dynamics
        for t in range(1,len(time)):
            model.battery_dynamics_constraint.add(model.E_SOC_t[t] == model.E_SOC_t[t-1] -\
                 model.P_charge_discharge_t[t] * dt)
         
        # Objective Function ##
        model.OBJ = pyo.Objective( expr = 
            # CAPEX
            -(wind_turbine_cost * model.Wind_MW[0] + \
              solar_PV_cost * model.Solar_MW[0] + \
              hpp_BOS_soft_cost * (model.Wind_MW[0] + model.Solar_MW[0]) + \
              hpp_grid_connection_cost * hpp_grid_connection + \
              solar_hardware_installation_cost * model.Solar_MW[0] + \
              wind_civil_works_cost * model.Wind_MW[0] + \
              battery_energy_cost * model.E_batt_MWh[0] + \
              (battery_power_cost + battery_BOP_installation_commissioning_cost + \
               battery_control_system_cost) * model.P_batt_MW[0]) + \
                # revenues and OPEX
                sum((sum(price_ts[t] * model.P_HPP_t[t] for t in model.IDX1) - \
                    (wind_fixed_onm_cost * model.Wind_MW[0] + \
                    solar_fixed_onm_cost * model.Solar_MW[0] + \
                    battery_energy_onm_cost * model.E_batt_MWh[0])) / \
                    np.power(1 + hpp_discount_factor,(i+1)) \
                    for i in model.IDX3) \
                    , sense=pyo.maximize \
            )
            
        
        opt = pyo.SolverFactory('glpk')
        results = opt.solve(model, tee=False)
        results.write()
        
        ## Return calculated results ##
        P_curtailment_ts = []
        P_HPP_ts = []    
        P_charge_discharge_ts = []
        E_SOC_ts = []
        
        for count in range(0, len(time)):
            P_curtailment_ts.append(model.P_curtailment_t[count]())
            P_HPP_ts.append(model.P_HPP_t[count]())
            P_charge_discharge_ts.append(model.P_charge_discharge_t[count]())
            E_SOC_ts.append(model.E_SOC_t[count]())
        
        

        P_RES_available_ts = wind_ts * \
            model.Wind_MW[0]() + solar_ts * model.Solar_MW[0]()

       
        AEP = sum(P_HPP_ts)
        AEP_per_year = np.ones(hpp_lifetime) * AEP

        # Investment cost
        hpp_investment_cost = \
            wind_turbine_cost * model.Wind_MW[0]() + \
            solar_PV_cost * model.Solar_MW[0]() + \
            battery_energy_cost * model.E_batt_MWh[0]() + \
            (battery_power_cost +
              battery_BOP_installation_commissioning_cost +
              battery_control_system_cost) * model.P_batt_MW[0]() + \
            hpp_grid_connection_cost * hpp_grid_connection + \
            hpp_BOS_soft_cost * (model.Wind_MW[0]() +
                                  model.Solar_MW[0]()) + \
            solar_hardware_installation_cost * model.Solar_MW[0]() + \
            wind_civil_works_cost * model.Wind_MW[0]()

        # Maintenance cost
        hpp_maintenance_cost = \
            np.ones(hpp_lifetime) * (
                wind_fixed_onm_cost * model.Wind_MW[0]() +
                solar_fixed_onm_cost * model.Solar_MW[0]() +
                battery_energy_onm_cost * model.E_batt_MWh[0]())

        LCOE = self.calculate_LCOE(
            hpp_investment_cost, hpp_maintenance_cost,
            hpp_discount_factor, hpp_lifetime, AEP_per_year)

       
        
        IRR = self.calculate_IRR(
            P_HPP_ts, price_ts, hpp_investment_cost,hpp_maintenance_cost)
        
        NPV = self.calculate_NPV(
            P_HPP_ts, price_ts, hpp_investment_cost,hpp_maintenance_cost, \
                hpp_discount_factor)

        return model.Wind_MW[0](), model.Solar_MW[0](), model.P_batt_MW[0](), \
            model.E_batt_MWh[0](), P_RES_available_ts, P_HPP_ts, \
                P_curtailment_ts, P_charge_discharge_ts, E_SOC_ts, \
                hpp_investment_cost, hpp_maintenance_cost, LCOE, NPV, IRR



    def calculate_LCOE(self,
                       investment_cost, maintenance_cost_per_year,
                       discount_rate, lifetime, AEP_per_year):
        LCOE = (investment_cost + sum((maintenance_cost_per_year[i]) / np.power(1 + discount_rate, i) for i in range(
            lifetime))) / sum(AEP_per_year[i] / np.power(1 + discount_rate, i) for i in range(lifetime))
        return LCOE

    def calculate_IRR(
            self,
            P_HPP_t,
            price_t,
            investment_cost,
            maintenance_cost_per_year):
        time = price_t.index
        Revenue_t = sum(price_t[t] * P_HPP_t[t] for t in range(0, len(time)))
        Profit_t = Revenue_t - maintenance_cost_per_year
        # print(Profit_t)
        # print(investment_cost)
        Cashflow = np.insert(Profit_t, 0, -investment_cost)
        IRR = npf.irr(Cashflow)
        return IRR

    def calculate_NPV(
            self,
            P_HPP_t,
            price_t,
            investment_cost,
            maintenance_cost_per_year,
            discount_rate):
        time = price_t.index
        Revenue_t = sum(price_t[t] * P_HPP_t[t] for t in range(0,len(time)))
        Profit_t = Revenue_t - maintenance_cost_per_year
        Cashflow = np.insert(Profit_t, 0, -investment_cost)
        # print(Cashflow)
        NPV = npf.npv(discount_rate, Cashflow)
        return NPV


def read_csv_Data(
    input_ts_filename,
    timename,
    timeFormat,
    timeZone,
    timeZone_analysis,
):
    """
    Method for reading timeseries csv files;
    convert the timeseries into specified time format;

    Output
    ------
    data: Formatted timeseries

    """
    data = pd.read_csv(input_ts_filename)
    data[timename] = pd.to_datetime(data[timename], format=timeFormat)
    data = data.set_index(timename, drop=True)

    ts = data.tz_localize(
        timeZone, ambiguous='infer').tz_convert(
        timeZone_analysis).tz_localize(None)

    return ts
