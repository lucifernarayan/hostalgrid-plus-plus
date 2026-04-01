# simulation/appliances.py

import random

class Appliance:
    def __init__(self, name, power_kw, deferrable=False):
        self.name = name
        self.power_kw = power_kw
        self.deferrable = deferrable   # can it be shifted to off-peak?
        self.is_on = False
        self.deferred = False

    def turn_on(self):
        self.is_on = True
        self.deferred = False

    def turn_off(self):
        self.is_on = False

    def defer(self):
        """Shift this appliance to off-peak hours"""
        if self.deferrable:
            self.is_on = False
            self.deferred = True

    def get_power(self):
        return self.power_kw if self.is_on else 0.0

    def __repr__(self):
        status = "ON" if self.is_on else ("DEFERRED" if self.deferred else "OFF")
        return f"{self.name} | {self.power_kw} kW | {status}"


class ApplianceManager:
    def __init__(self):
        self.appliances = [
            Appliance("Washing Machine", power_kw=2.0, deferrable=True),
            Appliance("Water Heater",    power_kw=3.0, deferrable=True),
            Appliance("Common AC",       power_kw=2.5, deferrable=False),
            Appliance("Gym Equipment",   power_kw=1.5, deferrable=True),
            Appliance("Kitchen",         power_kw=2.0, deferrable=False),
        ]

    def total_power(self):
        return round(sum(a.get_power() for a in self.appliances), 3)

    def defer_all_deferrable(self):
        """Defer all shiftable appliances — called during peak hours"""
        for a in self.appliances:
            if a.deferrable:
                a.defer()

    def turn_on_all(self):
        for a in self.appliances:
            a.turn_on()

    def summary(self):
        print("\n⚡ Appliance Status")
        for a in self.appliances:
            print(f"   {a}")
        print(f"   Total Load: {self.total_power()} kW")