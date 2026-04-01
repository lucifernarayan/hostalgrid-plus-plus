# simulation/hostel.py

import random

class Room:
    def __init__(self, room_id):
        self.room_id = room_id
        self.is_occupied = random.choice([True, False])
        self.temperature = random.uniform(22.0, 30.0)
        self.ac_on = self.is_occupied
        self.lights_on = self.is_occupied
        self.power_consumption = 0.0

    def update(self):
        """Recalculate power consumption based on current state"""
        self.power_consumption = 0.0
        if self.ac_on:
            self.power_consumption += 1.5       # AC = 1.5 kW
        if self.lights_on:
            self.power_consumption += 0.1       # Lights = 100W
        return self.power_consumption

    def __repr__(self):
        return (f"Room {self.room_id} | "
                f"Occupied: {self.is_occupied} | "
                f"Temp: {self.temperature:.1f}°C | "
                f"Power: {self.power_consumption:.2f} kW")
    
    # simulation/hostel.py (continued — same file)

class Hostel:
    def __init__(self, num_rooms=20):
        self.num_rooms = num_rooms
        self.rooms = [Room(i) for i in range(num_rooms)]
        self.total_power = 0.0
        self.total_complaints = 0

    def update_all_rooms(self):
        """Update every room and calculate total power"""
        self.total_power = sum(room.update() for room in self.rooms)
        return self.total_power

    def get_occupancy(self):
        return [1 if room.is_occupied else 0 for room in self.rooms]

    def get_temperatures(self):
        return [room.temperature for room in self.rooms]

    def get_total_power(self):
        return round(self.total_power, 3)

    def simulate_complaints(self):
        """
        Students complain if:
        - AC is off but room is occupied and hot
        - Lights are off but room is occupied
        """
        complaints = 0
        for room in self.rooms:
            if room.is_occupied:
                if not room.ac_on and room.temperature > 27:
                    complaints += 1
                if not room.lights_on:
                    complaints += 1
        self.total_complaints = complaints
        return complaints

    def summary(self):
        print(f"\n🏨 Hostel Summary")
        print(f"   Total Rooms    : {self.num_rooms}")
        print(f"   Occupied       : {sum(self.get_occupancy())}")
        print(f"   Total Power    : {self.get_total_power()} kW")
        print(f"   Complaints     : {self.simulate_complaints()}")
        print(f"   Avg Temp       : {sum(self.get_temperatures())/self.num_rooms:.1f}°C")