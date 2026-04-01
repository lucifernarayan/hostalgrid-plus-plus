# simulation/student.py

import random

class Student:
    def __init__(self, student_id, room_id):
        self.student_id = student_id
        self.room_id = room_id
        self.comfort_threshold = random.uniform(24.0, 28.0)
        self.complaint_count = 0

    def check_comfort(self, room_temperature, ac_on, lights_on):
        complaining = False

        if room_temperature > self.comfort_threshold and not ac_on:
            self.complaint_count += 1
            complaining = True

        if not lights_on:
            self.complaint_count += 1
            complaining = True

        return complaining

    def __repr__(self):
        return (
            f"Student {self.student_id} | "
            f"Room {self.room_id} | "
            f"Threshold: {self.comfort_threshold:.1f}C | "
            f"Complaints: {self.complaint_count}"
        )