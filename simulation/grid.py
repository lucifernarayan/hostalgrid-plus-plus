# simulation/grid.py

class Grid:
    def __init__(self):
        # Electricity tariff per hour (peak/off-peak)
        self.tariff_schedule = self._build_tariff()
        self.carbon_schedule = self._build_carbon()

    def _build_tariff(self):
        """Rs per kWh at each hour (0-23)"""
        tariff = {}
        for hour in range(24):
            if 9 <= hour <= 12 or 18 <= hour <= 22:
                tariff[hour] = 8.5    # peak hours — expensive
            elif 0 <= hour <= 5:
                tariff[hour] = 4.0    # night — cheap
            else:
                tariff[hour] = 6.0    # normal
        return tariff

    def _build_carbon(self):
        """gCO2 per kWh at each hour"""
        carbon = {}
        for hour in range(24):
            if 9 <= hour <= 12 or 18 <= hour <= 22:
                carbon[hour] = 0.82   # peak = more thermal = more carbon
            elif 0 <= hour <= 5:
                carbon[hour] = 0.45   # night = more renewables
            else:
                carbon[hour] = 0.63
        return carbon

    def get_tariff(self, hour):
        return self.tariff_schedule.get(hour % 24, 6.0)

    def get_carbon_rate(self, hour):
        return self.carbon_schedule.get(hour % 24, 0.63)

    def get_cost(self, power_kw, hour):
        """Cost for using power_kw for 1 hour"""
        return round(power_kw * self.get_tariff(hour), 4)

    def summary(self, hour):
        print(f"\n🔌 Grid at Hour {hour}")
        print(f"   Tariff      : Rs {self.get_tariff(hour)}/kWh")
        print(f"   Carbon Rate : {self.get_carbon_rate(hour)} gCO2/kWh")

