import pulp

# Hub capacities (tons per month)
hubs = {
    "CVG": 95650,
    "AFW": 44350,
}

# Focus city capacities (tons per month)
focus = {
    "Leipzig": 85000,
    "Hyderabad": 19000,
    "San Bernadino": 36000,
}

# Destination center demands (tons per month)
centers_demand = {
    "Paris": 6500,
    "Cologne": 640,
    "Hanover": 180,
    "Bengaluru": 9100,
    "Coimbatore": 570,
    "Delhi": 19000,
    "Mumbai": 14800,
    "Cagliari": 90,
    "Catania": 185,
    "Milan": 800,
    "Rome": 1700,
    "Katowice": 170,
    "Barcelona": 2800,
    "Madrid": 3700,
    "Castle Donington": 30,
    "London": 6700,
    "Mobile": 190,
    "Anchorage": 175,
    "Fairbanks": 38,
    "Phoenix": 2400,
    "Los Angeles": 7200,
    "Ontario": 100,
    "Riverside": 1200,
    "Sacramento": 1100,
    "San Francisco": 1900,
    "Stockton": 240,
    "Denver": 1500,
    "Hartford": 540,
    "Miami": 3400,
    "Lakeland": 185,
    "Tampa": 1600,
    "Atlanta": 3000,
    "Honolulu": 500,
    "Kahului/Maui": 16,
    "Kona": 63,
    "Chicago": 5100,
    "Rockford": 172,
    "Fort Wayne": 200,
    "South Bend": 173,
    "Des Moines": 300,
    "Wichita": 290,
    "New Orleans": 550,
    "Baltimore": 1300,
    "Minneapolis": 1700,
    "Kansas City": 975,
    "St. Louis": 1200,
    "Omaha": 480,
    "Manchester": 100,
    "Albuquerque": 450,
    "New York": 11200,
    "Charlotte": 900,
    "Toledo": 290,
    "Wilmington": 150,
    "Portland": 1200,
    "Allentown": 420,
    "Pittsburgh": 1000,
    "San Juan": 1100,
    "Nashville": 650,
    "Austin": 975,
    "Dallas": 3300,
    "Houston": 3300,
    "San Antonio": 1100,
    "Richmond": 600,
    "Seattle/Tacoma": 2000,
    "Spokane": 260,
}

hubs_list = list(hubs.keys())
focus_list = list(focus.keys())
centers_list = list(centers_demand.keys())

# 2. Route costs (per ton)
route_cost = {
    # Focus cities rows
    ("CVG", "Leipzig"): 1.5,
    ("Leipzig", "Hyderabad"): 1.6,
    ("CVG", "San Bernadino"): 0.5,
    ("AFW", "San Bernadino"): 0.5,

    # Centers rows
    ("CVG", "Paris"): 1.6,
    ("Leipzig", "Paris"): 0.5,
    ("Hyderabad", "Paris"): 1.1,

    ("CVG", "Cologne"): 1.5,
    ("Leipzig", "Cologne"): 0.5,
    ("Hyderabad", "Cologne"): 1.0,

    ("CVG", "Hanover"): 1.5,
    ("Leipzig", "Hanover"): 0.5,
    ("Hyderabad", "Hanover"): 1.0,

    ("Leipzig", "Bengaluru"): 1.5,
    ("Hyderabad", "Bengaluru"): 0.5,

    ("Leipzig", "Coimbatore"): 1.5,
    ("Hyderabad", "Coimbatore"): 0.5,

    ("Leipzig", "Delhi"): 1.5,
    ("Hyderabad", "Delhi"): 0.5,

    ("Leipzig", "Mumbai"): 1.5,
    ("Hyderabad", "Mumbai"): 0.5,

    ("CVG", "Cagliari"): 1.5,
    ("Leipzig", "Cagliari"): 0.5,
    ("Hyderabad", "Cagliari"): 1.0,

    ("CVG", "Catania"): 1.5,
    ("Leipzig", "Catania"): 0.5,
    ("Hyderabad", "Catania"): 1.0,

    ("CVG", "Milan"): 1.5,
    ("Leipzig", "Milan"): 0.5,
    ("Hyderabad", "Milan"): 1.0,

    ("CVG", "Rome"): 1.5,
    ("Leipzig", "Rome"): 0.5,
    ("Hyderabad", "Rome"): 1.1,

    ("CVG", "Katowice"): 1.4,
    ("Leipzig", "Katowice"): 0.5,
    ("Hyderabad", "Katowice"): 1.0,

    ("CVG", "Barcelona"): 1.5,
    ("Leipzig", "Barcelona"): 0.5,
    ("Hyderabad", "Barcelona"): 1.0,

    ("CVG", "Madrid"): 1.6,
    ("Leipzig", "Madrid"): 0.5,
    ("Hyderabad", "Madrid"): 1.1,

    ("CVG", "Castle Donington"): 1.4,
    ("Leipzig", "Castle Donington"): 0.5,

    ("CVG", "London"): 1.6,
    ("Leipzig", "London"): 0.75,
    ("Hyderabad", "London"): 1.1,

    ("CVG", "Mobile"): 0.5,
    ("AFW", "Mobile"): 0.5,
    ("San Bernadino", "Mobile"): 0.5,

    ("CVG", "Anchorage"): 1.3,
    ("AFW", "Anchorage"): 1.0,
    ("San Bernadino", "Anchorage"): 0.7,

    ("CVG", "Fairbanks"): 1.4,
    ("AFW", "Fairbanks"): 1.0,
    ("San Bernadino", "Fairbanks"): 0.7,

    ("CVG", "Phoenix"): 0.5,
    ("AFW", "Phoenix"): 0.5,
    ("San Bernadino", "Phoenix"): 0.5,

    ("CVG", "Los Angeles"): 0.5,
    ("AFW", "Los Angeles"): 0.5,

    ("CVG", "Ontario"): 0.5,
    ("AFW", "Ontario"): 0.5,

    ("CVG", "Riverside"): 0.5,
    ("AFW", "Riverside"): 0.5,

    ("CVG", "Sacramento"): 0.5,
    ("AFW", "Sacramento"): 0.5,
    ("San Bernadino", "Sacramento"): 0.5,

    ("CVG", "San Francisco"): 0.5,
    ("AFW", "San Francisco"): 0.5,
    ("San Bernadino", "San Francisco"): 0.5,

    ("CVG", "Stockton"): 0.5,
    ("AFW", "Stockton"): 0.5,
    ("San Bernadino", "Stockton"): 0.5,

    ("CVG", "Denver"): 0.5,
    ("AFW", "Denver"): 0.5,
    ("San Bernadino", "Denver"): 0.5,

    ("CVG", "Hartford"): 0.5,
    ("AFW", "Hartford"): 0.5,
    ("Leipzig", "Hartford"): 1.5,
    ("San Bernadino", "Hartford"): 0.5,

    ("CVG", "Miami"): 0.5,
    ("AFW", "Miami"): 0.5,
    ("San Bernadino", "Miami"): 0.7,

    ("CVG", "Lakeland"): 0.5,
    ("AFW", "Lakeland"): 0.5,
    ("San Bernadino", "Lakeland"): 0.7,

    ("CVG", "Tampa"): 0.5,
    ("AFW", "Tampa"): 0.5,
    ("San Bernadino", "Tampa"): 0.7,

    ("CVG", "Atlanta"): 0.5,
    ("AFW", "Atlanta"): 0.5,
    ("San Bernadino", "Atlanta"): 0.5,

    ("CVG", "Honolulu"): 0.5,
    ("AFW", "Honolulu"): 0.5,
    ("San Bernadino", "Honolulu"): 1.0,

    ("CVG", "Kahului/Maui"): 0.5,
    ("AFW", "Kahului/Maui"): 0.5,
    ("San Bernadino", "Kahului/Maui"): 1.0,

    ("CVG", "Kona"): 0.5,
    ("AFW", "Kona"): 0.5,
    ("San Bernadino", "Kona"): 1.0,

    ("CVG", "Chicago"): 0.5,
    ("AFW", "Chicago"): 0.5,
    ("San Bernadino", "Chicago"): 0.5,

    ("CVG", "Rockford"): 0.5,
    ("AFW", "Rockford"): 0.5,
    ("San Bernadino", "Rockford"): 0.7,

    ("CVG", "Fort Wayne"): 0.5,
    ("AFW", "Fort Wayne"): 0.5,
    ("San Bernadino", "Fort Wayne"): 0.5,

    ("CVG", "South Bend"): 0.5,
    ("AFW", "South Bend"): 0.5,
    ("San Bernadino", "South Bend"): 0.5,

    ("CVG", "Des Moines"): 0.5,
    ("AFW", "Des Moines"): 0.5,
    ("San Bernadino", "Des Moines"): 0.5,

    ("CVG", "Wichita"): 0.5,
    ("AFW", "Wichita"): 0.5,
    ("San Bernadino", "Wichita"): 0.5,

    ("CVG", "New Orleans"): 0.5,
    ("AFW", "New Orleans"): 0.5,
    ("San Bernadino", "New Orleans"): 0.5,

    ("CVG", "Baltimore"): 0.5,
    ("AFW", "Baltimore"): 0.5,
    ("Leipzig", "Baltimore"): 1.5,
    ("San Bernadino", "Baltimore"): 0.7,

    ("CVG", "Minneapolis"): 0.5,
    ("AFW", "Minneapolis"): 0.5,
    ("San Bernadino", "Minneapolis"): 0.5,

    ("CVG", "Kansas City"): 0.5,
    ("AFW", "Kansas City"): 0.5,
    ("San Bernadino", "Kansas City"): 0.5,

    ("CVG", "St. Louis"): 0.5,
    ("AFW", "St. Louis"): 0.5,
    ("San Bernadino", "St. Louis"): 0.5,

    ("CVG", "Omaha"): 0.5,
    ("AFW", "Omaha"): 0.5,
    ("San Bernadino", "Omaha"): 0.5,

    ("CVG", "Manchester"): 0.5,
    ("AFW", "Manchester"): 0.5,
    ("Leipzig", "Manchester"): 1.5,
    ("San Bernadino", "Manchester"): 0.7,

    ("CVG", "Albuquerque"): 0.5,
    ("AFW", "Albuquerque"): 0.5,
    ("San Bernadino", "Albuquerque"): 0.5,

    ("CVG", "New York"): 0.5,
    ("AFW", "New York"): 0.5,
    ("Leipzig", "New York"): 1.6,
    ("San Bernadino", "New York"): 0.7,

    ("CVG", "Charlotte"): 0.5,
    ("AFW", "Charlotte"): 0.5,
    ("San Bernadino", "Charlotte"): 0.7,

    ("CVG", "Toledo"): 0.5,
    ("AFW", "Toledo"): 0.5,
    ("San Bernadino", "Toledo"): 0.5,

    ("CVG", "Wilmington"): 0.5,
    ("AFW", "Wilmington"): 0.5,
    ("San Bernadino", "Wilmington"): 0.7,

    ("CVG", "Portland"): 0.5,
    ("AFW", "Portland"): 0.5,
    ("San Bernadino", "Portland"): 0.5,

    ("CVG", "Allentown"): 0.5,
    ("AFW", "Allentown"): 0.5,
    ("Leipzig", "Allentown"): 1.5,
    ("San Bernadino", "Allentown"): 0.7,

    ("CVG", "Pittsburgh"): 0.5,
    ("AFW", "Pittsburgh"): 0.5,
    ("San Bernadino", "Pittsburgh"): 0.6,

    ("CVG", "San Juan"): 0.5,
    ("AFW", "San Juan"): 0.5,
    ("San Bernadino", "San Juan"): 1.0,

    ("CVG", "Nashville"): 0.5,
    ("AFW", "Nashville"): 0.5,
    ("San Bernadino", "Nashville"): 0.5,

    ("CVG", "Austin"): 0.5,
    ("AFW", "Austin"): 0.25,
    ("San Bernadino", "Austin"): 0.5,

    ("CVG", "Dallas"): 0.5,
    ("San Bernadino", "Dallas"): 0.5,

    ("CVG", "Houston"): 0.5,
    ("AFW", "Houston"): 0.25,
    ("San Bernadino", "Houston"): 0.5,

    ("CVG", "San Antonio"): 0.5,
    ("AFW", "San Antonio"): 0.25,
    ("San Bernadino", "San Antonio"): 0.5,

    ("CVG", "Richmond"): 0.5,
    ("AFW", "Richmond"): 0.5,
    ("San Bernadino", "Richmond"): 0.7,

    ("CVG", "Seattle/Tacoma"): 0.5,
    ("AFW", "Seattle/Tacoma"): 0.5,
    ("San Bernadino", "Seattle/Tacoma"): 0.5,

    ("CVG", "Spokane"): 0.5,
    ("AFW", "Spokane"): 0.5,
    ("San Bernadino", "Spokane"): 0.5,
}

# 3. Classify arcs by type (hub->focus, hub->center, focus->center)
route_type = {}
for (s, d) in route_cost:
    if s in hubs_list and d in focus_list:
        route_type[(s, d)] = "hub to focus"
    elif s in hubs_list and d in centers_list:
        route_type[(s, d)] = "hub to center"
    elif s in focus_list and d in centers_list:
        route_type[(s, d)] = "focus to center"
    else:
        route_type[(s, d)] = "other"

hub_to_focus_arcs = [arc for arc, t in route_type.items() if t == "hub to focus"]
hub_to_center_arcs = [arc for arc, t in route_type.items() if t == "hub to center"]
focus_to_center_arcs = [arc for arc, t in route_type.items() if t == "focus to center"]

# 4. Build the linear programming model
model = pulp.LpProblem("Freight_Network_Distribution", pulp.LpMinimize)

# Decision variables:
x = pulp.LpVariable.dicts("x", (hubs_list, focus_list), lowBound=0, cat="Continuous")
y = pulp.LpVariable.dicts("y", (hubs_list, centers_list), lowBound=0, cat="Continuous")
z = pulp.LpVariable.dicts("z", (focus_list, centers_list), lowBound=0, cat="Continuous")

# 5. Objective: minimize total transportation cost
model += (
    pulp.lpSum(route_cost[(i, j)] * x[i][j] for (i, j) in hub_to_focus_arcs)
    + pulp.lpSum(route_cost[(i, k)] * y[i][k] for (i, k) in hub_to_center_arcs)
    + pulp.lpSum(route_cost[(j, k)] * z[j][k] for (j, k) in focus_to_center_arcs),
    "Total_Transportation_Cost"
)

# 6. Constraints
# 6.1 Hub capacity constraints
for i in hubs_list:
    model += (
        pulp.lpSum(x[i][j] for j in focus_list if (i, j) in route_cost)
        + pulp.lpSum(y[i][k] for k in centers_list if (i, k) in route_cost)
        <= hubs[i],
        f"HubCapacity_{i}"
    )

# 6.2 Focus city flow-balance and capacity constraints
for j in focus_list:
    inflow = pulp.lpSum(x[i][j] for i in hubs_list if (i, j) in route_cost)
    outflow = pulp.lpSum(z[j][k] for k in centers_list if (j, k) in route_cost)

    model += (inflow == outflow, f"FlowBalance_{j}")
    model += (inflow <= focus[j], f"FocusCapacity_{j}")

# 6.3 Demand constraints at destination centers
for k in centers_list:
    from_hubs = pulp.lpSum(y[i][k] for i in hubs_list if (i, k) in route_cost)
    from_focus = pulp.lpSum(z[j][k] for j in focus_list if (j, k) in route_cost)
    model += (from_hubs + from_focus == centers_demand[k], f"Demand_{k}")

# 7. Solve the model
solver = pulp.PULP_CBC_CMD(msg=False)
status = model.solve(solver)

print("Solver status:", pulp.LpStatus[status])
print("Optimal total cost:", pulp.value(model.objective))

# 8. Print solution
print("\nShipments from hubs to focus cities (x_ij):")
for i in hubs_list:
    for j in focus_list:
        if (i, j) in route_cost:
            val = pulp.value(x[i][j])
            if val is not None and val > 0:
                print(f"  x[{i},{j}] = {val:.2f} tons")

print("\nShipments from hubs directly to centers (y_ik):")
for i in hubs_list:
    for k in centers_list:
        if (i, k) in route_cost:
            val = pulp.value(y[i][k])
            if val is not None and val > 0:
                print(f"  y[{i},{k}] = {val:.2f} tons")

print("\nShipments from focus cities to centers (z_jk):")
for j in focus_list:
    for k in centers_list:
        if (j, k) in route_cost:
            val = pulp.value(z[j][k])
            if val is not None and val > 0:
                print(f"  z[{j},{k}] = {val:.2f} tons")
