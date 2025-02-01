import pandas as pd
import numpy as np
import folium
from folium import plugins
import geopy.distance
from collections import defaultdict

class DeliveryOptimizer:
    def __init__(self, store_location):
        self.store_location = store_location
        self.vehicle_constraints = {
            '3W': {
                'capacity': 5, 
                'max_radius': 15,  # max round trip distance
                'count': 50,
                'travel_time': 5,   # 5 min per km
                'delivery_time': 10, # 10 min per delivery
                'min_capacity_util': 0.5  # Reduced from 0.5 to allow more flexible usage
            },
            '4W-EV': {
                'capacity': 8, 
                'max_radius': 20,  # max round trip distance
                'count': 25,
                'travel_time': 5,   # 5 min per km
                'delivery_time': 10, # 10 min per delivery
                'min_capacity_util': 0.4  # Reduced from 0.5 to allow more flexible usage
            },
            '4W': {
                'capacity': 25, 
                'max_radius': float('inf'), 
                'count': float('inf'),
                'travel_time': 5,   # 5 min per km
                'delivery_time': 10, # 10 min per delivery
                'min_capacity_util': 0  # No minimum capacity utilization for 4W
            }
        }
        self.vehicle_availability = {
            '3W': [],   
            '4W-EV': [],  
            '4W': []    
        }
        
    def calculate_distance(self, coord1, coord2):
        """Calculate distance between two coordinates in kilometers"""
        return geopy.distance.geodesic(coord1, coord2).km
    
    def is_within_radius(self, shipment, vehicle_type):
        """Check if shipment is within vehicle's maximum radius"""
        distance = self.calculate_distance(
            (shipment['Latitude'], shipment['Longitude']),
            self.store_location
        )
        return distance <= self.vehicle_constraints[vehicle_type]['max_radius']
    
    def get_nearest_neighbor(self, current_coord, available_shipments, vehicle_type):
        """Find the nearest valid shipment respecting vehicle constraints"""
        min_distance = float('inf')
        nearest_shipment = None
        
        for _, shipment in available_shipments.iterrows():
            if not self.is_within_radius(shipment, vehicle_type):
                continue
                
            distance = self.calculate_distance(
                current_coord,
                (shipment['Latitude'], shipment['Longitude'])
            )
            
            if distance < min_distance:
                min_distance = distance
                nearest_shipment = shipment
                
        return nearest_shipment
    
    def create_route(self, available_shipments, vehicle_type):
        """Create a single optimized route with distance and time constraints"""
        if isinstance(available_shipments, list):
            available_shipments = pd.DataFrame(available_shipments)
            
        if available_shipments.empty:
            return None
            
        route_shipments = []
        current_coord = self.store_location
        remaining_shipments = available_shipments.copy()
        capacity = self.vehicle_constraints[vehicle_type]['capacity']
        max_radius = self.vehicle_constraints[vehicle_type]['max_radius']
        min_capacity_util = self.vehicle_constraints[vehicle_type]['min_capacity_util']
        
        total_distance = 0
        total_time = 0
        
        while len(route_shipments) < capacity and not remaining_shipments.empty:
            nearest = self.get_nearest_neighbor(
                current_coord,
                remaining_shipments,
                vehicle_type
            )
            
            if nearest is None:
                break
            
            distance_to_shipment = self.calculate_distance(
                current_coord, 
                (nearest['Latitude'], nearest['Longitude'])
            )
            distance_to_store = self.calculate_distance(
                (nearest['Latitude'], nearest['Longitude']),
                self.store_location
            )
            round_trip_distance = total_distance + distance_to_shipment + distance_to_store
            
            if max_radius != float('inf') and round_trip_distance > max_radius:
                break
            
            total_distance += distance_to_shipment
            total_time += (
                distance_to_shipment * self.vehicle_constraints[vehicle_type]['travel_time'] +
                self.vehicle_constraints[vehicle_type]['delivery_time']
            )
            
            route_shipments.append(nearest)
            current_coord = (nearest['Latitude'], nearest['Longitude'])
            remaining_shipments = remaining_shipments[
                remaining_shipments['Shipment ID'] != nearest['Shipment ID']
            ]
        
        if not route_shipments:
            return None
            
        # Check minimum capacity utilization
        if len(route_shipments) < capacity * min_capacity_util and vehicle_type != '4W':
            return None
            
        # Add return to store distance and time
        total_distance += self.calculate_distance(current_coord, self.store_location)
        total_time += (
            self.calculate_distance(current_coord, self.store_location) * 
            self.vehicle_constraints[vehicle_type]['travel_time']
        )
            
        return {
            'vehicle_type': vehicle_type,
            'shipments': route_shipments,
            'time_slot': route_shipments[0]['Delivery Timeslot'],
            'total_distance': total_distance,
            'total_time': total_time
        }

    def create_delivery_cluster(self, shipments, vehicle_type):
        """Create clusters of nearby deliveries for better vehicle utilization"""
        if shipments.empty:
            return None
            
        max_radius = self.vehicle_constraints[vehicle_type]['max_radius']
        capacity = self.vehicle_constraints[vehicle_type]['capacity']
        
        # Start with the closest shipment
        cluster = []
        current_shipment = shipments.iloc[0]
        cluster.append(current_shipment)
        
        # Find nearby shipments within the radius
        for _, shipment in shipments.iloc[1:].iterrows():
            if len(cluster) >= capacity:
                break
                
            distance = self.calculate_distance(
                (current_shipment['Latitude'], current_shipment['Longitude']),
                (shipment['Latitude'], shipment['Longitude'])
            )
            
            if distance <= max_radius / 2:  # Use half the max radius for clustering
                cluster.append(shipment)
                
        return pd.DataFrame(cluster)
    
    def optimize_deliveries(self, shipments):
        """Create optimized delivery routes with enhanced EV and 3W prioritization"""
        all_routes = []
        vehicle_counts = {k: v['count'] for k, v in self.vehicle_constraints.items()}
        unassigned_shipments = shipments.copy()
        
        # First pass: Try to optimize for 3W and 4W-EV with relaxed constraints
        for time_slot in shipments['Delivery Timeslot'].unique():
            time_slot_shipments = unassigned_shipments[
                unassigned_shipments['Delivery Timeslot'] == time_slot
            ]
            
            # Sort shipments by distance to prioritize closer ones for 3W and 4W-EV
            time_slot_shipments['distance_to_store'] = time_slot_shipments.apply(
                lambda x: self.calculate_distance(
                    (x['Latitude'], x['Longitude']), 
                    self.store_location
                ), 
                axis=1
            )
            time_slot_shipments = time_slot_shipments.sort_values('distance_to_store')
            
            # Try to maximize 3W and 4W-EV usage with cluster-based approach
            for vehicle_type in ['3W', '4W-EV']:
                while (vehicle_counts[vehicle_type] > 0 and 
                      not time_slot_shipments.empty):
                    if self.vehicle_availability[vehicle_type]:
                        vehicle_counts[vehicle_type] += 1
                        self.vehicle_availability[vehicle_type].pop(0)
                    
                    # Create clusters of nearby deliveries
                    cluster = self.create_delivery_cluster(
                        time_slot_shipments, 
                        vehicle_type
                    )
                    
                    if cluster is None or cluster.empty:
                        break
                        
                    route = self.create_route(cluster, vehicle_type)
                    
                    if route is None:
                        break
                    
                    all_routes.append(route)
                    vehicle_counts[vehicle_type] -= 1
                    self.vehicle_availability[vehicle_type].append(route)
                    
                    shipment_ids = [s['Shipment ID'] for s in route['shipments']]
                    time_slot_shipments = time_slot_shipments[
                        ~time_slot_shipments['Shipment ID'].isin(shipment_ids)
                    ]
                    unassigned_shipments = unassigned_shipments[
                        ~unassigned_shipments['Shipment ID'].isin(shipment_ids)
                    ]
        
        # Second pass: Handle remaining shipments with 4W vehicles
        if not unassigned_shipments.empty:
            routes_4w, unassigned = self.handle_remaining_with_4w(unassigned_shipments)
            all_routes.extend(routes_4w)
            unassigned_shipments = unassigned
        
        return all_routes, unassigned_shipments
    
    def handle_remaining_with_4w(self, unassigned_shipments):
        """Handle remaining shipments with 4W vehicles more efficiently"""
        routes = []
        
        for time_slot in unassigned_shipments['Delivery Timeslot'].unique():
            time_slot_shipments = unassigned_shipments[
                unassigned_shipments['Delivery Timeslot'] == time_slot
            ]
            
            while not time_slot_shipments.empty:
                # Try to create maximum capacity routes
                route = self.create_route(
                    time_slot_shipments.head(self.vehicle_constraints['4W']['capacity']), 
                    '4W'
                )
                
                if route is None:
                    break
                    
                routes.append(route)
                shipment_ids = [s['Shipment ID'] for s in route['shipments']]
                time_slot_shipments = time_slot_shipments[
                    ~time_slot_shipments['Shipment ID'].isin(shipment_ids)
                ]
                unassigned_shipments = unassigned_shipments[
                    ~unassigned_shipments['Shipment ID'].isin(shipment_ids)
                ]
        
        return routes, unassigned_shipments

    def visualize_routes_folium(self, routes):
        """Create an interactive map visualization using Folium"""
        m = folium.Map(
            location=[self.store_location[0], self.store_location[1]],
            zoom_start=12,
            tiles='cartodbpositron'
        )

        folium.Marker(
            location=[self.store_location[0], self.store_location[1]],
            popup='Store',
            icon=folium.Icon(color='black', icon='star', prefix='fa')
        ).add_to(m)

        colors = {
            '3W': '#3388ff',
            '4W-EV': '#32CD32',
            '4W': '#DC143C'
        }

        vehicle_groups = {
            vtype: folium.FeatureGroup(name=f'{vtype} Routes')
            for vtype in colors.keys()
        }

        for vehicle_type, constraints in self.vehicle_constraints.items():
            if constraints['max_radius'] != float('inf'):
                folium.Circle(
                    location=[self.store_location[0], self.store_location[1]],
                    radius=constraints['max_radius'] * 1000,
                    color=colors[vehicle_type],
                    fill=False,
                    weight=1,
                    dash_array='5,5',
                    popup=f'{vehicle_type} Range: {constraints["max_radius"]}km'
                ).add_to(vehicle_groups[vehicle_type])

        for i, route in enumerate(routes, 1):
            vehicle_type = route['vehicle_type']
            route_color = colors[vehicle_type]
            
            coords = [(self.store_location[0], self.store_location[1])]
            for shipment in route['shipments']:
                coords.append((shipment['Latitude'], shipment['Longitude']))
            coords.append((self.store_location[0], self.store_location[1]))

            folium.PolyLine(
                locations=coords,
                weight=2,
                color=route_color,
                opacity=0.8,
                popup=f'Route {i} ({vehicle_type})'
            ).add_to(vehicle_groups[vehicle_type])

            for j, shipment in enumerate(route['shipments'], 1):
                folium.CircleMarker(
                    location=[shipment['Latitude'], shipment['Longitude']],
                    radius=8,
                    color=route_color,
                    fill=True,
                    popup=f'Route {i}, Stop {j}<br>ID: {shipment["Shipment ID"]}<br>Time: {shipment["Delivery Timeslot"]}',
                    tooltip=f'Route {i}, Stop {j}'
                ).add_to(vehicle_groups[vehicle_type])

        for group in vehicle_groups.values():
            group.add_to(m)

        folium.LayerControl().add_to(m)
        plugins.Fullscreen().add_to(m)
        plugins.MeasureControl(position='topleft').add_to(m)

        all_coords = []
        for route in routes:
            for shipment in route['shipments']:
                all_coords.append([shipment['Latitude'], shipment['Longitude']])
        if all_coords:
            m.fit_bounds(all_coords)

        return m

    def calculate_route_metrics(self, route):
        """Calculate distance and time metrics for a route"""
        coords = [self.store_location]
        
        for shipment in route['shipments']:
            coords.append((shipment['Latitude'], shipment['Longitude']))
            
        coords.append(self.store_location)
        
        total_distance = 0
        for i in range(len(coords) - 1):
            total_distance += self.calculate_distance(coords[i], coords[i + 1])
            
        travel_time = total_distance / 30
        delivery_time = len(route['shipments']) * 5/60
        total_time = travel_time + delivery_time
        
        return {
            'total_distance': round(total_distance, 2),
            'estimated_time': round(total_time, 2),
            'num_deliveries': len(route['shipments']),
            'shipment_ids': [s['Shipment ID'] for s in route['shipments']]
        }

def save_route_summary(routes, optimizer, filename='route_summary(n5).html'):
    """Generate an HTML summary of all routes with metrics"""
    html_content = """
    <html>
    <head>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .route-card { 
                border: 1px solid #ddd; 
                margin: 10px 0; 
                padding: 15px; 
                border-radius: 5px;
            }
            .vehicle-3W { border-left: 5px solid #3388ff; }
            .vehicle-4W-EV { border-left: 5px solid #32CD32; }
            .vehicle-4W { border-left: 5px solid #DC143C; }
            .metrics { margin-top: 10px; }
            .summary { 
                background-color: #f8f9fa;
                padding: 15px;
                margin-bottom: 20px;
                border-radius: 5px;
            }
        </style>
    </head>
    <body>
    """

    # Add summary section
    vehicle_counts = defaultdict(int)
    total_deliveries = 0
    total_distance = 0

    # Calculate total deliveries and distance
    for route in routes:
        metrics = optimizer.calculate_route_metrics(route)
        vehicle_counts[route['vehicle_type']] += 1
        total_deliveries += metrics['num_deliveries']
        total_distance += metrics['total_distance']

        # Debug: Print each route's delivery count
        print(f"Route {route['vehicle_type']} - Deliveries: {metrics['num_deliveries']}")

    html_content += f"""
    <div class="summary">
        <h2>Route Summary</h2>
        <p>Total Routes: {len(routes)}</p>
        <p>Total Deliveries: {total_deliveries}</p>
        <p>Total Distance: {round(total_distance, 2)} km</p>
        <p>Vehicles Used:</p>
        <ul>
    """

    for vehicle_type, count in vehicle_counts.items():
        html_content += f"<li>{vehicle_type}: {count}</li>"

    html_content += "</ul></div>"

    # Add individual route cards
    for i, route in enumerate(routes, 1):
        metrics = optimizer.calculate_route_metrics(route)
        html_content += f"""
        <div class="route-card vehicle-{route['vehicle_type']}">
            <h3>Route {i}</h3>
            <p>Vehicle Type: {route['vehicle_type']}</p>
            <p>Time Slot: {route['time_slot']}</p>
            <div class="metrics">
                <p>Deliveries: {metrics['num_deliveries']}</p>
                <p>Total Distance: {metrics['total_distance']} km</p>
                <p>Estimated Time: {metrics['estimated_time']} hours</p>
                <p>Shipment IDs: {', '.join(map(str, metrics['shipment_ids']))}</p>
            </div>
        </div>
        """
    html_content += "</body></html>"

    with open(filename, 'w') as f:
        f.write(html_content)

# Example usage
if __name__ == "__main__":
    # Load data
    data = pd.read_csv('Copy of SmartRoute Optimizer.csv')
    
    # Initialize optimizer
    store_location = (19.075887, 72.877911)
    optimizer = DeliveryOptimizer(store_location)
    
    # Create optimized routes
    routes, unassigned = optimizer.optimize_deliveries(data)
    
    # Print route information
    for i, route in enumerate(routes, 1):
        metrics = optimizer.calculate_route_metrics(route)
        print(f"\nRoute {i}:")
        print(f"Vehicle Type: {route['vehicle_type']}")
        print(f"Time Slot: {route['time_slot']}")
        print(f"Number of Deliveries: {metrics['num_deliveries']}")
        print(f"Total Distance: {metrics['total_distance']} km")
        print(f"Estimated Time: {metrics['estimated_time']} hours")
        print(f"Shipment IDs: {', '.join(map(str, metrics['shipment_ids']))}")
    
    # Print unassigned shipments
    if not unassigned.empty:
        print("\nUnassigned Shipments:")
        print(f"Count: {len(unassigned)}")
        print("Shipment IDs:", ', '.join(map(str, unassigned['Shipment ID'].tolist())))
    
    # Visualize routes
    m = optimizer.visualize_routes_folium(routes)
    m.save('delivery_routes_map(n5).html')
    
    # Generate and save route summary
    save_route_summary(routes, optimizer)
    
    # Print basic statistics
    print(f"\nCreated {len(routes)} routes")
    print(f"Unassigned shipments: {len(unassigned)}")