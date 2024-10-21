import csv
from math import radians, sin, cos, sqrt, atan2
import heapq
import folium
from collections import deque
from heapq import heappop, heappush
import networkx as nx

# Definir las clases Airport, Flight y AirportGraph
class Airport:
    def __init__(self, code, name, city, country, latitude, longitude):
        self.code = code
        self.name = name
        self.city = city
        self.country = country
        self.latitude = latitude
        self.longitude = longitude

class Flight:
    def __init__(self, source_airport, dest_airport):
        self.source_airport = source_airport
        self.dest_airport = dest_airport

class AirportGraph:
    def __init__(self):
        self.connections = {}
        self.graph = nx.Graph()

    def add_flight(self, flight):
        source_code = flight.source_airport.code
        dest_code = flight.dest_airport.code
        distance = self.calculate_distance(flight.source_airport, flight.dest_airport)
        if source_code not in self.connections:
            self.connections[source_code] = []
        self.connections[source_code].append(flight)
        self.graph.add_edge(source_code, dest_code, weight=distance)

    def get_connections(self, airport_code):
        return self.connections.get(airport_code, [])

    def calculate_distance(self, airport1, airport2):
        lat1, lon1 = airport1.latitude, airport1.longitude
        lat2, lon2 = airport2.latitude, airport2.longitude
        distance = self.haversine_distance(lat1, lon1, lat2, lon2)
        return distance

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        # Convertir grados decimales a radianes
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

        # Fórmula de Haversine
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = 6371 * c  # Radio de la Tierra en kilómetros

        return distance

    def is_connected(self):
        # Verificar si el grafo es conexo
        if nx.is_connected(self.graph):
            print("The graph is connected.")
            return True, []
        else:
            components = list(nx.connected_components(self.graph))
            print(f"The graph is not connected and has {len(components)} components.")
            for i, comp in enumerate(components):
                print(f"Component {i + 1} has {len(comp)} vertices.")
            return False, components

    def kruskal_mst(self):
        # Obtener todas las aristas del grafo con sus pesos
        edges = [(u, v, data['weight']) for u, v, data in self.graph.edges(data=True)]
        # Ordenar las aristas por peso (ascendente)
        edges.sort(key=lambda x: x[2])

        parent = {code: code for code in self.graph.nodes} 
        rank = {code: 0 for code in self.graph.nodes} 
        component_weight = {code: 0 for code in self.graph.nodes}  # Peso total de cada componente

        def find(node):
            if parent[node] != node:
                parent[node] = find(parent[node])
            return parent[node]

        def union(u, v):
            root_u = find(u)
            root_v = find(v)

            if root_u != root_v:
                if rank[root_u] > rank[root_v]:
                    parent[root_v] = root_u
                    # Suma los pesos de la componente al hacer la unión
                    component_weight[root_u] += component_weight[root_v]
                elif rank[root_u] < rank[root_v]:
                    parent[root_u] = root_v
                    component_weight[root_v] += component_weight[root_u]
                else:
                    parent[root_v] = root_u
                    component_weight[root_u] += component_weight[root_v]
                    rank[root_u] += 1

        mst = []  # Lista de aristas del árbol de expansión mínima

        for u, v, weight in edges:
            # Si u y v no forman un ciclo, añadir al MST
            if find(u) != find(v):
                mst.append((u, v, weight))
                # Sumar el peso de la arista al total del árbol de la componente
                component_weight[find(u)] += weight
                union(u, v)

        unique_components = {find(code): component_weight[find(code)] for code in self.graph.nodes}

        return mst, unique_components

    def longest_paths_info_from_vertex(self, airport_code):
        if airport_code not in self.graph.nodes:
            return [] 

        # Comienzo de Dijkstra
        distances = {code: float('inf') for code in self.graph.nodes}
        distances[airport_code] = 0  # La distancia a sí mismo es 0
        visited = set()  # Nodos ya visitados

        priority_queue = [(0, airport_code)]

        while priority_queue:
            priority_queue.sort(key=lambda x: x[0])
            current_distance, current_node = priority_queue.pop(0)

            if current_node in visited:
                continue 

            visited.add(current_node)

            for neighbor, data in self.graph[current_node].items():
                weight = data['weight']
                new_distance = current_distance + weight

                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    priority_queue.append((new_distance, neighbor))

        # Ordenar los caminos accesibles desde el nodo origen
        accessible_paths = [(airport, dist) for airport, dist in distances.items() if dist < float('inf')]
        longest_paths = sorted(accessible_paths, key=lambda x: x[1], reverse=True)[:11]

        # Construir la lista con la información de los aeropuertos
        airports_info = []
        for airport, distance in longest_paths:
            airport_info = f"Code: {airport}\nDistancia desde {airport_code}: {distance:.2f} km"
            airports_info.append(airport_info)

        return airports_info

# Construir el grafo de aeropuertos desde un archivo CSV
def build_airport_graph(csv_filename):
    airport_graph = AirportGraph()
    airport_info = {}

    with open(csv_filename, 'r', encoding='utf-8') as archivo_csv:
        lector_csv = csv.reader(archivo_csv)
        next(lector_csv)  # Saltar la fila de encabezado

        for fila in lector_csv:
            source_code, source_name, source_city, source_country, source_lat, source_lon, \
                dest_code, dest_name, dest_city, dest_country, dest_lat, dest_lon = fila

            source_airport = Airport(source_code, source_name, source_city, source_country, float(source_lat), float(source_lon))
            dest_airport = Airport(dest_code, dest_name, dest_city, dest_country, float(dest_lat), float(dest_lon))

            flight = Flight(source_airport, dest_airport)

            airport_graph.add_flight(flight)
            airport_info[source_code] = {
                'Code': source_code,
                'Name': source_name,
                'City': source_city,
                'Country': source_country,
                'Latitude': float(source_lat),
                'Longitude': float(source_lon)
            }

            # Asegurarse de que la información del aeropuerto de destino también se guarde
            if dest_code not in airport_info:
                airport_info[dest_code] = {
                    'Code': dest_code,
                    'Name': dest_name,
                    'City': dest_city,
                    'Country': dest_country,
                    'Latitude': float(dest_lat),
                    'Longitude': float(dest_lon)
                }

    return airport_graph, airport_info

# Algoritmo de Dijkstra para encontrar el camino más corto
def dijkstra_shortest_path(graph, start_code, end_code):
    queue = []
    heapq.heappush(queue, (0, start_code))
    distances = {start_code: 0}
    previous = {start_code: None}

    while queue:
        current_distance, current_code = heapq.heappop(queue)

        if current_code == end_code:
            break

        for flight in graph.get_connections(current_code):
            neighbor = flight.dest_airport.code
            distance = graph.calculate_distance(flight.source_airport, flight.dest_airport)
            new_distance = current_distance + distance

            if neighbor not in distances or new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                previous[neighbor] = current_code
                heapq.heappush(queue, (new_distance, neighbor))

    return distances, previous

# Reconstruir el camino más corto
def reconstruct_path(previous, start_code, end_code):
    path = []
    current_code = end_code
    while current_code is not None:
        path.append(current_code)
        current_code = previous.get(current_code)
    path.reverse()
    return path if path[0] == start_code else None

# Graficar el camino más corto en un mapa
def plot_shortest_path_on_map(airport_info, path):
    m = folium.Map(location=[20, 0], zoom_start=2)
    coords = []
    for code in path:
        info = airport_info[code]
        lat = info['Latitude']
        lon = info['Longitude']
        coords.append((lat, lon))
        folium.Marker(
            location=[lat, lon],
            popup=(
                f"Code: {info['Code']}<br>"
                f"Name: {info['Name']}<br>"
                f"City: {info['City']}<br>"
                f"Country: {info['Country']}<br>"
                f"Latitude: {info['Latitude']}<br>"
                f"Longitude: {info['Longitude']}"
            ),
        ).add_to(m)
    folium.PolyLine(coords, color='red', weight=2.5, opacity=1).add_to(m)
    m.save("shortest_path.html")
    print("Shortest path map saved as 'shortest_path.html'")

# Visualizar todos los aeropuertos y conexiones en el mapa
def visualize_airports_on_map(airport_info, airport_graph):
    m = folium.Map(location=[20, 0], zoom_start=2)

    # Añadir conexiones para todos los vuelos
    for source_code, flights in airport_graph.connections.items():
        for flight in flights:
            source_info = airport_info[flight.source_airport.code]
            dest_info = airport_info[flight.dest_airport.code]
            coords = [
                (source_info['Latitude'], source_info['Longitude']),
                (dest_info['Latitude'], dest_info['Longitude'])
            ]
            folium.PolyLine(coords, color='blue', weight=1.5, opacity=0.7, tooltip=f"Weight: {airport_graph.calculate_distance(flight.source_airport, flight.dest_airport):.2f} km").add_to(m)

    # Guardar mapa como HTML
    m.save("mapa_aeropuertos.html")
    print("Map saved as 'mapa_aeropuertos.html'")

# Visualizar MST en un mapa diferente
def visualize_mst_on_map(airport_info, mst_edges):
    m = folium.Map(location=[20, 0], zoom_start=2)

    # Añadir aristas del MST
    for source, dest, weight in mst_edges:
        source_info = airport_info[source]
        dest_info = airport_info[dest]
        coords = [
            (source_info['Latitude'], source_info['Longitude']),
            (dest_info['Latitude'], dest_info['Longitude'])
        ]
        folium.PolyLine(coords, color='green', weight=2.5, opacity=0.9, tooltip=f"Weight: {weight:.2f} km").add_to(m)

    # Añadir nodos del MST (aeropuertos)
    for code, info in airport_info.items():
        folium.Marker(
            location=[info['Latitude'], info['Longitude']],
            popup=(
                f"Code: {info['Code']}<br>"
                f"Name: {info['Name']}<br>"
                f"City: {info['City']}<br>"
                f"Country: {info['Country']}<br>"
                f"Latitude: {info['Latitude']}<br>"
                f"Longitude: {info['Longitude']}"
            ),
        ).add_to(m)

    # Guardar MST como HTML
    m.save("mst_map.html")
    print("MST map saved as 'mst_map.html'")

# Mostrar informacion de los caminos top 10 caminos mas largos
def show_airport_info_and_longest_paths(airport_graph, airport_info, start_code):
    print("\nAirport Information:")
    info = airport_info[start_code]
    print(f"Code: {info['Code']}")
    print(f"Name: {info['Name']}")
    print(f"City: {info['City']}")
    print(f"Country: {info['Country']}")
    print(f"Latitude: {info['Latitude']}")
    print(f"Longitude: {info['Longitude']}")

    # Según el código encontrar los caminos top 10 caminos mas largos
    longest_paths_info = airport_graph.longest_paths_info_from_vertex(start_code)
    print("\nTop 10 Longest Paths from Airport:")
    for airport_info in longest_paths_info:
        print(airport_info)

# MAIN
if __name__ == "__main__":
    airport_graph, airport_info = build_airport_graph('flights_final.csv')

    # Inputs
    start_airport_code = input("Enter the start airport code: ")
    end_airport_code = input("Enter the end airport code: ")

    # Verificar que sea conexo
    is_connected, components = airport_graph.is_connected()

    # Calcular MST con Kruskal
    mst_edges = []
    unique_components = {}
    if not is_connected:
        for component in components:
            subgraph = AirportGraph()
            for code in component:
                for flight in airport_graph.get_connections(code):
                    subgraph.add_flight(flight)
            mst, component_weights = subgraph.kruskal_mst()
            mst_edges.extend(mst)
            unique_components.update(component_weights)
    else:
        mst, unique_components = airport_graph.kruskal_mst()
        mst_edges.extend(mst)

    # Mostrar el peso del MST para cada componente en la consola
    for component, weight in unique_components.items():
        print(f"Component Root: {component}, Total Weight of MST: {weight:.2f} km")

    distances, previous = dijkstra_shortest_path(airport_graph, start_airport_code, end_airport_code)
    path = reconstruct_path(previous, start_airport_code, end_airport_code)

    if path:
        print("\nShortest Path:")
        for code in path:
            if code in airport_info:
                info = airport_info[code]
                print(f"Airport Code: {info['Code']}")
                print(f"Name: {info['Name']}")
                print(f"City: {info['City']}")
                print(f"Country: {info['Country']}")
                print(f"Latitude: {info['Latitude']}")
                print(f"Longitude: {info['Longitude']}")
                print("-" * 40)

        plot_shortest_path_on_map(airport_info, path)
    else:
        print("No path exists between the two airports.")

    visualize_airports_on_map(airport_info, airport_graph)

    # Visualizar el MST en un mapa separado
    if mst_edges:
        visualize_mst_on_map(airport_info, mst_edges)

    # Mostrar informacion del aeropuesto y los caminos mas largos
    show_airport_info_and_longest_paths(airport_graph, airport_info, start_airport_code)
