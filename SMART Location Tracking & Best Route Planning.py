import serial
import time
import requests
from math import radians, cos, sin, asin, sqrt


# Setup Serial Communication with SIM8200EA-M2
ser = serial.Serial('/dev/ttyUSB2', baudrate=115200, timeout=1)


def send_at_command(command, timeout=1):
    ser.write((command + '\r').encode())
    time.sleep(timeout)
    response = ser.read(ser.inWaiting()).decode()
    return response


def get_gnss_location():
    # AT Command to start GNSS
    send_at_command('AT+CGNSPWR=1')
   
    # Get GNSS information
    response = send_at_command('AT+CGNSINF', 2)
   
    # Example response: +CGNSINF: 1,1,20210312150824.000,40.730610,-73.935242,10.00,0.0,0.0,0.0,1.0,1.2,1.0
    if "+CGNSINF: 1,1," in response:
        parts = response.split(',')
        latitude = float(parts[3])
        longitude = float(parts[4])
        return latitude, longitude
    else:
        return None


def haversine(lon1, lat1, lon2, lat2):
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])


    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles.
    return c * r


def find_closest_bin(current_location, bin_locations):
    closest_bin = None
    min_distance = float('inf')
    for bin_location in bin_locations:
        distance = haversine(current_location[1], current_location[0],
                             bin_location[1], bin_location[0])
        if distance < min_distance:
            min_distance = distance
            closest_bin = bin_location
    return closest_bin


def plan_route(start_location, bin_locations):
    route = []
    current_location = start_location


    while bin_locations:
        closest_bin = find_closest_bin(current_location, bin_locations)
        route.append(closest_bin)
        bin_locations.remove(closest_bin)
        current_location = closest_bin


    return route


# Main Program
def main():
    # Get current location of the waste bin using GNSS
    location = get_gnss_location()
    if location:
        print(f"Current Location: Lat: {location[0]}, Lon: {location[1]}")
    else:
        print("Failed to get GNSS location")
        return


    # List of other bin locations (Lat, Lon)
    bin_locations = [(3.11793, 101.65535), (3.12827, 101.65087), (3.11790, 101.66091)]
   
    # Plan route
    route = plan_route(location, bin_locations)
    print("Optimized Collection Route:")
    for idx, point in enumerate(route):
        print(f"Stop {idx+1}: Lat: {point[0]}, Lon: {point[1]}")

   api_key = "YOUR_API_KEY"
   
    # Print directions for the route
    print_route_directions(route, api_key)

if __name__ == "__main__":
    main()
