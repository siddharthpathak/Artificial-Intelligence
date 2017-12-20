
# Fall 2017, B-551 Elements of AI. Prof. Crandall
# Assignment 1, Question 1


# The search problem has the given data and routes between several different cities. We stored all the routes along with their other info into a dictionary.
# Storing the data into dictionary made fetching the neighbor cities from its current cities fast in constant time.
# Once the neighbor cities are found, the search algorithm determines which city to explore first.


# State Space: All the neighbors of current city/junction along with their cost,distance,speed and highway name connecting them. Each state also contains the
# parent city -- the city from which the neighbor city was reached.

# Successor Function: Successor function checks the city_routes dictionary city_routes dictionary stores the road-segments.txt data for faster access.
# For each neighbor of the current city, it creates a state which contains the distance,time,highway name between the current city and neighbors.

# Cost:
# The edge weights differ as cost function differs.
# For distance cost function, edge weight is the distance between two cities.
# For time cost function, edge weight is the time required to travel between two cities. Time take is distance between two cities divided by the maximum highway speed
# For segment cost function, edge weight is constant 1. Each connection between two cites is 1 segment in the search graph

#Heuristic:
# For A*, We used great circle distance between the current city and the goal city. For calculating great circle distance, We used the latitude and longitude
# provided in the city-gps file for each city.
# The data contains some noise, like few cities present in the city_routes file don't have their latitude and longitude present in city-gps file.
# If the data is not present then We assumed the city's heuristic to be zero.
# Because of absence of latitude and longitude values, the heuristic underestimates the cost, and makes it weak.
# The heuristic is admissible, but not consistent. For majority of the cities it is admissible and consistent as the data gives the correct latitude and
# longitude values. But for some cities, the latitude and longitude data is not correct, which makes the heuristic inadmissible in some test cases.
# Heuristic is not consistent as some cities don't have correct data, because of which we kept track of city and its respective cost in visited array.
# If the city has been visited before but with higher cost then visit it again.
# For time heuristic, We divided the great circle distance by speed of 100. We assumed the value to be 100 as the maximum speed in the given data is 65.
# For segment heuristic, We divided the great circle distance by 3000. We assumed the maximum number of segments between two cities to be 3000.




#Search Algorithm:
# We defined three seach algorithm for BFS/DFS,Uniform Cost Search and A* search.
# BFS search algorithm traverses nodes with minimum depth first. It doesn't consider the cost function. It also keeps track of visited states
# to avoid expanding the same city twice.
# In DFS search it pops the node and expands it as far as possible. It doesn't consider the cost function. It also keeps track of visited states
# to avoid expanding the same city twice.
# In uniform cost function, the algorithm pops the node which has the lowest edge weight in the fringe. The edge weight depends upom the cost function.
# Edge weight is the cost till the parent city and the edge weight of parent city to its successors.
# Uniform Cost function requires fringe to be a priority queue. We used Python's heapq module for implementing priority queue.
# A* algorithm pops the node which has minimum edge weight. In case of A*, edge weight is the summation of the distance travelled till now plus the heuristic.
# As the heuristic is not consistent, We kept track of the visited nodes along with their heuristic when they were visited. If the heuristic when they were visited
# first is more than the current node's heuristic then expand the node again.


#Problems faced:
# Majority of the problems We faced were because of missing data. For example, once We decided my heuristic to be the great circle distance between two cities,
# We was not able to calculate it for city as they didn't have latitude and longitude in the city-gps.txt file.
# Also, some the cities have wrong latitude and longitude values. We thought my A* search algorithm is not working properly, but after looking into the data and a lot of debugging,
# We realised that few cities don't have correct Latitude and Longitude values. For example, Mission_Hills,_California has wrong latitude and longitude values.

#Assumptions:

# If in the given data, the distance between two cities is absent or 0 then we ignored that city.
# Similarly, if speed between two cities is absent or 0 then we ignored that city.
# If a city doesn't have its latitude and longitude values in city-gps, city's heuristic to be 0.



# Analysis
# 1. A* seems to work best for each routing option. Because of A* the number of nodes expanded are less than that of Uniform Cost and BFS/DFS.
# But in some cases, the heuristic is weak because of data noise, as it underestimates the distance and is not consistent.
# In terms of optimality, Uniform cost search always gives optimal path but is not as efficient as A* with a good heuristic.
# If data didn't have noise then A* star would have been the best routing algorithm, but as the heuristic is weak the computation time difference between A* and UCS is not much.

# 2. A* with a correct heuristic is more efficient than other algorithms. BFS and DFS don't give optimal path. but for some cases they do take less computation time.
# A* takes less time to compute because of heuristic function it expands less number of nodes as compared to the other search algorithms. But it spends some time in calculating
# heuristic value for each city. If the time spent is less for calculating heuristic, then A* will run faster. Performance of A* star depends on how strong the heuristic function is.


# Number of nodes expanded and computation time taken in A* vs UCS:
# San_Jose,_California to Miami,_Florida A* : 2018,0.198 seconds UCS : 5848, 0.280 seconds
# San_Jose,_California to Bloomington,_Indiana A*: 937, 0.164 seconds UCS: 3879, 0.232 seconds
# Bloomington,_Indiana to Seattle,_Washintgon A*: 2866, 0.226 seconds UCS: 6272 0.286 seconds
# Bloomington,_Indiana to Texas,_Dallas A* 544, 0.151 seconds UCS: 3987 0.227 seconds

#3 Heuristic function:
# We used Great Circle distance as heuristic value. It is admissible given the latitude and longitude values are accurate in the given data.
# As the data has noise, because of which some cities don't have latitude and longitude values.It makes the heuristic weak.
# Heuristic function can be improved by having accurate data for all the cities.
# Heuristic is not consistent, few nodes are expanded again, which decreases the efficiency of A* by some amount.
# For time and segment cost function, the heuristic is weak. Heuristic gives data fro the distance between the two cities in miles.
# It doesn't give information about the time or segments between them. We had to assume the maximum speed and maximum number of segments
# between two cities to make. It makes heuristic weaker, but is admissible.


import heapq
import math
import sys

city_routes = {}  # to store the file data in Dictionary
gps = {}          # to store longitude and latitude of Cities
visited_states = {}


states = ['Mississippi', 'Iowa', 'Oklahoma', 'Wyoming', 'Minnesota', 'New_Jersey', 'Arkansas', 'Indiana', 'Maryland', 'Louisiana', 'New_Hampshire', 'Texas', 'New_York', 'Arizona', 'Wisconsin', 'Michigan', 'Kansas', 'Utah', 'Virginia', 'Oregon', 'Connecticut', 'Montana', 'California', 'Idaho', 'New_Mexico', 'South_Dakota', 'Massachusetts', 'Vermont', 'Georgia', 'Pennsylvania', 'Florida', 'North_Dakota', 'Tennessee', 'Nebraska', 'Kentucky', 'Missouri', 'Ohio', 'Alabama', 'Illinois', 'Colorado', 'Washington', 'West_Virginia', 'South_Carolina', 'Rhode_Island', 'North_Carolina', 'Nevada', 'Delaware', 'Maine']


def get_coordinates(city_gps):
    """
        Reads the city latitude and longitude values from
        into gps dictionary.
    """
    input_file = open(city_gps, "r")
    for line in input_file:
        temp = line.strip().split(" ")
        gps[temp[0]] = (float(temp[1]), float(temp[2]))


def read_input(road_segments):
    """
        Reads the road-segments data into city_routes Dictionary.
        Ignores routes which have speed or distance as 0 or blank.
    """
    input_file = open(road_segments, "r")
    for l in input_file:
        temp = l.split(" ")
        if temp[-3] != '' and temp[-2] != '':
            if int(temp[-3]) != 0 and int(temp[-2]) != 0:
                if temp[0] in city_routes:
                    city_routes[temp[0]].append(temp[1:])
                else:
                    city_routes[temp[0]] = [temp[1:]]
                if temp[1] in city_routes:
                    city_routes[temp[1]].append(temp[:1]+temp[2:])
                else:
                    city_routes[temp[1]] = [temp[:1]+temp[2:]]


def print_route(end_city, goal):
    """
    Prints the route to the goal in a similar way to Google Maps.
    """
    total_miles,time_required =0,0
    temp_end_city = end_city
    temp = goal[temp_end_city]
    result = []
    while temp[0] is not None:
        result.append(["From ", temp[0], " to ", temp_end_city, "via", temp[-1].strip(), "(", temp[1], " miles)"])
        total_miles = total_miles + float(temp[1])
        time_required = time_required + float(temp[2])
        temp_end_city = temp[0]
        temp = goal[temp_end_city]
    for r in reversed(result):
        for x in r:
            print x,
        print
    temp = goal[end_city]
    result = []
    result.append(end_city)
    while temp[0] is not None:
        result.append(temp[0])
        temp_end_city = temp[0]
        temp = goal[temp_end_city]

    print "Total Miles:", total_miles, "Time Required:", time_required , "at average speed of:", float(total_miles / time_required), "mph"
    print total_miles,time_required,
    for p in result[::-1]:
        print p,


def is_goal(s,end_city):
    """
        Checks if t`he goal city has reached.
    """
    return s == end_city

# Referred: https://en.wikipedia.org/wiki/Great-circle_distance


def heuristic(current_city,goal_city):
    """
    Returns heuristic value using Great Circle Formula.
    If city or junction doesn't have latitude/longitude info in ctiy-gps file then take its heuristic as 0.
    """

    if current_city not in gps:
        return 0
    x1,y1 = gps[current_city]
    x2,y2 = gps[goal_city]
    x1 = math.radians(x1)
    y1 = math.radians(y1)
    x2 = math.radians(x2)
    y2 = math.radians(y2)
    distance = float(69.1105 * (math.degrees(math.acos(math.sin(x1) * math.sin(x2) \
         + math.cos(x1) * math.cos(x2) * math.cos(y1 - y2)))))
    return distance


def successors(temp,cost="bfs/dfs"):
    """
        Generates successors for bfs,dfs and uniform cost search.
    """
    suc = city_routes[temp[-1]]
    res = []
    if cost == "distance":
        return [[int(s[-3])+int(temp[0])]+[int(s[-3])]+[float(s[-3])/float(s[-2])]+[s[-1]]+ [temp[-1]]+ [s[0]] for s in suc]
    elif cost == "time":
        return [[float(s[-3])/float(s[-2])+float(temp[0])]+[int(s[-3])]+[float(s[-3])/float(s[-2])]+[s[-1]]+[temp[-1]]+[s[0]] for s in suc]
    elif cost == "longtour":
        return [[-(int(s[-3]))+int(temp[0])]+[int(s[-3])]+[float(s[-3])/float(s[-2])]+[s[-1]]+[temp[-1]]+[s[0]] for s in suc]
    elif cost == "segments":
        return [[1+int(temp[0])]+[int(s[-3])]+[float(s[-3])/float(s[-2])]+[s[-1]]+[temp[-1]]+[s[0]] for s in suc]
    elif cost == "statetour":
        for s in suc:
            if s[0].split(",")[-1][1:] not in states:
                continue
            if s[0].split(",")[-1][1:] not in visited_states and s[0].split(",")[-1][1:] in states:
                res.append([((int(s[-3])) + int(temp[0]))-len(visited_states)*10000] + [int(s[-3])] + [float(s[-3]) / float(s[-2])] + [s[-1]] + [temp[-1]] + [s[0]])
            else:
                res.append([int(s[-3]) + int(temp[0])] + [int(s[-3])] + [float(s[-3]) / float(s[-2])] + [s[-1]] + [temp[-1]] + [s[0]])
        return res
    else:  #if bfs and dfs
        return [[0]+[int(s[-3])]+[float(s[-3])/float(s[-2])]+[s[-1]]+[temp[-1]]+[s[0]]for s in suc]


def successors_heuristic(temp,cost):
    """
        Generates successors for A* algorithm.
    """
    suc = city_routes[temp[-1]]
    if cost == "distance":
        return [[float(s[-3])+float(temp[1])+heuristic((s[0]),end_city)]+[float(s[-3])+float(temp[1])]+[int(s[-3])]+[float(s[-3])/float(s[-2])]+[s[-1]]+[temp[-1]]+[s[0]] for s in suc]
    elif cost == "time":
        return [[float(s[-3])/float(s[-2])+float(temp[1])+heuristic((s[0]),end_city)/100.0]+[float(s[-3])/float(s[-2])+float(temp[1])]+[int(s[-3])]+[float(s[-3])/float(s[-2])]+[s[-1]]+[temp[-1]]+[s[0]] for s in suc]
    elif cost == "segments":
        return [[1+float(temp[1])+heuristic((s[0]),end_city)/3000]+[1+float(temp[1])]+[int(s[-3])]+[float(s[-3])/float(s[-2])]+[s[-1]]+[temp[-1]]+[s[0]] for s in suc]
    elif cost == "longtour":
        return [[-(float(s[-3])+float(temp[1])+ (heuristic((s[0]),end_city)))]+[float(s[-3])+float(temp[1])]+[int(s[-3])]+[float(s[-3])/float(s[-2])]+[s[-1]]+[temp[-1]]+[s[0]] for s in suc]
    elif cost == "statetour":
        return
    else:
        return [[float(s[-3])+float(temp[1])+heuristic((s[0]),end_city)]+[float(s[-3])+float(temp[1])]+[int(s[-3])]+[float(s[-3])/float(s[-2])]+[s[-1]]+[temp[-1]]+[s[0]] for s in suc]


def solve_dfs_bfs(start_city,end_city,type):
    """
        Search algorithm for BFS and DFS. Keeps track of visited cities to avoid loopy and redundant paths.
    """
    visited = {}
    fringe = []
    trace_route = {}
    fringe.append([0,0,0,None,None,start_city])
    while len(fringe) > 0:
        if type == "bfs":
            temp = fringe.pop(0)
        else:
            temp = fringe.pop()
        if temp[-1] in visited:
            pass
        else:
            trace_route[temp[-1]] = (temp[-2], temp[1], temp[2], temp[-3])
            visited[temp[-1]] = 1
            for s in successors(temp):
                if is_goal(s[-1],end_city):
                    trace_route[s[-1]] = (s[-2], s[1], s[2], s[-3])
                    return trace_route
                fringe.append(s)
    return False


def solve_uniform(start_city,end_city,cost):
    """
        Search Algorithm for uniform cost function.
    """
    nodes_visited = 0
    visited = {}
    fringe = []
    trace_route = {}
    heapq.heappush(fringe,[0,0,0,None,None,start_city])
    while len(fringe) > 0:
        temp = heapq.heappop(fringe)
        if is_goal(temp[-1],end_city):
            trace_route[temp[-1]] = (temp[-2], temp[1], temp[2], temp[-3])
            return trace_route
        if temp[-1] in visited:
            pass
        else:
            nodes_visited += 1
            trace_route[temp[-1]] = (temp[-2], temp[1], temp[2], temp[-3])
            visited[temp[-1]] = 1
            visited_states[temp[-1].split(",")[-1][1:]] = 1
            for s in successors(temp,cost):
                heapq.heappush(fringe,s)
    return False


# Saw the video: https://www.youtube.com/watch?v=6TsL96NAZCo to understand the working of A*.
def solve_a_star(start_city,end_city,cost):
    """
        Search algorithm for A*. Keeps track of visited cities along with their heuristic.
        If a city has visited before with higher heuristic value, then visit that city again.
    """
    nodes_visited = 0
    visited = {}
    fringe = []
    trace_route = {}
    heapq.heappush(fringe,[0,0,0,None,None,start_city])
    while len(fringe) > 0:
        temp = heapq.heappop(fringe)
        if is_goal(temp[-1],end_city):
            trace_route[temp[-1]] = (temp[-2], temp[2], temp[3], temp[-3])
            return trace_route
        if temp[-1] in visited:
            if visited[temp[-1]] > temp[0]:
                visited.pop(temp[-1])
        if temp[-1] in visited:
            pass
        else:
            nodes_visited += 1
            trace_route[temp[-1]] = (temp[-2], temp[2], temp[3], temp[-3])
            visited[temp[-1]] = temp[0]
            for s in successors_heuristic(temp,cost):
                # If the element is already in fringe with a high value then remove that element.
                # But if we keep both the elements in the fringe then the lower value element will be popped and marked as visited.
                # If it's marked as visited, then even if we pop it next time it won't be explored. Removing element from fringe
                # is increasing the time taken to run, as it searches the whole fringe and sorts it again after removing.
                # Reference: Piazza Question 151.
                # for i,s1 in enumerate(fringe):
                #     if s1[-1] == s[-1]:
                #         if s1[0] > s[0]:
                #             fringe.pop(i)
                #             heapq.heapify(fringe)
                heapq.heappush(fringe,s)
    return False



start_city,end_city,routing_algorithm,cost_function = sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]

if start_city == end_city:
    print "Start and End cities are the same!"
    print 0, 0, start_city, end_city
    sys.exit(0)
read_input("road-segments.txt")
get_coordinates("city-gps.txt")

if routing_algorithm == "bfs" or routing_algorithm == "dfs":
    result = solve_dfs_bfs(start_city,end_city,routing_algorithm)
    if result is False:
        print "No Route found"
    else:
        print_route(end_city,result)
elif routing_algorithm == "uniform":
    result = solve_uniform(start_city,end_city,cost_function)
    if result is False:
        print "No Route found"
    else:
        print_route(end_city,result)
else:
    result = solve_a_star(start_city,end_city,cost_function)
    if result is False:
        print "No Route found"
    else:
        print_route(end_city,result)