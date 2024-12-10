import io
import math
import random
import time
import matplotlib.pyplot as plt

class TSP_ACO:
    def __init__(self, numberOfCities=10, numberOfAnts=10, alpha=2, beta=4, pheromonesAmount=0.5, runIterations=150, evaporation=0.5, debugMode=False):
        self.numberOfCities = numberOfCities
        self.numberOfAnts = numberOfAnts
        self.alpha = alpha # importance of pheromones
        self.beta = beta # importance of distance
        self.pheromonesAmount = pheromonesAmount
        self.runIterations = runIterations
        self.evaporation = evaporation
        self.debugMode = debugMode
        self.PlotInitiated = False

        self.cityList = [
            self.City(66, 59),
            self.City(53, 109),
            self.City(155, 112),
            self.City(57, 139),
            self.City(94, 18),
            self.City(112, 136),
            self.City(111, 49),
            self.City(71, 109),
            self.City(141, 56),
            self.City(26, 124)
            ]
        self.edges = []
        self.ants = []
        self.initialPheromones = 1.0
        
        self.bestDistance = float("inf")
        self.bestTour = []

        # Self initialization
        # self.randomGenerateCities(self.numberOfCities)
        self.initializeEdges()    

    # Randomly generate cities
    def randomGenerateCities(self, amount):
        for i in range(amount):
            city = self.City(random.randint(0, 200), random.randint(0, 200))
            self.cityList.append(city)

    # Calculate the distance between cities
    def calculateDistance(self, a, b) -> float:
        return math.sqrt((b.x - a.x)**2 + (b.y - a.y)**2)

    # Create the edges for given cities
    def initializeEdges(self):
        length = len(self.cityList)
        self.edges = [[None] * length for _ in range(length)]
        for i in range(length):
            for j in range(i + 1, length):
                distance = self.calculateDistance(self.cityList[i], self.cityList[j])
                self.edges[i][j] = self.edges[j][i] = self.Edge(self.cityList[i], self.cityList[j], self.initialPheromones, distance)

    # Create the ants
    def initializeAnts(self):
        self.ants = [self.Ant(self.alpha, self.beta, self.cityList, self.edges, self.pheromonesAmount, self.debugMode) for _ in range(self.numberOfAnts)]

    # ACO algorithm
    def aco(self):
        for _ in range(self.runIterations):
            print(f"Iteration: {_}")
            self.initializeAnts()
            for ant in self.ants:
                tour = ant.findTour()
                distance = ant.getDistance(tour)
                ant.addPheromone(tour, distance)
                if distance < self.bestDistance:
                    self.bestDistance = distance
                    self.bestTour = tour
                    print("New best distance:", self.bestDistance)
                self.plotTour(ant.tour) # each ant's tour
            # self.plotTour(self.bestTour)   # the best tour of each iteration      

            for i in range(len(self.cityList)):
                for j in range(i + 1, len(self.cityList)):
                    self.edges[i][j].pheromones *= (1.0 - self.evaporation)

    # Run the instance
    def run(self):
        if self.debugMode:
            print("Starting ACO algorithm...")
        start = time.time()
        self.aco()
        runtime = time.time() - start
        if self.debugMode:
            print("ACO algorithm finished.")
        return [runtime, self.bestDistance, self.bestTour]

    # Graph a tour of the instance
    def plotTour(self, tour):
        x = 0
        y = 0
        if self.PlotInitiated == False:
            x = [city.x for city in self.cityList]
            y = [city.y for city in self.cityList]

            plt.figure(figsize=(10, 6))
            plt.grid(True)
            plt.title(f'Best Tour: Distance = {self.bestDistance}')
            plt.ion()
            self.PlotInitiated = True

        if self.PlotInitiated == True:
            plt.clf()
            x = [city.x for city in self.cityList]
            y = [city.y for city in self.cityList]
            plt.scatter(x, y, c='red')
            for i in range(len(tour)):
                start = tour[i]
                end = tour[(i + 1) % len(tour)]
                plt.plot([start.x, end.x], [start.y, end.y], 'b-')

            for i, city in enumerate(tour):
                plt.text(city.x, city.y, f'{i}', fontsize=12, ha='right')
            
            plt.pause(0.001)
            
    def plotFinal(self, tour):
        x = [city.x for city in self.cityList]
        y = [city.y for city in self.cityList]

        plt.figure(figsize=(10, 6))
        plt.grid(True)
        plt.title(f'Best Tour: Distance = {self.bestDistance}')

        plt.scatter(x, y, c='red')
        for i in range(len(tour)):
            start = tour[i]
            end = tour[(i + 1) % len(tour)]
            plt.plot([start.x, end.x], [start.y, end.y], 'b-')

        for i, city in enumerate(tour):
            plt.text(city.x, city.y, f'{i}', fontsize=12, ha='right')
        
        plt.ioff()
        plt.show()
        
    class City:
        def __init__(self, x, y) -> None:
            self.x = x
            self.y = y

        def __str__(self):
            return f"({self.x}, {self.y})"

    class Edge:
        def __init__(self, start, end, pheromones, distance) -> None:
            self.start = start
            self.end = end
            self.pheromones = pheromones
            self.distance = distance

        def __str__(self) -> str:
            return f"{self.start}, {self.end} dist: {self.distance}"

    class Ant:
        def __init__(self, alpha, beta, cities, edges, pheromonesAmount, debugMode) -> None:
            self.alpha = alpha
            self.beta = beta
            self.cities = cities
            self.edges = edges
            self.pheromonesAmount = pheromonesAmount
            self.tour = []
            self.distanceTraveled = 0
            self.currentCity = cities[0]
            self.debugMode = debugMode

        # Select the node the ant will go next to
        def selectNode(self):
            probabilities = []
            unvisitedCities = [city for city in self.cities if city not in self.tour and city != self.cities[0]]
            currentCityIndex = self.cities.index(self.currentCity)

            matchingEdges = [edge for edge in self.edges[currentCityIndex] if edge and edge.end in unvisitedCities]

            for edge in matchingEdges:
                probabilities.append(self.calculateProbability(edge, matchingEdges))

            nextCity = self.randomByWeight(unvisitedCities, probabilities)
            self.currentCity = nextCity
            return nextCity

        # Calculate the probability that an ant will move along an edge
        def calculateProbability(self, edge, allowedEdges) -> float:
            totalDesire = sum(pow(allowedEdge.pheromones, self.alpha) * pow((1.0 / (allowedEdge.distance + 0.0000001)), self.beta) for allowedEdge in allowedEdges)
            desire = pow(edge.pheromones, self.alpha) * pow((1 / (edge.distance + 0.0000001)), self.beta)
            return desire / (totalDesire + 0.0000001)

        # Weighted Random Algorithm
        def randomByWeight(self, values, weights):
            total = sum(weights)
            if total == 0:
                return random.choice(values)
            randNum = random.uniform(0.0, total)
            cursor = 0.0
            for i in range(len(weights)):
                cursor += weights[i]
                if cursor >= randNum:
                    return values[i]
            return values[-1]

        # Add pheromone to the edges
        def addPheromone(self, tour, distance, weight=1.0):
            pheromoneToAdd = self.pheromonesAmount / (distance + 0.00000001)
            for i in range(len(tour)):
                indexA = self.cities.index(tour[i])
                indexB = self.cities.index(tour[(i + 1) % len(tour)])
                self.edges[indexA][indexB].pheromones += weight * pheromoneToAdd

        # Find a complete path for the ant
        def findTour(self):
            self.tour = [self.cities[0]]
            for _ in range(1, len(self.cities)):
                self.tour.append(self.selectNode())
            return self.tour

        # Get total distance traveled of the ant
        def getDistance(self, tour):
            distanceTraveled = 0
            for i in range(len(tour)):
                indexA = self.cities.index(tour[i])
                indexB = self.cities.index(tour[(i + 1) % len(tour)])
                distanceTraveled += self.edges[indexA][indexB].distance
            return distanceTraveled

# Initialization
tspaco = TSP_ACO()
results = tspaco.run()
print("Time taken:", results[0])
print("Best distance:", results[1])
print("Best tour:")
for city in results[2]:
    print(city)

# Plot the tour
tspaco.plotFinal(tspaco.bestTour)

