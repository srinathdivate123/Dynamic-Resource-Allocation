#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#define NUM_SERVERS 4
#define NUM_TASKS 10
#define NUM_PARTICLES 50
#define MAX_ITERATIONS 2000
#define INERTIA_WEIGHT 0.7
#define COGNITIVE_WEIGHT 1.5
#define SOCIAL_WEIGHT 1.5
#define MAX_CAPACITY 500
#define MIN_CAPACITY 100
#define MIN_CPU 10
#define MAX_CPU 50
#define MIN_MEMORY 5
#define MAX_MEMORY 30
#define MIN_EXECUTION_TIME 1
#define MAX_EXECUTION_TIME 10

typedef struct
{
    int id;
    double capacity;
    double currentLoad;
    double cpuUtilization;
    double memoryUtilization;
} Server;

typedef struct
{
    int id;
    double cpuRequirements;
    double memoryRequirements;
    double execution_time;
} Task;

typedef struct
{
    int assignment[NUM_TASKS];
    double fitness;
    int p_best[NUM_TASKS];
} Particle;

Server servers[NUM_SERVERS];
Particle particles[NUM_PARTICLES];
Particle g_bestParticle;
Task tasks[NUM_TASKS];

double evaluateFitness(Particle *particle);

void initialize()
{
    srand(time(NULL));

    // Servers with random capacities within constraints
    for (int i = 0; i < NUM_SERVERS; ++i)
    {
        servers[i].id = i;
        servers[i].capacity = MIN_CAPACITY + (rand() % (MAX_CAPACITY - MIN_CAPACITY + 1));
        servers[i].currentLoad = 0.0;
        servers[i].cpuUtilization = 0.0;
        servers[i].memoryUtilization = 0.0;
    }

    // Taskes with random CPU & memory requirements and execution times
    for (int i = 0; i < NUM_TASKS; ++i)
    {
        tasks[i].id = i;
        tasks[i].cpuRequirements = MIN_CPU + (rand() % (MAX_CPU - MIN_CPU + 1));
        tasks[i].memoryRequirements = MIN_MEMORY + (rand() % (MAX_MEMORY - MIN_MEMORY + 1));
        tasks[i].execution_time = MIN_EXECUTION_TIME + (rand() % (MAX_EXECUTION_TIME - MIN_EXECUTION_TIME + 1));
    }

    // Initialize particles with random assignments
    for (int i = 0; i < NUM_PARTICLES; ++i)
    {
        for (int j = 0; j < NUM_TASKS; ++j)
            particles[i].assignment[j] = rand() % NUM_SERVERS;
        particles[i].fitness = evaluateFitness(&particles[i]);
        for (int j = 0; j < NUM_TASKS; ++j)
            particles[i].p_best[j] = particles[i].assignment[j];
    }

    // Initialize the fittness of global best particle to -INFINITY 
    g_bestParticle.fitness = -INFINITY;
}

double evaluateFitness(Particle *particle)
{
    // Calculate the total resource wastage across all servers
    double totalWastage = 0.0;

    // Calculate the load on each server based on task assignments
    for (int i = 0; i < NUM_SERVERS; ++i)
    {
        servers[i].currentLoad = 0.0;
        servers[i].cpuUtilization = 0.0;
        servers[i].memoryUtilization = 0.0;
    }

    for (int i = 0; i < NUM_TASKS; ++i)
    {
        int serverId = particle->assignment[i];
        double taskMemoryReq = tasks[i].memoryRequirements;
        double serverCapacity = servers[serverId].capacity;

        // Update server load and calculate wastage
        servers[serverId].currentLoad += taskMemoryReq;
        double wastage = fmax(0, servers[serverId].currentLoad - serverCapacity);
        totalWastage += wastage;

        // Update server CPU and memory utilization
        servers[serverId].cpuUtilization += tasks[i % NUM_TASKS].cpuRequirements;
        servers[serverId].memoryUtilization += tasks[i % NUM_TASKS].memoryRequirements;
    }

    // Calculate CPU and Memory Utilization as a percentage
    for (int i = 0; i < NUM_SERVERS; ++i)
    {
        servers[i].cpuUtilization = (servers[i].cpuUtilization / servers[i].capacity) * 100.0;
        servers[i].memoryUtilization = (servers[i].memoryUtilization / servers[i].capacity) * 100.0;
    }

    // Fitness is the negative of total wastage (minimize wastage)
    double fitness = -totalWastage;

    return fitness;
}

void dynamicResourceAllocation()
{
    // Monitor server loads and adjust capacities dynamically
    for (int i = 0; i < NUM_SERVERS; ++i)
    {
        servers[i].cpuUtilization = servers[i].cpuUtilization / servers[i].capacity;
        servers[i].memoryUtilization = servers[i].memoryUtilization / servers[i].capacity;
        printf("Server %d CPU Utilization is %.2f%%\n", i, servers[i].cpuUtilization*100);

        if (servers[i].cpuUtilization > 0.70)
        {
            servers[i].capacity += 10; // Increase capacity of server
            printf("Increased capacity of server %d by 10 units\n\n", i);
        }
        else if (servers[i].cpuUtilization < 0.2)
        {
            servers[i].capacity -= 10; // Decrease capacity of server
            printf("Decreased capacity of server %d by 10 units\n\n", i);
        }

        // Ensure capacity remains within constraints
        servers[i].capacity = fmax(MIN_CAPACITY, fmin(MAX_CAPACITY, servers[i].capacity));
    }
}

void updateParticle(Particle *particle)
{
    // These random values are used to introduce stochasticity into the PSO algorithm
    // The values are used to compute the cognitive and social components of the velocity update

    double cognitiveComponent = (double)rand() / RAND_MAX;
    double socialComponent = (double)rand() / RAND_MAX; 

    for (int i = 0; i < NUM_TASKS; ++i)
    {
        double inertiaTerm = INERTIA_WEIGHT * particle->p_best[i];
        double cognitiveTerm = COGNITIVE_WEIGHT * cognitiveComponent * (particle->p_best[i] - particle->assignment[i]);
        double socialTerm = SOCIAL_WEIGHT * socialComponent * (g_bestParticle.assignment[i] - particle->assignment[i]);

        // These three terms together contribute to the calculation of velocityUpdate, which represents how the particle should adjust its task assignments (position) in the solution space
        // The balance between these terms guides the particle's movement as it seeks to explore and exploit the solution space to find better assignments that optimize the given fitness function

        double heuristicValue = inertiaTerm + cognitiveTerm + socialTerm; // This is the velocity of the particle
        double newAssignment = particle->assignment[i] + heuristicValue;

        // Ensure that the newAssignment server which we are trying to assign is >0 and less than NUM_SERVERS
        if (newAssignment < 0)
            newAssignment = 0;
        else if (newAssignment >= NUM_SERVERS)
            newAssignment = NUM_SERVERS - 1;

        particle->assignment[i] = (int)newAssignment;
    }

    // Update p_best if needed
    double currentFitness = evaluateFitness(particle);
    if (currentFitness > particle->fitness)
    {
        particle->fitness = currentFitness;
        for (int i = 0; i < NUM_TASKS; ++i)
            particle->p_best[i] = particle->assignment[i];
    }

    // Update g_bestParticle if needed
    if (currentFitness > g_bestParticle.fitness)
    {
        g_bestParticle.fitness = currentFitness;
        for (int i = 0; i < NUM_TASKS; ++i)
            g_bestParticle.assignment[i] = particle->assignment[i];
    }
}

int main()
{
    initialize();
    clock_t start, end;
    double totalEnergyConsumed = 0.0;

    // Start measuring time
    start = clock();
    int utilized = 0;
    for (int iteration = 0; iteration < MAX_ITERATIONS; ++iteration)
    {
        for (int i = 0; i < NUM_PARTICLES; ++i)
            updateParticle(&particles[i]);


        // Implement dynamic resource allocation at appropriate intervals
        if (iteration % 100 == 0)
            dynamicResourceAllocation();
    }

    // Stop measuring time
    end = clock();
    double totalResponseTime = ((double)(end - start)) / CLOCKS_PER_SEC;

    // Calculate throughput
    int tasksExecuted = 0;
    for (int i = 0; i < NUM_TASKS; ++i)
    {
        int serverId = g_bestParticle.assignment[i];
        double taskMemoryReq = tasks[i].memoryRequirements;
        double serverCapacity = servers[serverId].capacity;

        if (serverCapacity - servers[serverId].currentLoad >= taskMemoryReq)
            tasksExecuted++;

        // Calculate energy consumed for this task (you can adjust this value)
        double taskEnergy = 0.1 * taskMemoryReq; // An appropriate constant
        totalEnergyConsumed += taskEnergy;
    }

    double throughput = tasksExecuted / totalResponseTime;

    // Calculate energy efficiency
    double energyEfficiency = throughput / totalEnergyConsumed;

    // Output the best assignment found
    printf("Best Assignment:\n");
    for (int i = 0; i < NUM_TASKS; ++i)
    {
        int taskServerId = g_bestParticle.assignment[i];
        printf("Task %d -> Server %d\n", i, taskServerId);

        // Update server resource usage based on task allocation
        servers[taskServerId].currentLoad += tasks[i].cpuRequirements;

        // Output resource utilization
        printf("CPU Utilization on Server %d: %.2f%%\n", taskServerId,
               (servers[taskServerId].cpuUtilization));
        printf("Memory Utilization on Server %d: %.2f%%\n", taskServerId,
               (servers[taskServerId].memoryUtilization));
        printf("Burst Time on Server %d: %.2f seconds\n\n", taskServerId, tasks[i].execution_time);
    }
    printf("Global Best Fitness: %.8f\n", g_bestParticle.fitness);

    // Calculate resource utilization
    double totalResourcesUsed = 0.0;
    double totalServerCapacity = 0.0;

    for (int i = 0; i < NUM_TASKS; ++i)
    {
        int serverId = g_bestParticle.assignment[i];
        double taskMemoryReq = tasks[i].memoryRequirements;
        totalResourcesUsed += taskMemoryReq;
    }

    for (int i = 0; i < NUM_SERVERS; ++i)
        totalServerCapacity += servers[i].capacity;

    double resourceUtilizationPercentage = (totalResourcesUsed / totalServerCapacity) * 100.0;

    // Output additional metrics
    printf("Resource Utilization: %.2f%%\n", resourceUtilizationPercentage);
    printf("Total Response Time: %.2f seconds\n", totalResponseTime);
    printf("Total Energy Consumed: %.2f units\n", totalEnergyConsumed);
    printf("Energy Efficiency: %.2f tasks/Joule\n", energyEfficiency);
    printf("Throughput: %.2f tasks/second\n\n", throughput);
    return 0;
}