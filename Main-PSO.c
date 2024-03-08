#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#define NUM_THREADS 4
#define NUM_PROCESSES 10
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
} Thread;

typedef struct
{
    int id;
    double cpuRequirements;
    double memoryRequirements;
    double burst_time;
} Process;

typedef struct
{
    int assignment[NUM_PROCESSES];
    double fitness;
    int p_best[NUM_PROCESSES];
} Particle;

Thread threads[NUM_THREADS];
Particle particles[NUM_PARTICLES];
Particle g_bestParticle;
Process processes[NUM_PROCESSES];

double evaluateFitness(Particle *particle);

void initialize()
{
    srand(time(NULL));

    // Threads with random capacities within constraints
    for (int i = 0; i < NUM_THREADS; ++i)
    {
        threads[i].id = i;
        threads[i].capacity = MIN_CAPACITY + (rand() % (MAX_CAPACITY - MIN_CAPACITY + 1));
        threads[i].currentLoad = 0.0;
        threads[i].cpuUtilization = 0.0;
        threads[i].memoryUtilization = 0.0;
    }

    // Processes with random CPU & memory requirements and execution times
    for (int i = 0; i < NUM_PROCESSES; ++i)
    {
        processes[i].id = i;
        processes[i].cpuRequirements = MIN_CPU + (rand() % (MAX_CPU - MIN_CPU + 1));
        processes[i].memoryRequirements = MIN_MEMORY + (rand() % (MAX_MEMORY - MIN_MEMORY + 1));
        processes[i].burst_time = MIN_EXECUTION_TIME + (rand() % (MAX_EXECUTION_TIME - MIN_EXECUTION_TIME + 1));
    }

    // Initialize particles with random assignments
    for (int i = 0; i < NUM_PARTICLES; ++i)
    {
        for (int j = 0; j < NUM_PROCESSES; ++j)
            particles[i].assignment[j] = rand() % NUM_THREADS;
        particles[i].fitness = evaluateFitness(&particles[i]);
        for (int j = 0; j < NUM_PROCESSES; ++j)
            particles[i].p_best[j] = particles[i].assignment[j];
    }

    // Initialize the fittness of global best particle to -INFINITY 
    g_bestParticle.fitness = -INFINITY;
}

double evaluateFitness(Particle *particle)
{
    // Calculate the total resource wastage across all threads
    double totalWastage = 0.0;

    // Calculate the load on each thread based on process assignments
    for (int i = 0; i < NUM_THREADS; ++i)
    {
        threads[i].currentLoad = 0.0;
        threads[i].cpuUtilization = 0.0;
        threads[i].memoryUtilization = 0.0;
    }

    for (int i = 0; i < NUM_PROCESSES; ++i)
    {
        int threadId = particle->assignment[i];
        double processMemoryReq = processes[i].memoryRequirements;
        double threadCapacity = threads[threadId].capacity;

        // Update thread load and calculate wastage
        threads[threadId].currentLoad += processMemoryReq;
        double wastage = fmax(0, threads[threadId].currentLoad - threadCapacity);
        totalWastage += wastage;

        // Update thread CPU and memory utilization
        threads[threadId].cpuUtilization += processes[i % NUM_PROCESSES].cpuRequirements;
        threads[threadId].memoryUtilization += processes[i % NUM_PROCESSES].memoryRequirements;
    }

    // Calculate CPU and Memory Utilization as a percentage
    for (int i = 0; i < NUM_THREADS; ++i)
    {
        threads[i].cpuUtilization = (threads[i].cpuUtilization / threads[i].capacity) * 100.0;
        threads[i].memoryUtilization = (threads[i].memoryUtilization / threads[i].capacity) * 100.0;
    }

    // Fitness is the negative of total wastage (minimize wastage)
    double fitness = -totalWastage;

    return fitness;
}

void dynamicResourceAllocation()
{
    // Monitor thread loads and adjust capacities dynamically
    for (int i = 0; i < NUM_THREADS; ++i)
    {
        threads[i].cpuUtilization = threads[i].cpuUtilization / threads[i].capacity;
        threads[i].memoryUtilization = threads[i].memoryUtilization / threads[i].capacity;
        printf("Thread %d CPU Utilization is %.2f%%\n", i, threads[i].cpuUtilization*100);

        if (threads[i].cpuUtilization > 0.70)
        {
            threads[i].capacity += 15; // Increase capacity of thread
            printf("Increased CPU Utlization of thread %d by 15 units\n\n", i);
        }
        else if (threads[i].cpuUtilization < 0.2)
        {
            threads[i].capacity -= 15; // Decrease capacity of thread
            printf("Decreased CPU Utlization of thread %d by 15 units\n\n", i);
        }

        // Ensure capacity remains within constraints
        threads[i].capacity = fmax(MIN_CAPACITY, fmin(MAX_CAPACITY, threads[i].capacity));
    }
}

void updateParticle(Particle *particle)
{
    // These random values are used to introduce stochasticity into the PSO algorithm
    // The values are used to compute the cognitive and social components of the velocity update

    double cognitiveComponent = (double)rand() / RAND_MAX;
    double socialComponent = (double)rand() / RAND_MAX; 

    for (int i = 0; i < NUM_PROCESSES; ++i)
    {
        double inertiaTerm = INERTIA_WEIGHT * particle->p_best[i];
        double cognitiveTerm = COGNITIVE_WEIGHT * cognitiveComponent * (particle->p_best[i] - particle->assignment[i]);
        double socialTerm = SOCIAL_WEIGHT * socialComponent * (g_bestParticle.assignment[i] - particle->assignment[i]);

        // These three terms together contribute to the calculation of velocityUpdate, which represents how the particle should adjust its process assignments (position) in the solution space
        // The balance between these terms guides the particle's movement as it seeks to explore and exploit the solution space to find better assignments that optimize the given fitness function

        double heuristicValue = inertiaTerm + cognitiveTerm + socialTerm; // This is the velocity of the particle
        double newAssignment = particle->assignment[i] + heuristicValue;

        // Ensure that newAssignment respects thread capacities and constraints
        if (newAssignment < 0)
            newAssignment = 0;
        else if (newAssignment >= NUM_THREADS)
            newAssignment = NUM_THREADS - 1;

        particle->assignment[i] = (int)newAssignment;
    }

    // Update p_best if needed
    double currentFitness = evaluateFitness(particle);
    if (currentFitness > particle->fitness)
    {
        particle->fitness = currentFitness;
        for (int i = 0; i < NUM_PROCESSES; ++i)
            particle->p_best[i] = particle->assignment[i];
    }

    // Update g_bestParticle if needed
    if (currentFitness > g_bestParticle.fitness)
    {
        g_bestParticle.fitness = currentFitness;
        for (int i = 0; i < NUM_PROCESSES; ++i)
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
    int programsProcessed = 0;
    for (int i = 0; i < NUM_PROCESSES; ++i)
    {
        int threadId = g_bestParticle.assignment[i];
        double processMemoryReq = processes[i].memoryRequirements;
        double threadCapacity = threads[threadId].capacity;

        if (threadCapacity - threads[threadId].currentLoad >= processMemoryReq)
            programsProcessed++;

        // Calculate energy consumed for this process (you can adjust this value)
        double programEnergy = 0.1 * processMemoryReq; // An appropriate constant
        totalEnergyConsumed += programEnergy;
    }

    double throughput = programsProcessed / totalResponseTime;

    // Calculate energy efficiency
    double energyEfficiency = throughput / totalEnergyConsumed;

    // Output the best assignment found
    printf("Best Assignment:\n");
    for (int i = 0; i < NUM_PROCESSES; ++i)
    {
        int programThreadId = g_bestParticle.assignment[i];
        printf("Process %d -> Thread %d\n", i, programThreadId);

        // Update thread resource usage based on process allocation
        threads[programThreadId].currentLoad += processes[i].cpuRequirements; // Change programId to i

        // Output resource utilization
        printf("CPU Utilization on Thread %d: %.2f%%\n", programThreadId,
               (threads[programThreadId].cpuUtilization));
        printf("Memory Utilization on Thread %d: %.2f%%\n", programThreadId,
               (threads[programThreadId].memoryUtilization));
        printf("Burst Time on Thread %d: %.2f seconds\n\n", programThreadId, processes[i].burst_time); // Change programId to i
    }
    printf("Global Best Fitness: %f\n", g_bestParticle.fitness);

    // Calculate resource utilization
    double totalResourcesUsed = 0.0;
    double totalThreadCapacity = 0.0;

    for (int i = 0; i < NUM_PROCESSES; ++i)
    {
        int threadId = g_bestParticle.assignment[i];
        double processMemoryReq = processes[i].memoryRequirements;
        totalResourcesUsed += processMemoryReq;
    }

    for (int i = 0; i < NUM_THREADS; ++i)
        totalThreadCapacity += threads[i].capacity;

    double resourceUtilizationPercentage = (totalResourcesUsed / totalThreadCapacity) * 100.0;

    // Output additional metrics
    printf("Resource Utilization: %.2f%%\n", resourceUtilizationPercentage);
    printf("Total Response Time: %.2f seconds\n", totalResponseTime);
    printf("Total Energy Consumed: %.2f units\n", totalEnergyConsumed);
    printf("Energy Efficiency: %.2f processes/Joule\n", energyEfficiency);
    printf("Throughput: %.2f processes/second\n\n", throughput);
    return 0;
}
