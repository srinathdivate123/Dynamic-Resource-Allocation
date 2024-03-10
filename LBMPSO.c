//  Load Balancing Multi-Objective Particle Swarm Optimization (LBMPSO) algorithm
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>

#define NUM_PROCESSES 10
#define NUM_THREADS 4
#define TMAX 100

struct Process
{
    int process_id;
    double EET; // Estimated Execution Time
    double TTs; // Total Transfer Size
    double length;
    double process_size;
    double burst_time;
};

struct Thread
{
    int thread_id;
    double MIPS; // Million Instructions Per Second
    double memory;
    double CPU;
    double load;
};

struct Particle
{
    int process_id;
    int thread_assigned;
    double velocity;
    double pbest_fitness;
    int pbest_thread;
};

// Function to initialize a particle (Process)
struct Particle initialize_particle(int process_id, int num_threads)
{
    struct Particle particle;
    particle.process_id = process_id;
    particle.thread_assigned = rand() % num_threads;
    particle.velocity = 0.0;
    particle.pbest_fitness = DBL_MIN; // Initialize pbest with a low value
    particle.pbest_thread = -1;
    return particle;
}

// Function to compute the fitness value for a particle (process)
double compute_fitness(struct Particle particle, struct Process processes[], struct Thread threads[], int num_processes, int num_threads)
{
    int process_id = particle.process_id;
    int thread_id = particle.thread_assigned;
    processes[process_id].burst_time = processes[process_id].length / (threads[thread_id].MIPS * threads[thread_id].CPU);

    threads[thread_id].load += processes[process_id].burst_time;

    double resource_utilization = threads[thread_id].load / (threads[thread_id].MIPS * threads[thread_id].CPU);

    return resource_utilization;
}

// Update Pbest for each particle
void update_pbest(struct Particle particles[], struct Process processes[], struct Thread threads[], int num_processes, int num_threads)
{
    for (int i = 0; i < num_processes; i++)
    {
        double current_fitness = compute_fitness(particles[i], processes, threads, num_processes, num_threads);
        if (current_fitness > particles[i].pbest_fitness)
        {
            particles[i].pbest_fitness = current_fitness;
            particles[i].pbest_thread = particles[i].thread_assigned;
        }
    }
}

// global best-known fitness
struct Particle find_gbest(struct Particle particles[], int num_processes)
{
    struct Particle gbest = particles[0];
    for (int i = 1; i < num_processes; i++)
        if (particles[i].pbest_fitness > gbest.pbest_fitness)
            gbest = particles[i];
    return gbest;
}

// LBMPSO (Load Balancing Multi-Objective Particle Swarm Optimization)
void run_lbmpso(struct Particle particles[], struct Process processes[], struct Thread threads[], int num_processes, int num_threads, int tmax)
{
    for (int iteration = 0; iteration < tmax; iteration++)
    {
        // Update Pbest for each particle
        update_pbest(particles, processes, threads, num_processes, num_threads);

        // global best-known fitness value
        struct Particle gbest = find_gbest(particles, num_processes);

        // Load balancing
        for (int i = 0; i < num_processes; i++)
        {
            int original_thread = particles[i].thread_assigned;
            double original_load = threads[original_thread].load;

            for (int j = 0; j < num_threads; j++)
                if (j != original_thread)
                {
                    particles[i].thread_assigned = j;
                    double new_load = threads[j].load + processes[i].burst_time;

                    if (new_load < threads[original_thread].load)
                    {
                        threads[original_thread].load = new_load;
                        threads[j].load = new_load;
                    }
                    else
                        particles[i].thread_assigned = original_thread;
                }
        }
    }
}

int main()
{
    srand(time(NULL));

    struct Process processes[NUM_PROCESSES];
    struct Thread threads[NUM_THREADS];
    struct Particle particles[NUM_PROCESSES];

    // Initialize processes
    for (int i = 0; i < NUM_PROCESSES; i++)
    {
        processes[i].process_id = i;
        processes[i].EET = rand() % 20 + 10;
        processes[i].TTs = rand() % 10 + 5;
        processes[i].length = rand() % 100 + 50;
        processes[i].process_size = rand() % 50 + 25;
        processes[i].burst_time = 0.0;
    }

    // Initialize Threads (preset for demonstration purposes)
    for (int i = 0; i < NUM_THREADS; i++)
    {
        threads[i].thread_id = i;
        threads[i].MIPS = 1000.0;   // Preset MIPS
        threads[i].memory = 4096.0; // Preset memory
        threads[i].CPU = 4.0;       // Preset CPU
        threads[i].load = 0.0;
    }

    for (int i = 0; i < NUM_PROCESSES; i++)
        particles[i] = initialize_particle(i, NUM_THREADS);

    run_lbmpso(particles, processes, threads, NUM_PROCESSES, NUM_THREADS, TMAX);

    // Find the best solution (Gbest) after the algorithm terminates
    struct Particle best_solution = find_gbest(particles, NUM_PROCESSES);

    printf("Optimal process assignment:\n");
    printf("Process ID %d is assigned to Thread %d\n", best_solution.process_id, best_solution.thread_assigned);

    printf("All processes assignments:\n");
    for (int i = 0; i < NUM_PROCESSES; i++)
        printf("Process ID %d is assigned to Thread %d\n", i, particles[i].thread_assigned);

    return 0;
}