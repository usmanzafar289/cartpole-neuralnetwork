import os
import neat
import multiprocessing

class NeuralNetworkAgent:
    env = None
    config = None
    
    def __init__(self, env):
        self.env = env
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'config')
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'config')
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    def run(self):
        pop = neat.Population(self.config)
        pop.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        pop.add_reporter(stats)
        pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), self.sample_genome)
        winner = pop.run(pe.evaluate)
        
        net = neat.nn.FeedForwardNetwork.create(winner, self.config)
        observation = self.env.reset()
        done = False
        i = 0
        while not done:
            self.env.render()
            i += 1
            action = round(net.activate(observation)[0])
            observation, reward, done, info = self.env.step(action)

    def sample_genome(self, genome, config):
        fitnesses = []
        for i in range(5):
            fitnesses.append(self.sample(genome, config))
        return min(fitnesses)

    def sample(self, genome, config):
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        observation = self.env.reset()
        done = False
        i = 0
        while not done:
            i += 1
            action = round(net.activate(observation)[0])
            observation, reward, done, info = self.env.step(action)
            if done:
                return self.fitness(i)

    def fitness(self, fitness):
        return fitness