#import multiprocessing
#import time
import json
import os
import pickle
from configparser import ConfigParser
from ax.service.managed_loop import optimize

from hex.training.repeated_self_training import RepeatedSelfTrainer, load_reference_models
from hex.utils.logger import logger


class BayesianOptimization:
    """
    runs Bayesian Optimization with given parameters and 20 steps
    optimizes by ELO value compared to starting model
    """
    def __init__(self, parameters, config):
        self.parameters = parameters
        self.config = config
        self.reference_models = load_reference_models(config)

    def train_evaluate(self, parameters):
        trainer = RepeatedSelfTrainer(self.config)
        trainer.reference_models = self.reference_models

        for parameter_name, value in parameters.items():
            logger.info(f"{parameter_name}: {value}")
            section = next(parameter["section"] for parameter in self.parameters
                if parameter["name"] == parameter_name)
            trainer.config[section][parameter_name] = str(value)
        trainer.config["REPEATED SELF TRAINING"]["num_iterations"] = \
            str(int(1/trainer.config["TRAIN"].getfloat("epochs")))
        logger.info(f"num_iterations: {trainer.config['REPEATED SELF TRAINING']
            .getint('num_iterations')}")

        trainer.repeated_self_training()

        return trainer.get_best_rating()

    def optimizer_run(self):
        best_parameters, values, experiment, model = optimize(
            parameters=self.parameters,
            evaluation_function=self.train_evaluate,
            objective_name='elo',
            total_trials=20
        )
        return best_parameters, values, experiment, model


def main():
    parameters_path = "bo_parameters.json"
    if not os.path.isfile(parameters_path):
        with open(parameters_path, 'w') as file:
            file.write("[]")

    with open("bo_parameters.json") as file:
        parameters = json.load(file)

    if parameters == []:
        logger.info(f"parameter list in file {parameters_path} is empty")
        raise SystemExit

    config = ConfigParser()
    config.read('config.ini')
    bo = BayesianOptimization(parameters, config)

    try:
        best_parameters, values, experiment, model = bo.optimizer_run()

        logger.info("=== best parameters are ===")
        logger.info(best_parameters)

        with open("bo_results.p", "wb") as file:
            pickle.dump((best_parameters, values), file)
        #ax.save(experiment, "bo_results.json")

    except KeyboardInterrupt:
        print("Shutdown requested...exiting")

    raise SystemExit


if __name__ == '__main__':
    main()
