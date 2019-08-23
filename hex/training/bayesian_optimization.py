import time
import json
import os
import pickle
from configparser import ConfigParser
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

from hex.training.repeated_self_training import RepeatedSelfTrainer, load_reference_models
from hex.utils.logger import logger


def parameter_dict_to_named_arg(pdict):
    low, high = pdict["bounds"]
    if type(low) == type(high):
        if type(low) == float:
            prior = "log-uniform" if pdict.get("log_scale") else "uniform"
            return Real(low=low, high=high, prior=prior, name=pdict["name"])
        elif type(low) == int:
            return Int(low=low, high=high, name=pdict["name"])
    else:
        logger.info(f"=== parameter {pdict['name']} doesn't match (known) types ===")
        raise SystemExit

def bayesian_optimization():
    #runs Bayesian Optimization with given parameters and "loop_count" steps
    #optimizes by ELO value compared to starting and reference models

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
    reference_models = load_reference_models(config)
    space = [parameter_dict_to_named_arg(pdict) for pdict in parameters]

    @use_named_args(space)
    def train_evaluate(**params):
        trainer = RepeatedSelfTrainer(config)
        trainer.reference_models = reference_models

        start_time = time.time()
        for parameter_name, value in params.items():
            logger.info(f"Bayesian Optimization {parameter_name}: {value}")
            section = next(parameter["section"] for parameter in parameters
                if parameter["name"] == parameter_name)
            trainer.config[section][parameter_name] = str(value)
        epochs = trainer.config["TRAIN"].getfloat("epochs")

        trainer.prepare_rst()
        loop_idx = config.getint('REPEATED SELF TRAINING', 'start_index') + 1

        while True:
            if time.time() - start_time < config["BAYESIAN OPTIMIZATION"].getfloat("loop_time"):
                trainer.rst_loop(loop_idx)
                loop_idx += 1
            else:
                break

        return trainer.get_best_rating()

    res_gp = gp_minimize(
        func=train_evaluate,
        dimensions=space,
        n_calls=config["BAYESIAN OPTIMIZATION"].getint("loop_count"),
        n_random_starts=config["BAYESIAN OPTIMIZATION"].getint("random_count"),
        noise=config["BAYESIAN OPTIMIZATION"].getfloat("noise")
        )

    logger.info("=== best parameters are ===")
    logger.info(res_gp.x, res_gp.fun)


if __name__ == '__main__':
    bayesian_optimization()
