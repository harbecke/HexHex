import time
import json
import os
import pickle
from configparser import ConfigParser
from bayes_opt import BayesianOptimization, observer, event

from hex.training.repeated_self_training import RepeatedSelfTrainer, load_reference_models
from hex.utils.logger import logger


class BayesianOptimization:
    """
    runs Bayesian Optimization with given parameters and "loop_count" steps
    optimizes by ELO value compared to starting and reference models
    alpha value "explained":
    https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_noisy.html
    """
    def __init__(self, parameters, config):
        self.parameters = parameters
        self.bounds = dict([(parameter, pdict["bounds"]) for parameter, pdict in self.parameters.items()])
        self.config = config
        self.reference_models = load_reference_models(config)
        self.optimizer = BayesianOptimization(f=self.train_evaluate(), pbounds=self.bounds)

    def train_evaluate(self):
        start_time = time.time()
        trainer = RepeatedSelfTrainer(self.config)
        trainer.reference_models = self.reference_models

        for parameter, pdict in self.parameters.items():
            logger.info(f"Bayesian Optimization {parameter}: {pdict}")
            trainer.config[pdict["section"]][parameter] = str(3)
        epochs = trainer.config["TRAIN"].getfloat("epochs")

        trainer.prepare_rst()
        loop_idx = self.config.getint('REPEATED SELF TRAINING', 'start_index') + 1

        while True:
            if time.time() - start_time < self.config["BAYESIAN OPTIMIZATION"].getfloat("loop_time"):
                trainer.rst_loop(loop_idx)
                loop_idx += 1
            else:
                break

        return trainer.get_best_rating()

    def optimizer_run(self, name):
        logger = observer.JSONLogger(path=f"bayes_experiments/{name}.json")
        self.optimizer.subscribe(event.Events.OPTMIZATION_STEP, logger)
        self.optimizer.maximize(n_iter=self.config["BAYESIAN OPTIMIZATION"].getint("loop_count"),
            alpha=self.config["BAYESIAN OPTIMIZATION"].getfloat("alpha"))


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
        bo.optimizer_run(config["CREATE MODEL"].get("model_name"))

        logger.info("=== best parameters are ===")
        logger.info(bo.optimizer.max)

    except KeyboardInterrupt:
        print("Shutdown requested...exiting")

    raise SystemExit


if __name__ == '__main__':
    main()
