import multiprocessing
import time
from ax.service.managed_loop import optimize

from hex.training.repeated_self_training import RepeatedSelfTrainer


class BayesianOptimization:
    def __init__(self, parameters):
        self.parameters = parameters

    def train_evaluate(self):
        trainer = RepeatedSelfTrainer("config.ini")

        for key, value in parameters.items():
            section = next(parameter["section"] for parameter in self.parameters if parameter["name"] == key)
            trainer.config[self.parameters[key][section]][key] = str(value)

        trainer.repeated_self_training()
        #p = multiprocessing.Process(target=trainer.repeated_self_training())
        #p.start()

        # Wait for 10 seconds or until process finishes
        #p.join(10)

        # If thread is still active
        #if p.is_alive():

            # Terminate
            #p.terminate()
            #p.join()
        return trainer.get_best_rating()

    def optimizer_run(self):
        best_parameters, values, experiment, model = optimize(
            parameters=self.parameters,
            evaluation_function=self.train_evaluate,
            objective_name='elo'
        )
        return best_parameters, values, experiment, model


if __name__ == '__main__':
    parameters = [
            {"name": "learning_rate", "type": "range", "bounds": [1e-6, 1e-3], "log_scale": True, "section": "TRAIN"},
            {"name": "weight_decay", "type": "range", "bounds": [1e-8, 1.0], "log_scale": True, "section": "TRAIN"}
        ]
    bo = BayesianOptimization(parameters)
    best_parameters, values, experiment, model = bo.optimizer_run()

    print(best_parameters)
    means, covariances = values
    print(means, covariances)
