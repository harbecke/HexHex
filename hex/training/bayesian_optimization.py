import multiprocessing
import time
from ax.service.managed_loop import optimize

from hex.training.repeated_self_training import RepeatedSelfTrainer


def train_evaluate(parameters):
    trainer = RepeatedSelfTrainer("config.ini")

    for key, value in parameters.items():
        trainer.config["TRAIN"][key] = str(value)
    
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

if __name__ == '__main__':
    parameters = [
            {"name": "learning_rate", "type": "range", "bounds": [1e-6, 1e-3], "log_scale": True, "section": "TRAIN"},
            {"name": "weight_decay", "type": "range", "bounds": [1e-8, 1.0], "log_scale": True, "section": "TRAIN"}
        ]
    best_parameters, values, experiment, model = optimize(
        parameters=parameters,
        evaluation_function=train_evaluate,
        objective_name='elo'
    )
    print(best_parameters)
    means, covariances = values
    print(means, covariances)
