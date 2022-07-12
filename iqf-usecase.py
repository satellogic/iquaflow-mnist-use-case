import os

from iquaflow.datasets import DSWrapper
from iquaflow.experiments import ExperimentInfo, ExperimentSetup
from iquaflow.experiments.task_execution import PythonScriptTaskExecution

from custom_iqf import CustomNoiseModifier

#Define name of IQF experiment
experiment_name = "iq-mnist-use-case"

#Define path of the original(reference) dataset
data_path_train = "data/mnist_png/training"
data_path_valid = "data/mnist_png/validation"

#DS wrapper is the class that encapsulate a dataset
ds_wrapper_train = DSWrapper(data_path=data_path_train)
ds_wrapper_valid = DSWrapper(data_path=data_path_valid)

#Define path of the training script
python_ml_script_path = 'custom_train.py'

#List of modifications that will be applied to the original dataset:

ds_modifiers_list = [
    CustomNoiseModifier(params = {"sigma": 1*f})
    for f in [0,0.5,1,2,4,8,16,32]
]

# Task execution executes the training loop
# In this case the training loop is an empty script,
# this is because we evaluate the performance directly on the result of the modifiers.
task = PythonScriptTaskExecution( model_script_path = python_ml_script_path )

#Experiment definition, pass as arguments all the components defined beforehand
experiment = ExperimentSetup(
    experiment_name   = experiment_name,
    task_instance     = task,
    ref_dsw_train     = ds_wrapper_train,
    ref_dsw_val       = ds_wrapper_valid,
    ds_modifiers_list = ds_modifiers_list,
    repetitions       = 1
)

#Execute the experiment
experiment.execute()
