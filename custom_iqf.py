import numpy as np

from iquaflow.datasets import DSModifier_dir

def add_noise( img, mean=0, sigma=0.1 ):
    """This function adds noise to an array"""
    row,col= img.shape
    noise = np.random.normal(mean,sigma,(row,col))
    return np.clip( img + noise.reshape(row,col), 0, 1)

class CustomNoiseModifier(DSModifier_dir):
    
    def __init__(
       self,
       ds_modifier = None,
       params = {"sigma": 10},
    ):
        sufix = str(params['sigma'])
        self.name = f"noise_{sufix}"
        self.params: Dict[str, Any] = params
        self.ds_modifier = ds_modifier
        self.params.update({"modifier": "{}".format(self._get_name())})

    def _mod_img(self, img):
        return 255*add_noise( img/255, mean=0, sigma=self.params['sigma'] )