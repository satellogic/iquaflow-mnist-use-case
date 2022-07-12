import numpy as np

from iquaflow.datasets import DSModifier_dir

def add_noise( img, mean=0, var=0.1 ):
    """This function adds noise to an array"""
    row,col= img.shape
    noise = np.random.normal(mean,var**0.5,(row,col))
    return img + noise.reshape(row,col)

class CustomNoiseModifier(DSModifier_dir):
    
    def __init__(
       self,
       ds_modifier = None,
       params = {"variance": 256*10},
    ):
        sufix = str(params['variance'])
        self.name = f"noise_{sufix}"
        self.params: Dict[str, Any] = params
        self.ds_modifier = ds_modifier
        self.params.update({"modifier": "{}".format(self._get_name())})

    def _mod_img(self, img):
        return add_noise( img, mean=0, var=self.params['variance'] )