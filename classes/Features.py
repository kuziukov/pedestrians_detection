import numpy as np


class Features(object):

    def __init__(self, templates):
        self._templates = templates

    def generate(self, block) -> list:
        response = []
        for template in self._templates:
            x, y, size, W = template
            w, h = size
            for channel in range(block.shape[2]):
                box = np.copy(block[y:y + h, x:x + w, channel])
                response.append(np.sum(np.multiply(box, W)))
        return response
