# Purpose of script:
# Contain SRO-related functions for DANSE implementation.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS
# Created: 2023-11-24

import numpy as np
from .scene import WASN
from .config import Configuration

class EventCreator:

    def __init__(self, cfg: Configuration, wasn: WASN):
        self.cfg = cfg
        self.wasn = wasn

    def events_creator(self):
        if self.cfg.mode == 'online':
            self.get_events_online()
        elif self.cfg.mode == 'wola':
            raise NotImplementedError('Event handling for SRO-affected WOLA mode not implemented yet.')
        elif self.cfg.mode == 'batch':
            pass  # no need for events in batch mode

    def get_events_online(self):
        """Get events for online DANSE."""
        blockSize = self.cfg.B
        hopSize = self.cfg.overlapB
        Ns = int(blockSize * (1 - hopSize))
        # Create time vectors
        # TODO


