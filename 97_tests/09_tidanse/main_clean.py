# Purpose of script:
# Evaluate behavior of (TI-)(GEVD-)DANSE algorithms on a simulated dataset,
# in batch and online mode, using stochastic signals and randomized 
# steering matrices.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS
# Created: 2023-11-13

import os
import sys
from utils.danse import Launcher
from utils.scene import SceneCreator
from utils.post import PostProcessor
from utils.config import Configuration

# Current working directory
CWD = os.path.dirname(os.path.realpath(__file__))
# YAML file containing configuration
YAML_FILE = f'{CWD}/params.yaml'
EXPORT = True
# EXPORT = False

def main():
    """Main function (called by default when running script)."""
    cfg = Configuration()
    cfg.from_yaml(relative_to_absolute_path(YAML_FILE))

    mmsePerAlgo, mmseCentral, filtersPerAlgo, filtersCentral, RyyPerAlgo, RnnPerAlgo = [], [], [], [], [], []
    for nMC in range(cfg.mcRuns):
        print(f"MC run {nMC+1}/{cfg.mcRuns}")
        # Create acoustic scene
        scene = SceneCreator(cfg)
        scene.prepare_scene()
        # Launch simulation
        sim = Launcher(scene)
        sim.run()
        # Save results
        mmsePerAlgo.append(sim.mmsePerAlgo)
        mmseCentral.append(sim.mmseCentral)
        filtersPerAlgo.append(sim.filtersPerAlgo)
        filtersCentral.append(sim.filtersCentral)
        RyyPerAlgo.append(sim.RyyPerAlgo)
        RnnPerAlgo.append(sim.RnnPerAlgo)
    
    # Post-process results
    pp = PostProcessor(
        mmsePerAlgo, mmseCentral,
        filtersPerAlgo, filtersCentral,
        RyyPerAlgo, RnnPerAlgo,
        sim.vadSaved,
        cfg, export=EXPORT
    )
    pp.perform_post_processing()
# 
    return 0


def relative_to_absolute_path(path: str):
    """Convert relative path to absolute path."""
    return os.path.join(os.path.dirname(__file__), path)
    

if __name__ == '__main__':
    sys.exit(main())