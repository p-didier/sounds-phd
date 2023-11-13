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

YAML_FILE = 'params.yaml'

def main():
    """Main function (called by default when running script)."""
    
    cfg = Configuration()
    cfg.from_yaml(relative_to_absolute_path(YAML_FILE))

    mmsePerAlgo, mmseCentral = [], []
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
    
    # Post-process results
    fig, axes = PostProcessor(mmsePerAlgo, mmseCentral, cfg).plot_mmse()
    # Export
    if not os.path.exists(cfg.exportFolder):
        os.makedirs(cfg.exportFolder)
    fig.savefig(os.path.join(cfg.exportFolder, 'mmse.pdf'), bbox_inches='tight')




def relative_to_absolute_path(path: str):
    """Convert relative path to absolute path."""
    return os.path.join(os.path.dirname(__file__), path)
    

if __name__ == '__main__':
    sys.exit(main())