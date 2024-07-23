from pytorch_lightning.cli import LightningCLI
import os
from model.main import DepthNetModule

def cli_main():
    """The main function for the CLI."""
    cli = LightningCLI(DepthNetModule, datamodule_class=None)

if __name__ == "__main__":
    cli_main()
