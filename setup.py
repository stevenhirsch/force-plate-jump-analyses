"""Python package setup"""
import os
from setuptools import setup, Command

class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options = []

    def initialize_options(self):
        """
        Define initial options for the clean command.

        This method is required by setuptools.Command but is empty 
        as this command doesn't have any options.
        """

    def finalize_options(self):
        """
        Finalize options for the clean command.

        This method is required by setuptools.Command but is empty 
        as this command doesn't have any options to finalize.
        """

    def run(self):
        """
        Execute the clean command.

        This method removes build artifacts, distribution files,
        compiled Python files, and egg-info directories.
        """
        os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info')


setup(
    cmdclass={
        'clean': CleanCommand,
    }
)
