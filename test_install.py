import jumpmetrics
print(f"Successfully imported jumpmetrics version {jumpmetrics.__version__}")

# Test importing specific modules
from jumpmetrics.core.core import ForceTimeCurveCMJTakeoffProcessor
print('Successfully imported ForceTimeCurveCMJTakeoffProcessor')