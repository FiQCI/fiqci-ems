# Logging

Configuring logging for FiQCI EMS with Qiskit and M3.

This python package has useful log messages added, however, Qiskit and M3 have quite verbose logging at the `INFO` level. Therefore, a simple logging setup would be


```python
import logging

# Configure logging to see FiQCI EMS messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Set specific logger levels
logging.getLogger('fiqci.ems').setLevel(logging.INFO)
logging.getLogger('mthree').setLevel(logging.WARNING)
logging.getLogger('qiskit').setLevel(logging.WARNING)
```
