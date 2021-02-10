# ==== TESTED FOR QISKIT 0.10.1 =========

# Basic Modules
import os
import datetime
import sys
import math
import matplotlib.pyplot as plt
import numpy as np
import pprint as pp
from scipy import linalg
import time
import json
import pandas as pd

from IPython.display import clear_output

# Qiskit Modules
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute
from qiskit import IBMQ
from qiskit import Aer

"""
from qiskit import load_qasm_string
IS DEPRECATED, USE INSTEAD
QuantumCircuit.fromm_qasm_str()
"""

# Circuit Visualization
from qiskit.visualization.circuit_visualization import _latex_circuit_drawer as latex_circuit_drawer
from qiskit.visualization.circuit_visualization import circuit_drawer
from qiskit.visualization.circuit_visualization import _matplotlib_circuit_drawer as matplotlib_circuit_drawer

# Backend Visualization (Qiskit 0.10.1)
# This requires you to apply a few bug-fixes to qiskit.
# 1. Go to wherever the qiskit python package is stored, (for me in /usr/local/lib/Python3.7/site-packages/qiskit).
# 2. In qiskit/visualization/__init__.py, change "from qiskit._util import _has_connection" to "from qiskit.util import _has_connection"
# 3. In _backend_monitor.py and _gate_map.py, change "from qiskit.qiskiterror import QISKitError" to "from qiskit.qiskiterror import QiskitError"
# 4. Then you can load _backend_monitor:
from qiskit.tools.jupyter import backend_monitor
from qiskit.tools.jupyter import backend_overview

#from qiskit.tools.jupyter import _backend_monitor as backend_monitor
#from qiskit.tools.jupyter import _backend_overview as backend_overview

""" Usage: 
backend = IBMQ.get_backend('ibmq_20_tokyo')
_backend_monitor.detailed_map(backend)
_backend_monitor.gates_tab(backend)
_backend_monitor.qubits_tab(backend)
_backend_monitor.config_tab(backend)
"""
# AQUA
# from qiskit.aqua.input import EnergyInput
from qiskit.aqua.components.optimizers import SPSA, NELDER_MEAD, COBYLA
# from qiskit.aqua.components.variational_forms import RY
from qiskit.aqua.algorithms import VQE, QAOA, ExactEigensolver
# from qiskit.aqua.algorithms.adaptive.qaoa.var_form import QAOAVarForm
from qiskit.aqua import Operator, run_algorithm
from qiskit.quantum_info import Pauli  # For constructing Operators
from qiskit.aqua import QuantumInstance

# JOBSTATUS
from qiskit.providers import ibmq
from qiskit.providers.jobstatus import JobStatus

# TRANSPILER
from qiskit.compiler import transpile, assemble
from qiskit.transpiler import PassManager
from qiskit.transpiler import passes
from qiskit.transpiler import CouplingMap # CouplingMap([[0, 1]])

# DAG
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.visualization import dag_drawer
from qiskit.transpiler.basepasses import TransformationPass # For writing your own passes that modify the DAG
from qiskit.transpiler.passes import Unroller

from qiskit.extensions.standard.u3 import U3Gate
from qiskit.extensions.standard.cx import CnotGate


# FIXED BUG: In "cx_cancellation.py", need dag.collect_runs(["CX"]), not dag.collect_runs(["cx"])


# PARAMETERIZED CIRCUITS
from qiskit.circuit import Parameter


# IGNIS

# Measurement mitigation
from qiskit.ignis.mitigation.measurement import (complete_meas_cal,
                                                 CompleteMeasFitter, 
                                                 MeasurementFilter)
from qiskit.ignis.mitigation.measurement import (
    tensored_meas_cal,
    TensoredMeasFitter,
    TensoredFilter)

# Process Tomography
from qiskit.ignis.verification.tomography import (
    state_tomography_circuits,
    process_tomography_circuits,
    ProcessTomographyFitter,
    StateTomographyFitter
)

""" PULSE NOTE: In qiskit/pulse/schedule.py, in function 'draw', change 
    from qiskit.tools import visualization 
    to
    from qiskit import visualization
"""

""" PULS NOTE: In qiskit/compiler/assemble.py, in function 'assemble', after line "elif all(isinstance(exp, ScheduleComponent) for exp in experiments):", add the line:

run_config.rep_time = int(run_config.rep_time)

"""

from qiskit.pulse import pulse_lib, Schedule

from qiskit.pulse.channels import (DriveChannel, 
                                   MeasureChannel, 
                                   ControlChannel, 
                                   AcquireChannel, 
                                   MemorySlot, 
                                   SnapshotChannel,
                                  PulseChannelSpec)
from qiskit.pulse.commands import (SamplePulse, 
                                   FrameChange, 
                                   PersistentValue,
                                   Acquire, 
                                   Snapshot)

from scipy.optimize import curve_fit



print("modules.py loaded.")
print ('Current date/time: {}'.format(datetime.datetime.now()))  
