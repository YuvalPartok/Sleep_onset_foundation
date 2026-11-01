import neuralbench
from neuralbench.registry import ALL_MODELS, TASKS

print("EEG tasks:", ", ".join(TASKS["eeg"]))
print("Available models:", ", ".join(ALL_MODELS))