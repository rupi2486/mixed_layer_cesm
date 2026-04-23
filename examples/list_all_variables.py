"""
List every variable available in the CESM2-LE S3 archive,
grouped by component / scenario / forcing.

Run:
    python examples/list_all_variables.py
"""

from grab_cesm import COMPONENTS, FORCINGS, SCENARIOS, list_variables

for component in COMPONENTS:
    for scenario in SCENARIOS:
        for forcing in FORCINGS:
            try:
                variables = list_variables(component, scenario, forcing)
            except Exception as e:
                print(f"  {component} / {scenario} / {forcing}: ERROR — {e}")
                continue
            if variables:
                print(f"\n{component} / {scenario} / {forcing}")
                for v in variables:
                    print(f"  {v}")
