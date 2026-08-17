from numguard_fin.cli import main

commands = [
    [
        "inspect",
        "--path",
        "data/fixtures/finqa_structured_fixture.json",
        "--split",
        "dev",
        "--output",
        "validation/dataset_audit.csv",
    ],
    [
        "run",
        "--path",
        "data/fixtures/finqa_structured_fixture.json",
        "--split",
        "dev",
        "--engineering-check",
        "--output-dir",
        "validation/engineering_checks",
        "--figures-dir",
        "validation/engineering_check_figures",
    ],
    [
        "validate",
        "--predictions",
        "validation/engineering_checks/predictions.csv",
        "--maximum-candidates",
        "160",
        "--allow-engineering",
    ],
    [
        "counterfactual",
        "--path",
        "data/fixtures/finqa_structured_fixture.json",
        "--split",
        "dev",
        "--output",
        "validation/counterfactual_audit.csv",
    ],
]

for command in commands:
    code = main(command)
    if code:
        raise SystemExit(code)
