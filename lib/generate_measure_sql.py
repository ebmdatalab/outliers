"""Gets OpenPrescribing measure JSON and makes SQL for each measure.

Attributes:
    changes (dict): Dictionary of changes to be made
                    to the OpenPrescribing measure JSON
"""
import json
import os
import requests
from jsondiff import diff


def get_measure_json(measures, run_name):
    """Gets measure JSON from OpenPrescribing github repo
    and writes to a local file
    
    Args:
        measures (list): List of measures to be imported
        run_name (str): Label for running these measures
                        to create a folder
    """
    output_dir = f"data/measure_json/{run_name}"
    os.makedirs(output_dir, exist_ok=True)
    for measure in measures:
        filename = os.path.join(output_dir, f"{measure}.json")
        url = f"https://raw.githubusercontent.com/ebmdatalab/openprescribing/db97f60eb914f55e5914d2c7d17416e4bcc37630/openprescribing/measure_definitions/{measure}.json"
        r = requests.get(url)
        with open(filename, "wb") as f:
            f.write(r.content)


changes = {
    "numerator_columns": {
        "from": ["SUM(actual_cost) AS numerator"],
        "to": ["SUM(items) AS numerator"],
    }
}


def modify_measure_json(path):
    """Selectively modifies fields within the measure JSON
    according to changes dict
    
    Args:
        path (str): Path to locate measure JSON to be modified
    
    Returns: 
        JSON: Modified measure JSON
    """
    with open(path) as f:
        measure_def = json.load(f)
        measure_def_before = measure_def.copy()
    for c in changes:
        if c in measure_def:
            if measure_def[c] == changes[c]["from"]:
                measure_def[c] = changes[c]["to"]
                print(
                    "Changes from OP definition:", diff(measure_def_before, measure_def)
                )
    with open(path, "w") as f:
        json.dump(measure_def, f)
    return measure_def


# This is adapted from frontend/management/commands/import_measures.py
def build_num_or_denom_fields(measure_def, num_or_denom):
    def full_attr_name(attr):
        return num_or_denom + "_" + attr

    def get_measure_value(attr):
        value = measure_def[full_attr_name(attr)]
        if isinstance(value, list):
            value = " ".join(value)
        value = value.replace("{", "")
        value = value.replace("}", "")
        return value

    type_ = measure_def[num_or_denom + "_type"]

    if type_ == "custom":
        columns = get_measure_value("columns")
        from_ = get_measure_value("from")
        where = get_measure_value("where")

    elif type_ == "bnf_items":
        columns = "SUM(items) AS {}".format(num_or_denom)
        from_ = "hscic.normalised_prescribing_standard"
        where = get_measure_value("where")

    elif type_ == "bnf_quantity":
        columns = "SUM(quantity) AS {}".format(num_or_denom)
        from_ = "hscic.normalised_prescribing_standard"
        where = get_measure_value("where")

    elif type_ == "bnf_cost":
        columns = "SUM(actual_cost) AS {}".format(num_or_denom)
        from_ = "hscic.normalised_prescribing_standard"
        where = get_measure_value("where")

    elif type_ == "list_size":
        assert num_or_denom == "denominator"
        columns = "SUM(total_list_size / 1000.0) AS denominator"
        from_ = "hscic.practice_statistics"
        where = "1 = 1"

    elif type_ == "star_pu_antibiotics":
        assert num_or_denom == "denominator"
        columns = "CAST(JSON_EXTRACT(MAX(star_pu), '$.oral_antibacterials_item') AS FLOAT64) AS denominator"
        from_ = "hscic.practice_statistics"
        where = "1 = 1"

    else:
        assert False, type_

    return {
        full_attr_name("columns"): columns,
        full_attr_name("from"): from_,
        full_attr_name("where"): where,
    }


def build_sql(run_name):
    """Takes all JSON files in the run_name folder and uses
    build_num_or_denom_fields to add fields to the SQL template.
    Writes folder with SQL corresponding to the measure_sql folder.
    
    Args:
        run_name (str): Name of folder where JSON is found
    """
    output_dir = f"data/measure_sql/{run_name}"

    os.makedirs(output_dir, exist_ok=True)

    measure_def_paths = [
        entry.path
        for entry in os.scandir(f"data/measure_json/{run_name}")
        if entry.name.endswith(".json")
    ]

    with open("data/template.sql") as f:
        template = f.read()

    for path in measure_def_paths:
        measure_id = os.path.basename(path).split(".")[0]
        print(measure_id)

        measure_def = modify_measure_json(path)

        context = {}
        context.update(build_num_or_denom_fields(measure_def, "numerator"))
        context.update(build_num_or_denom_fields(measure_def, "denominator"))

        with open(os.path.join(output_dir, measure_id + ".sql"), "w") as f:
            f.write(template.format(**context))
