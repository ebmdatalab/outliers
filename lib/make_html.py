"""
Party adapted from https://github.com/ebmdatalab/html-template-demo
"""

import markupsafe
import jinja2


def df_to_html(df):
    return markupsafe.Markup(df.to_html(escape=True, classes=["table","thead-light","table-bordered","table-sm"])).unescape()


def write_to_template(entity_name, table_high, table_low, output_file):
    with open("../data/template.html") as f:
        template = jinja2.Template(f.read())

    context = {
        "entity_name": entity_name,
        "table_high": df_to_html(table_high),
        "table_low": df_to_html(table_low),
    }

    with open(f"../data/html/{output_file}.html", "w") as f:
        f.write(template.render(context))
