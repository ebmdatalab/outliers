"""
Party adapted from https://github.com/ebmdatalab/html-template-demo
"""

import markupsafe
import jinja2
from lxml import html

# hack: ideally header should be fixed in df, not html
def make_tr(thlist):
    tr = html.Element("tr")
    for th in thlist:
        e_th = html.Element("th")
        e_th.text = th
        tr.append(e_th)
    return tr


def merge_table_header(table):
    tabroot = html.fragment_fromstring(table)
    merged_th = []
    for tr in tabroot.xpath('thead/tr'):
        for i, th in enumerate(tr):
            while len(merged_th) < i + 1:
                merged_th.append("")
            if not th.text_content() == "":
                merged_th[i] = th.text_content()
        tr.drop_tree()
    tabroot.xpath('thead')[0].append(make_tr(merged_th))
    return html.tostring(tabroot).decode('utf-8')


def df_to_html(df):
    return merge_table_header(
        markupsafe.Markup(
            df.to_html(
                escape=True,
                classes=["table", "thead-light", "table-bordered", "table-sm"],
            )
        ).unescape()
    )


def selective_title(str):
    ALLCAPS = ["NHS", "PCN", "CCG"]
    return " ".join([w.title() if w not in ALLCAPS else w for w in str.split(" ")])


def write_to_template(entity_name, table_high, table_low, output_file):
    with open("../data/template.html") as f:
        template = jinja2.Template(f.read())

    context = {
        "entity_name": selective_title(entity_name),
        "table_high": df_to_html(table_high),
        "table_low": df_to_html(table_low),
    }

    with open(f"../data/html/{output_file}.html", "w") as f:
        f.write(template.render(context))
