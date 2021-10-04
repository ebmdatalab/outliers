"""
Partly adapted from https://github.com/ebmdatalab/html-template-demo
"""

import markupsafe
import jinja2
from lxml import html


def make_tr(thlist):
    """
    Make a 'tr' lxml Element from list of 'th' lxml Elements

    Parameters
    ----------
    thlist : list
        List of strings containing desired inner text of th Elements

    Returns
    -------
    Element
        lxml html 'tr' Element with 'th' children in order of input list
    """
    tr = html.Element("tr")
    for th in thlist:
        e_th = html.Element("th")
        e_th.text = th
        tr.append(e_th)
    return tr


# hack: ideally header should be fixed in df, not html
def merge_table_header(table):
    """
    Merge duplicate header rows of html table into one

    Replaces blank inner text of <th> in <tr>s of <thead> with non-blank
    inner text from corresponding <th>s in subsequent <tr>s

    Parameters:
    -----------
    table : str
        html <table> element containing <thead> of at least one <tr>

    Returns:
    --------
    str
        utf-8 encoded string of html <table> element
    """
    tabroot = html.fragment_fromstring(table)
    merged_th = []
    for tr in tabroot.xpath("thead/tr"):
        for i, th in enumerate(tr):
            while len(merged_th) < i + 1:
                merged_th.append("")
            if not th.text_content() == "":
                merged_th[i] = th.text_content()
        tr.drop_tree()
    tabroot.xpath("thead")[0].append(make_tr(merged_th))
    return html.tostring(tabroot).decode("utf-8")


def df_to_html(df):
    """
    Return formatted html table from Pandas DataFrame

    Uses native DataFrame.to_html function with selected bootstrap table
    classes, merges duplicate header rows that this generates, then
    unescapes the results using markupsafe

    Parameters
    ----------
    df : DataFrame
        Pandas DataFrame to be converted to html

    Returns
    -------
    str
        html fragment containing <table> element
    """
    return merge_table_header(
        markupsafe.Markup(
            df.to_html(
                escape=True,
                classes=["table", "thead-light", "table-bordered", "table-sm"],
            )
        ).unescape()
    )


def selective_title(str):
    """
    Convert string to Title Case except for key initialisms

    Splits input string by space character, applies Title Case to each element
    except for ["NHS", "PCN", "CCG"], joins elements back together with space

    Parameters
    ----------
    str : str
        string to be selectively converted to Title Case

    Returns
    -------
    str
        Selectively title-cased string

    """
    ALLCAPS = ["NHS", "PCN", "CCG"]
    return " ".join([w.title() if w not in ALLCAPS else w for w in str.split(" ")])


def write_to_template(entity_name, table_high, table_low, output_file):
    """
    Populate jinja template with outlier report data

    Calls df_to_html to generated <table> fragments, correctly formats enetity name,
    passes these to jinja template and renders final html

    Parameters
    ----------
    entity_name : str
        Name of entity for which report is being run
    table_high : DataFrame
        Table of items which entity prescribes higher than average
    table_low : DataFrame
        Table of items which entity prescribes lower than average
    output_file : str
        file name (not full path) of html file to be written

    Returns
    -------
    str
        Complete HTML outlier report
    """
    with open("../data/template.html") as f:
        template = jinja2.Template(f.read())

    context = {
        "entity_name": selective_title(entity_name),
        "table_high": df_to_html(table_high),
        "table_low": df_to_html(table_low),
    }

    with open(f"../data/html/{output_file}.html", "w") as f:
        f.write(template.render(context))
