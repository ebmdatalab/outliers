"""
Partly adapted from https://github.com/ebmdatalab/html-template-demo
"""

import markupsafe
import jinja2
from lxml import html

definitions = {
    "Chemical Items": "number of prescribed items containing this chemical",
    "Subparagraph Items": "count of all prescribed items "
    "from this subparagraph",
    "Ratio": "Ratio of chemical items to subparagraph items",
    "Mean": "Population mean number of chemical items prescribed",
    "std": "Standard Deviation",
    "Z_Score": "Number of standard deviations prescribed"
    "item count is away from the mean",
}


def add_definitions(df):
    """
    Add html abbr/tooltip definition for column header items

    Parameters
    ----------
    df : DataFrame
        data frame to perform column replacement on
    Returns
    -------
    DataFrame
        data frame with column definitons added
    """
    return df.rename(
        columns=lambda x: make_abbr(x, definitions[x])
        if x in definitions
        else x
    )


def format_url(df):
    """
    Replace index column values with html anchor pointing at URL in URL column
    then drop URL column

    Parameters
    ----------
    df : DataFrame
        Data frame on which to perform replacment
    Returns
    -------
    df : DataFrame
        Data Frame with index column values turned into URLs
    """

    ix_col = df.index.name
    df = df.reset_index()
    df[ix_col] = df.apply(
        lambda x: f'<a href="{x["URL"]}">{x[ix_col]}</a>', axis=1
    )
    df = df.drop("URL", axis=1)

    df = df.set_index(ix_col)

    return df


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
        tr.append(th)
    return tr


def make_abbr(text, title):
    """
    Make a 'abbr' html element from body text and its definition (title)

    Parameters
    ----------
    text : str
        Text to be definied
    title : str
        Definition for text
    Returns
    -------
    str

    """
    return f'<abbr title="{title}">{text}</abbr>'


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
                merged_th[i] = th
        tr.drop_tree()
    tabroot.xpath("thead")[0].append(make_tr(merged_th))
    return html.tostring(tabroot).decode("utf-8")


def df_to_html(df):
    """
    Return formatted html table from Pandas DataFrame

    Pre-formats DataFrame with URL formatting for first column,
    title-casing column headers.
    Uses native DataFrame.to_html function with selected bootstrap table
    classes, merges duplicate header rows that this generates, then
    unescapes the results using markupsafe

    Parameters
    ----------
    df : DataFrame
        Pandas DataFrame to be converted to html

    Returns
    -------
    table : str
        html fragment containing <table> element
    """
    df = format_url(df)
    df = df.rename(columns=lambda x: selective_title(x))
    df = add_definitions(df)
    table = df.to_html(
        escape=True,
        classes=["table", "thead-light", "table-bordered", "table-sm"],
    )
    table = markupsafe.Markup(table).unescape()
    table = merge_table_header(table)

    return table


def selective_title(str):
    """
    Convert string to Title Case except for key initialisms

    Splits input string by space character, applies Title Case to each element
    except for ["NHS", "PCN", "CCG", "BNF", "std"], then
    joins elements back together with space

    Parameters
    ----------
    str : str
        string to be selectively converted to Title Case

    Returns
    -------
    str
        Selectively title-cased string

    """
    ALLCAPS = ["NHS", "PCN", "CCG", "BNF", "std"]
    return " ".join(
        [w.title() if w not in ALLCAPS else w for w in str.split(" ")]
    )


def write_to_template(
    entity_name,
    table_high,
    table_low,
    output_path,
    template_path,
    date_from,
    date_to
):
    """
    Populate jinja template with outlier report data

    Calls df_to_html to generated <table> fragments,
    correctly formats entity name,
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

    with open(template_path) as f:
        template = jinja2.Template(f.read())

    context = {
        "entity_name": selective_title(entity_name),
        "table_high": df_to_html(table_high),
        "table_low": df_to_html(table_low),
        "date_from": date_from,
        "date_to": date_to
    }

    with open(output_path, "w") as f:
        f.write(template.render(context))
