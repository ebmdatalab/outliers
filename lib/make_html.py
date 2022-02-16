"""
Partly adapted from https://github.com/ebmdatalab/html-template-demo
"""

from datetime import date
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
    "Z_Score": "Number of standard deviations prescribed "
    "item count is away from the mean",
}

REPORT_DATE_FORMAT = "%B %Y"


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


def add_item_rows(table, items_df):
    """
    Adds hidden rows containing item prescription counts

    Iterates rows of html table body
    extracts BNF chemical id from each row
    filters items_df by chemical id
    creates hidden row from filtered dataframe
    rebuilds table body with visible (original) and hidden rows

    Parameters
    ----------
    table : str
        html table built from primary outlier dataframe
    items_df : DataFrame
    Returns
    -------
    table_root : str
        utf-8 encoded string of html table root element
    """

    def make_hidden_row(df, id, analyse_url):
        """
        Builds tr of precription items hidden by bootstrap collapse class

        Creates tr html element containing full with td and div within
        For each row of input df, generate BNF-item-specific analyse URL
        from input analyse_url show BNF item name and prescription count
        and add to div

        Parameters
        ----------
        df : DataFrame
            chemical, bnf_code, bnf_name, and numerator of items prescribed
        id : str
            Unique css id for tr to be built
        analyse_url : str
            URL to openprescribing anaylse page for current entity and chemical
        Returns
        -------
        tr : lxml Element
            html tr element
        """
        tr = html.Element("tr")
        tr.set("id", f"{id}_items")
        tr.set("class", "collapse")
        td = html.Element("td")
        td.set("colspan", "9")
        td.set("class", "hiddenRow")
        ul = html.Element("ul")
        ul.set("class", "my-0 ps-4 py-2")

        for i, r in df.reset_index().iterrows():
            url = analyse_url.replace(r["chemical"], r["bnf_code"])
            name = r["bnf_name"]
            count = r["numerator"]
            list_item = html.Element("li")
            anchor = html.Element("a")
            anchor.set("href", url)
            anchor.set("target", "_blank")
            anchor.text = f"{name} : {count}"
            list_item.append(anchor)
            ul.append(list_item)

        td.append(ul)
        tr.append(td)
        return tr

    def make_open_button(id):
        """
        Create open/expand prescription item detail button

        Parameters
        ----------
        id : str
            Unique css id for target tr to be expanded
        Returns
        -------
        bt_open : lxml Element
            html button element
        """
        bt_open = html.Element("button")
        bt_open.set("class", "btn btn-outline-primary btn-sm btn-xs ms-2 px-2")
        bt_open.set("data-bs-target", f"#{id}_items")
        bt_open.set("data-bs-toggle", "collapse")
        bt_open.set("type", "button")
        bt_open.text = "☰"
        return bt_open

    table_root = html.fragment_fromstring(table)
    table_id = table_root.get("id")

    hidden_rows = []
    visible_rows = table_root.xpath("tbody/tr")
    for i, tr in enumerate(visible_rows):
        # create a unique id for this row
        id = f"{table_id}_{i}"

        # hack:extract the id of the BNF chemical from the analyse URL
        analyse_url = tr.xpath("th/a")[0].get("href")
        chemical_id = analyse_url.split("/")[-1].split("&")[2].split("=")[1]

        # add an open button to the end of the first column
        tr.xpath("th")[0].append(make_open_button(id))

        hidden_rows.append(
            make_hidden_row(
                items_df[items_df.chemical == chemical_id], id, analyse_url
            )
        )
    tbody = table_root.xpath("tbody")[0]
    tbody.drop_tree()
    tbody = html.Element("tbody")
    for hr, vr in zip(hidden_rows, visible_rows):
        tbody.append(vr)
        tbody.append(hr)
    table_root.append(tbody)
    return html.tostring(table_root).decode("utf-8")


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
    return f'<abbr data-bs-toggle="tooltip" data-bs-placement="top" title="{title}">{text}</abbr>'



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


def df_to_html(dfs, id):
    """
    Return formatted html table from Pandas DataFrame

    Pre-formats DataFrame with URL formatting for first column,
    title-casing column headers.
    Uses native DataFrame.to_html function with selected bootstrap table
    classes, merges duplicate header rows that this generates, then
    unescapes the results using markupsafe

    Parameters
    ----------
    dfs : tuple(DataFrame)
        primary and item detail DataFrames to be converted to html
    id : css id of generated html table

    Returns
    -------
    table : str
        html fragment containing <table> element
    """
    df = dfs[0]
    items_df = dfs[1]
    df = format_url(df)
    df = df.rename(columns=lambda x: selective_title(x))
    df = add_definitions(df)
    table = df.to_html(
        escape=True,
        classes=["table", "table", "table-sm", "table-bordered"],
        table_id=id,
    )
    table = markupsafe.Markup(table).unescape()
    table = add_item_rows(table, items_df)
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
    ALLCAPS = ["NHS", "PCN", "CCG", "BNF", "std", "STP", "(STP)", "NHS"]
    return " ".join(
        [w.title() if w not in ALLCAPS else w for w in str.split(" ")]
    )


def write_to_template(
    entity_name,
    tables_high,
    tables_low,
    output_path,
    template_path,
    from_date: date,
    to_date: date,
    entity_type: str,
    entity_code: str,
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
    tables_high : tuple(DataFrame)
        Table of items which entity prescribes higher than average
    tables_low : tuple(DataFrame)
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
        "table_high": df_to_html(tables_high, "table_high"),
        "table_low": df_to_html(tables_low, "table_low"),
        "from_date": from_date.strftime(REPORT_DATE_FORMAT),
        "to_date": to_date.strftime(REPORT_DATE_FORMAT),
        "entity_type": entity_type,
        "entity_code": entity_code,
    }

    with open(output_path, "w") as f:
        f.write(template.render(context))
