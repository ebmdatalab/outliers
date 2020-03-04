"""
Party adapted from https://github.com/ebmdatalab/html-template-demo
"""
#from base64 import b64encode
#from io import BytesIO

import markupsafe
import jinja2
#import matplotlib.pyplot as plt
#import numpy as np
#import pandas as pd


def df_to_html(df):
    return markupsafe.Markup(df.to_html(escape=True)).unescape()

def write_to_template(table_high, table_low, output_file):
	with open("../data/template.html") as f:
	    template = jinja2.Template(f.read())

	context = {
	    #"title": "Template demo",
	    "table_high": df_to_html(table_high),
	    "table_low": df_to_html(table_low),
	}

	with open(f"../data/html/{output_file}.html", "w") as f:
	    f.write(template.render(context))