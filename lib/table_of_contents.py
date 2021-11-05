import os.path
import jinja2
from lib.make_html import selective_title


class TableOfContents:
    """
    Builds Markdown or html table of contents

    Attributes
    ----------
    hierarchy : Dict[]
        hierarchical dictionary of entity codes
    urlprefix : str, optional
        prefix to be appended to the URL of each item
        added to the table of contents
    heading : str, optional
        override standard "Table of Contents" header
    html_template : str, optional
        path to jinja2 template file for html output
    """

    def __init__(
        self,
        url_prefix,
        heading="Table of contents",
        html_template="../data/toc_template.html",
    ):
        self.items = {}
        self.hierarchy = {}
        self.url_prefix = url_prefix
        self.heading = heading
        self.html_template = html_template

    def add_item(self, code, name, file_path, entity=""):
        """
        Adds a file to the table of contents

        Parameters
        ----------
        path : str
            relative path to the item
        entity : str, optional
            entity type of the item
        """
        if entity not in self.items.keys():
            self.items[entity] = {code: {"name": name, "file_path": file_path}}
        else:
            self.items[entity][code] = {"name": name, "file_path": file_path}

    def _get_context(self, output_path):
        ctx = {"header": self.heading}
        ctx["stps"] = []
        for stp_code, ccgs in self.hierarchy.items():
            stp_item = self._get_item_context(stp_code, "stp", output_path)
            stp_item["ccgs"] = []
            for ccg_code, pcns in ccgs.items():
                ccg_item = self._get_item_context(ccg_code, "ccg", output_path)
                ccg_item["pcns"] = []
                for pcn_code, practices in pcns.items():
                    pcn_item = self._get_item_context(
                        pcn_code, "pcn", output_path
                    )
                    pcn_item["practices"] = []
                    for practice_code in practices:
                        pcn_item["practices"].append(
                            self._get_item_context(
                                practice_code, "practice", output_path
                            )
                        )
                    pcn_item["practices"].sort(key=lambda x: x["name"])
                    ccg_item["pcns"].append(pcn_item)
                ccg_item["pcns"].sort(key=lambda x: x["name"])
                stp_item["ccgs"].append(ccg_item)
            stp_item["ccgs"].sort(key=lambda x: x["name"])
            ctx["stps"].append(stp_item)
        ctx["stps"].sort(key=lambda x: x["name"])
        return ctx

    def _get_item_context(self, entity_code, entity_type, output_path):
        entity_item = self.items[entity_type][entity_code]
        return {
            "code": entity_code,
            "name": selective_title(entity_item["name"]),
            "href": self.url_prefix
            + self._full_path(
                output_path,
                self._relative_path(output_path, entity_item["file_path"]),
            ),
        }

    def write_html(self, output_path):
        """
        Write table of contents as html to index.html in output path

        Parameters
        ----------
        output_path : str
            directory to write markdown, items are assumed to be in same
            or sub-directory for relative path derivation
        """
        assert self.items, "no items to write"
        with open(self.html_template) as f:
            template = jinja2.Template(f.read())

        context = self._get_context(output_path)

        with open(os.path.join(output_path, "index.html"), "w") as f:
            f.write(template.render(context))

    def write_markdown(self, output_path, link_to_html_toc=False):
        """
        Write table of contents as Markdown to README.md in output path

        Parameters
        ----------
        output_path : str
            directory to write markdown, items are assumed to be in same
            or sub-directory for relative path derivation
        """
        assert self.items, "no items to write"
        tocfile = output_path + "/README.md"
        with open(tocfile, "w") as f:
            if link_to_html_toc:
                f.write(self._print_markdown_link_html())
            else:
                f.write(self._print_markdown(output_path))

    def _print_markdown_link_html(self):
        html_toc_url = f"{self.url_prefix}{'data/index.html'}"
        return f"# [{self.heading}]({html_toc_url})"

    def _print_markdown(self, output_path):
        """
        Renders table of contents as Markdown

        Derives path of each item relative to the output path by finding
        common parent directories and removing. If your item paths and
        output paths do not have the same base then this may
        produce unexpected behaviour.

        Parameters
        ----------
        output_path : str
            directory where markdown file will eventually be written
            items are assumed to be in same,
            or sub-directory for relative path derivation
        Returns
        -------
        toc : str
            table of contents in Markdown format
        """
        toc = f"# {self.heading}"
        for entity in self.items.keys():
            toc = toc + "\n" + f"* {entity}"
            for _, v in self.items[entity].items():
                file = v["file_path"]
                relative_path = self._relative_path(output_path, file)
                fullpath = self._full_path(relative_path)
                toc = toc + (
                    "\n" + f"  * [{relative_path}]"
                    f"({self.url_prefix}{fullpath})"
                )
        return toc

    @staticmethod
    def _relative_path(output_path, file_path) -> str:
        common_path = os.path.commonpath([output_path, file_path])
        path_parts = TableOfContents._split_all(file_path)
        for cp in TableOfContents._split_all(common_path):
            path_parts.remove(cp)
        return os.path.join(*path_parts)

    @staticmethod
    def _full_path(output_path, relative_path) -> str:
        return os.path.join(os.path.basename(output_path), relative_path)

    @staticmethod
    def _split_all(path):
        """
        splits a path into a list of all its elements using the local separator

        adapted from
        https://www.oreilly.com/library/view/python-cookbook/0596001673/ch04s16.html

        Parameters
        ----------
        path : str
            path to split, OS agnostic
        Returns
        -------
        allparts : list
            list of all parts of the path as strings
        """
        allparts = []
        while 1:
            parts = os.path.split(path)
            if parts[0] == path:  # sentinel for absolute paths
                allparts.insert(0, parts[0])
                break
            elif parts[1] == path:  # sentinel for relative paths
                allparts.insert(0, parts[1])
                break
            else:
                path = parts[0]
                allparts.insert(0, parts[1])
        return allparts
