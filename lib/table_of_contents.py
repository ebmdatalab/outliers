import os.path


class MarkdownToC:
    """
    Builds Markdown table of contents with optional custom heading and URLs
    """
    md = ""
    files = {}
    urlprefix = ""

    def __init__(self, urlprefix=None, heading=None):
        """
        MarkdownToC constructor

        Parameters
        ----------
        urlprefix : str, optional
            prefix to be appended to the URL of each item
            added to the table of contents
        heading : str, optional
            override standard "Table of Contents" header
        """
        self.md = "# Table of contents" if not heading else heading
        self.urlprefix = urlprefix or ""

    def add_file(self, path, entity=""):
        """
        Adds a file to the table of contents

        Parameters
        ----------
        path : str
            relative path to the item
        entity : str, optional
            entity type of the item
        """
        if entity not in self.files.keys():
            self.files[entity] = [path]
        else:
            self.files[entity].append(path)

    def write_toc(self, output_path):
        """
        Write table of contents as Markdown to README.md in output path

        Parameters
        ----------
        output_path : str
            directory to write markdown, items are assumed to be in same
            or sub-directory for relative path derivation
        """
        tocfile = output_path + "/README.md"
        with open(tocfile, "w") as f:
            f.write(self.__print_toc(output_path))

    def __print_toc(self, output_path):
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
        toc = self.md
        for entity in self.files.keys():
            toc = toc + "\n" + f"* {entity}"
            for file in self.files[entity]:
                common_path = os.path.commonpath([output_path, file])
                path_parts = self.__split_all(file)
                for cp in self.__split_all(common_path):
                    path_parts.remove(cp)
                relative_path = os.path.join(*path_parts)

                toc = toc + "\n" + f"  * [{relative_path}]"
                fullpath = os.path.join(
                    os.path.basename(output_path),
                    relative_path)
                toc = toc + f"({self.urlprefix}{fullpath})"

        return toc

    @staticmethod
    def __split_all(path):
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
