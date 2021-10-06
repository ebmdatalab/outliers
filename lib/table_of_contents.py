import os.path


class MarkdownToC:
    md = ""
    files = {}
    urlprefix = ""

    def __init__(self, urlprefix):
        self.md = "# Table of contents"
        self.urlprefix = urlprefix

    def add_file(self, path, entity=""):
        if entity not in self.files.keys():
            self.files[entity] = [path]
        else:
            self.files[entity].append(path)

    def write_toc(self, output_path):
        tocfile = output_path + "/README.md"
        with open(tocfile, "w") as f:
            f.write(self.__print_toc(output_path))

    def __print_toc(self, output_path):
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
        """splits a path into a list of all its elements

        adapted from
        https://www.oreilly.com/library/view/python-cookbook/0596001673/ch04s16.html
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
