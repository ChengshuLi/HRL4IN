import logging


class GibsonLogger(logging.Logger):
    def __init__(
            self,
            name,
            level,
            filename=None,
            filemode="a",
            stream=None,
            format=None,
            dateformat=None,
            style="%",
    ):
        super().__init__(name, level)
        if filename is not None:
            handler = logging.FileHandler(filename, filemode)
        else:
            handler = logging.StreamHandler(stream)
        self._formatter = logging.Formatter(format, dateformat, style)
        handler.setFormatter(self._formatter)
        super().addHandler(handler)

    def add_filehandler(self, log_filename):
        filehandler = logging.FileHandler(log_filename)
        filehandler.setFormatter(self._formatter)
        self.addHandler(filehandler)


logger = GibsonLogger(name="gibson", level=logging.INFO, format="%(asctime)-15s %(message)s")
