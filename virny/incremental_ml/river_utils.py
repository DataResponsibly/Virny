import random
import typing
import datetime as dt

from io import StringIO
from river import base
from river.stream.iter_csv import DictReader


def ddict2dict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = ddict2dict(v)
    return dict(d)


def df_to_stream_buffer(df):
    buffer = StringIO()  # creating an empty buffer
    df.to_csv(buffer, index=False)  # filling that buffer
    buffer.seek(0) # set to the start of the stream

    return buffer


def iter_pd_dataset(
        pd_dataset,
        target: typing.Union[str, typing.List[str]] = None,
        converters: dict = None,
        parse_dates: dict = None,
        drop: typing.List[str] = None,
        drop_nones=False,
        fraction=1.0,
        seed: int = None,
        **kwargs,
) -> base.typing.Stream:

    buffer = df_to_stream_buffer(pd_dataset)
    for x in DictReader(fraction=fraction, rng=random.Random(seed), f=buffer, **kwargs):
        if drop:
            for i in drop:
                del x[i]

        # Cast the values to the given types
        if converters is not None:
            for i, t in converters.items():
                if str(t) == "<class 'int'>":
                    # Fix an issue with converting '1.0' to an int type
                    x[i] = int(float(x[i]))
                else:
                    x[i] = t(x[i])

        # Drop Nones
        if drop_nones:
            for i in list(x):
                if x[i] is None:
                    del x[i]

        # Parse the dates
        if parse_dates is not None:
            for i, fmt in parse_dates.items():
                x[i] = dt.datetime.strptime(x[i], fmt)

        # Separate the target from the features
        y = None
        if isinstance(target, list):
            y = {name: x.pop(name) for name in target}
        elif target is not None:
            y = x.pop(target)

        yield x, y
