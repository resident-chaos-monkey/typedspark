"""Module containing classes and functions related to TypedSpark Columns."""

from logging import warn
from typing import Generic, Optional, TypeVar, Union, get_args, get_origin, get_type_hints

from pyspark.sql import Column as SparkColumn
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import DataType

from typedspark._core.datatypes import StructType

T = TypeVar("T", bound=DataType)


class EmptyColumn(SparkColumn):
    """Column object to be instantiated when there is no active Spark session."""

    def __init__(self, *args, **kwargs) -> None:  # pragma: no cover
        pass


class Column(SparkColumn, Generic[T]):
    """Represents a ``Column`` in a ``Schema``. Can be used as:

    .. code-block:: python

        class A(Schema):
            a: Column[IntegerType]
            b: Column[StringType]
    """

    def __new__(
        cls,
        name: str,
        dataframe: Optional[DataFrame] = None,
        curid: Optional[int] = None,
        dtype: Optional[T] = None,
        parent: Union[DataFrame, "Column", None] = None,
        alias: Optional[str] = None,
    ):
        """``__new__()`` instantiates the object (prior to ``__init__()``).

        Here, we simply take the provided ``name``, create a pyspark
        ``Column`` object and cast it to a typedspark ``Column`` object.
        This allows us to bypass the pypsark ``Column`` constuctor in
        ``__init__()``, which requires parameters that may be difficult
        to access.
        """
        # pylint: disable=unused-argument

        if dataframe is not None and parent is None:
            parent = dataframe
            warn("The use of Column(dataframe=...) is deprecated, use Column(parent=...) instead.")

        column: SparkColumn
        if SparkSession.getActiveSession() is None:
            column = EmptyColumn()  # pragma: no cover
        elif alias is not None:
            column = col(f"{alias}.{name}")
        elif parent is not None:
            # If parent is a Column (struct field access), use col() to avoid parent[name]
            # If parent is a DataFrame, use parent[name] as usual
            if isinstance(parent, Column):
                # For struct field access, we need to construct the full path
                if hasattr(parent, 'full_path'):
                    column = col(f"{parent.full_path}.{name}")
                else:
                    column = col(f"{parent.str}.{name}")
            else:
                column = parent[name]
        else:
            column = col(name)

        column.__class__ = Column  # type: ignore
        return column

    def __init__(
        self,
        name: str,
        dataframe: Optional[DataFrame] = None,
        curid: Optional[int] = None,
        dtype: Optional[T] = None,
        parent: Union[DataFrame, "Column", None] = None,
        alias: Optional[str] = None,
    ):
        # pylint: disable=unused-argument
        self.str = name
        self._dtype = dtype if dtype is not None else DataType
        self._curid = curid
        self._parent = parent

    def __hash__(self) -> int:
        return hash((self.str, self._curid))

    @property
    def full_path(self) -> str:
        """Full path of the column including parent structure.
        Example:
        .. code-block:: python
            from pyspark.sql.types import IntegerType, StringType
            from typedspark import DataSet, StructType, Schema, Column

            class Values(Schema):
                name: Column[StringType]
                severity: Column[IntegerType]


            class Actions(Schema):
                consequences: Column[StructType[Values]]

        Both `Actions.consequences.dtype.schema.severity.full_path` and
        `Actions.consequences.severity.full_path` will yield the name
        of the field `severity` including the full path: `consequences.severity`

        """
        if isinstance(self._parent, Column):
            return f"{self._parent.full_path}.{self.str}"
        return self.str

    @property
    def dtype(self) -> T:
        """Get the datatype of the column, e.g. Column[IntegerType] -> IntegerType."""
        dtype = self._dtype

        if get_origin(dtype) == StructType:
            return StructType(
                schema=get_args(dtype)[0],
                parent=self,
            )  # type: ignore

        return dtype()  # type: ignore

    def __getattr__(self, name: str) -> "Column":
        """Allow direct access to nested schema fields without .dtype.schema.

        This enables accessing nested fields directly, e.g.:
            Actions.consequences.severity
        instead of:
            Actions.consequences.dtype.schema.severity
        """
        # Only handle attribute access for StructType columns
        # and only if the attribute doesn't already exist on Column/SparkColumn
        if (
            hasattr(self, "_dtype")
            and get_origin(self._dtype) == StructType
            and not hasattr(SparkColumn, name)
        ):
            # Get the schema class from the StructType generic
            schema_class = get_args(self._dtype)[0]

            # Check if the schema has this attribute using get_type_hints
            try:
                type_hints = get_type_hints(schema_class)
            except (NameError, AttributeError):
                # Forward references or missing annotations - can't introspect
                pass
            else:
                # Only proceed if name exists in type hints
                if name in type_hints:
                    try:
                        nested_dtype = schema_class._get_dtype(name)
                    except AttributeError:
                        # Schema doesn't implement _get_dtype - skip nested field creation
                        pass
                    else:
                        # Successfully got dtype, create the nested column
                        return Column(
                            name,
                            dtype=nested_dtype,
                            parent=self,
                            curid=getattr(self, "_curid", None),
                            alias=getattr(self, "_alias", None),
                        )

        # If not a nested field access, raise AttributeError as usual
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def __repr__(self) -> str:
        spark = SparkSession.getActiveSession()
        if spark is None:  # pragma: no cover
            return f"Column<'{self.str}'> (no active Spark session)"

        return super().__repr__()
