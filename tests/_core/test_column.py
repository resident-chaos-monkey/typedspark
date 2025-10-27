from dataclasses import dataclass
from typing import Annotated

import pandas as pd
import pytest
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from pyspark.sql.types import IntegerType, LongType, StringType

from typedspark import Column, ColumnMeta, Schema, StructType
from typedspark._utils.create_dataset import create_partially_filled_dataset


class A(Schema):
    a: Column[LongType]
    b: Column[StringType]


def test_column(spark: SparkSession):
    (
        spark.createDataFrame(
            pd.DataFrame(
                dict(
                    a=[1, 2, 3],
                )
            )
        )
        .filter(A.a == 1)
        .withColumn(A.b.str, lit("a"))
    )


def test_column_doesnt_exist():
    with pytest.raises(TypeError):
        A.z


@pytest.mark.no_spark_session
def test_column_reference_without_spark_session():
    a = A.a
    assert a.str == "a"


def test_column_with_deprecated_dataframe_param(spark: SparkSession):
    df = create_partially_filled_dataset(spark, A, {A.a: [1, 2, 3]})
    Column("a", dataframe=df)


@dataclass
class MyColumnMeta(ColumnMeta):
    primary_key: bool = False


class Persons(Schema):
    id: Annotated[
        Column[LongType],
        MyColumnMeta(
            comment="Identifies the person",
            primary_key=True,
        ),
    ]
    name: Column[StringType]
    age: Column[LongType]


def test_get_metadata():
    assert Persons.get_metadata()["id"] == {
        "comment": "Identifies the person",
        "primary_key": True,
    }


@pytest.mark.no_spark_session
def test_column_repr_no_spark_session():
    spark = SparkSession.getActiveSession()
    if spark is None:
        assert repr(A.a) == "Column<'a'> (no active Spark session)"
    else:
        assert repr(A.a) == "Column<'a'>"


class Cause(Schema):
    source: Column[StringType]


class Values(Schema):
    name: Column[StringType]
    severity: Column[IntegerType]
    cause: Column[StructType[Cause]]


class Actions(Schema):
    consequences: Column[StructType[Values]]


def test_full_path_1():
    assert Actions.consequences.full_path == "consequences"


def test_full_path_2():
    assert Actions.consequences.dtype.schema.severity.full_path == "consequences.severity"


def test_full_path_3():
    assert (
        Actions.consequences.dtype.schema.cause.dtype.schema.source.full_path
        == "consequences.cause.source"
    )


def test_direct_nested_field_access_level_1():
    """Test direct access to first level nested fields without .dtype.schema"""
    # New syntax should work the same as old syntax
    assert Actions.consequences.severity.full_path == "consequences.severity"
    assert Actions.consequences.severity.full_path == Actions.consequences.dtype.schema.severity.full_path


def test_direct_nested_field_access_level_2():
    """Test direct access to second level nested fields without .dtype.schema"""
    # New syntax should work the same as old syntax for deeply nested fields
    assert Actions.consequences.cause.source.full_path == "consequences.cause.source"
    assert (
        Actions.consequences.cause.source.full_path
        == Actions.consequences.dtype.schema.cause.dtype.schema.source.full_path
    )


def test_both_syntaxes_equivalent():
    """Demonstrate that both old and new syntaxes produce equivalent results"""
    # Level 1 access
    old_l1 = Actions.consequences.dtype.schema.severity
    new_l1 = Actions.consequences.severity
    assert old_l1.full_path == new_l1.full_path == "consequences.severity"

    # Level 2 access
    old_l2 = Actions.consequences.dtype.schema.cause.dtype.schema.source
    new_l2 = Actions.consequences.cause.source
    assert old_l2.full_path == new_l2.full_path == "consequences.cause.source"

    # Mixed access (old + new)
    mixed = Actions.consequences.dtype.schema.cause.source  # old.new
    assert mixed.full_path == "consequences.cause.source"


def test_new_syntax_examples():
    """Show examples of the improved syntax"""
    # Before: verbose nested access
    old_severity = Actions.consequences.dtype.schema.severity
    old_cause_source = Actions.consequences.dtype.schema.cause.dtype.schema.source

    # After: clean direct access
    new_severity = Actions.consequences.severity
    new_cause_source = Actions.consequences.cause.source

    # Both produce the same results
    assert old_severity.full_path == new_severity.full_path
    assert old_cause_source.full_path == new_cause_source.full_path

    # New syntax is much more readable
    assert new_severity.full_path == "consequences.severity"
    assert new_cause_source.full_path == "consequences.cause.source"


def test_direct_nested_field_access_invalid_field():
    """Test that accessing non-existent nested fields raises AttributeError"""

    with pytest.raises(AttributeError, match="object has no attribute 'nonexistent'"):
        Actions.consequences.nonexistent


def test_direct_nested_field_access_on_non_struct():
    """Test that accessing fields on non-struct columns raises AttributeError"""

    # A.a is not a struct type, so accessing .some_field should fail
    with pytest.raises(AttributeError, match="object has no attribute 'some_field'"):
        A.a.some_field


def test_direct_nested_field_access_name_conflicts():
    """Test that field names conflicting with Column methods don't interfere"""
    # Test that a field named 'name' would use Column.name method, not nested field access
    # This is expected behavior to maintain Column method precedence

    class FieldWithConflictingName(Schema):
        name: Column[StringType]  # This conflicts with Column.name method

    class TestSchema(Schema):
        field: Column[StructType[FieldWithConflictingName]]

    # The 'name' attribute should return the Column method, not nested field access
    col = TestSchema.field
    # Column.name should be a method, not a Column instance
    assert callable(col.name), "Column.name should be the PySpark method, not nested field access"

    # But non-conflicting fields should work fine
    class NonConflictingField(Schema):
        custom_name: Column[StringType]  # This doesn't conflict

    class TestSchema2(Schema):
        field: Column[StructType[NonConflictingField]]

    # This should work as nested field access
    nested_field = TestSchema2.field.custom_name
    assert isinstance(nested_field, Column), "Non-conflicting field should return Column instance"
    assert nested_field.full_path == "field.custom_name"
