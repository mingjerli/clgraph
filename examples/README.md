# clpipe Examples

This directory contains comprehensive examples demonstrating various features of clpipe for SQL column lineage analysis.

## Basic Examples

### `simple_example.py`
Basic introduction to clpipe with CTEs and joins.

**Run:**
```bash
python examples/simple_example.py
```

**Features demonstrated:**
- Basic column lineage tracing
- CTEs (Common Table Expressions)
- JOIN operations
- Backward and forward lineage

---

### `pipeline_example.py`
Multi-query pipeline analysis demonstrating cross-query lineage tracking.

**Run:**
```bash
python examples/pipeline_example.py
```

**Features demonstrated:**
- Multi-query pipelines
- Table-level dependencies
- Cross-query column lineage
- Pipeline execution order

---

## Advanced Features

### `metadata_and_export_example.py`
Working with metadata, descriptions, and exporting lineage.

**Run:**
```bash
python examples/metadata_and_export_example.py
```

**Features demonstrated:**
- Column metadata (PII, owner, tags)
- Metadata propagation
- JSON/CSV/GraphViz export
- Custom metadata fields

---

### `metadata_comments_example.py`
Extracting and using metadata from SQL comments.

**Run:**
```bash
python examples/metadata_comments_example.py
```

**Features demonstrated:**
- SQL comment parsing
- Inline metadata extraction
- Column descriptions from comments

---

### `llm_description_generation.py`
Using LLMs to generate column descriptions.

**Run:**
```bash
python examples/llm_description_generation.py
```

**Features demonstrated:**
- LLM-powered description generation
- Automated documentation
- Description propagation

---

### `pipeline_execution_example.py`
Pipeline execution and orchestration.

**Run:**
```bash
python examples/pipeline_execution_example.py
```

**Features demonstrated:**
- Synchronous pipeline execution
- Asynchronous pipeline execution
- Airflow DAG generation
- Error handling and recovery

---

## Set Operations (NEW)

### `set_operations_example.py`
Comprehensive examples of UNION, INTERSECT, and EXCEPT operations.

**Run:**
```bash
python examples/set_operations_example.py
```

**Features demonstrated:**
- UNION ALL - combining datasets
- UNION DISTINCT - deduplication
- Three-way and multi-way UNIONs
- INTERSECT - finding common elements
- EXCEPT - finding differences
- Set operations with CTEs
- Set operations with subqueries

**Examples included:**
1. Basic UNION ALL - Active and archived users
2. UNION DISTINCT - User ID deduplication
3. Three-way UNION - Multiple data sources
4. INTERSECT - Common elements
5. EXCEPT - Set differences
6. UNION with CTEs - Complex aggregations
7. UNION with subqueries - Nested patterns

---

## PIVOT Operations (NEW)

### `pivot_example.py`
Comprehensive examples of PIVOT operations for transforming rows to columns.

**Run:**
```bash
python examples/pivot_example.py
```

**Features demonstrated:**
- Basic PIVOT - quarterly data transformation
- PIVOT from base tables
- Multiple aggregation functions
- PIVOT with CTEs
- PIVOT with filters and JOINs
- Real-world financial reporting

**Examples included:**
1. Basic PIVOT - Quarterly sales by product
2. PIVOT from base table - Regional revenue
3. Multiple aggregations - Sales and orders
4. PIVOT with CTE - User activity analysis
5. PIVOT with filters - Product performance
6. Real-world example - Financial reporting

**Use cases:**
- Creating cross-tabulations
- Building dashboards
- Transforming time-series data
- Financial reporting

---

## UNPIVOT Operations (NEW)

### `unpivot_example.py`
Comprehensive examples of UNPIVOT operations for normalizing data.

**Run:**
```bash
python examples/unpivot_example.py
```

**Features demonstrated:**
- Basic UNPIVOT - quarterly normalization
- Multiple measure columns
- NULL handling with INCLUDE NULLS
- UNPIVOT with CTEs
- Real-world survey data analysis
- PIVOT vs UNPIVOT comparison

**Examples included:**
1. Basic UNPIVOT - Quarterly revenue
2. Multiple measures - Sales and costs
3. NULL handling - Include/exclude NULLs
4. UNPIVOT with CTE - Normalized reporting
5. Real-world example - Survey responses
6. PIVOT vs UNPIVOT comparison

**Use cases:**
- Normalizing wide-format data
- Survey data analysis
- Time-series normalization
- Database schema migration

**Note:** UNPIVOT support in sqlglot may be limited for some dialects.

---

## Running Examples

All examples can be run directly with Python:

```bash
# Basic example
python examples/simple_example.py

# Set operations
python examples/set_operations_example.py

# PIVOT operations
python examples/pivot_example.py

# UNPIVOT operations
python examples/unpivot_example.py
```

Or using uv (if configured):

```bash
uv run python examples/simple_example.py
```

---

## Example Output

Each example includes:
- ✅ Clear SQL query examples
- ✅ Explanation of what the query does
- ✅ Visual before/after transformations (for PIVOT/UNPIVOT)
- ✅ Lineage analysis results
- ✅ Node and edge counts
- ✅ Output column listings
- ✅ Source table identification

---

## SQL Files

The `sql_files/` directory contains example SQL queries used in pipeline analysis:

```bash
examples/sql_files/
├── staging/      # Staging layer queries
├── analytics/    # Analytics layer queries
└── reporting/    # Reporting layer queries
```

---

## What's Next?

After exploring these examples, check out:

1. **Documentation**: Visit the full documentation for API reference
2. **Tests**: Look at `tests/` for more usage patterns
3. **Your own queries**: Try analyzing your own SQL queries!

---

## Support

For questions or issues:
- GitHub Issues: https://github.com/mingjerli/clpipe/issues
- Documentation: (link to docs site)

---

## Contributing

Have a good example to share? Contributions are welcome!
1. Fork the repository
2. Add your example to this directory
3. Update this README
4. Submit a pull request
