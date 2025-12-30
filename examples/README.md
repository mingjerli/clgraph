# clgraph Examples

This directory contains comprehensive Jupyter notebook examples demonstrating various features of clgraph for SQL column lineage analysis.

## Basic Examples

### `simple_example.ipynb`
Basic introduction to clgraph with CTEs and joins.

**Features demonstrated:**
- Basic column lineage tracing
- CTEs (Common Table Expressions)
- JOIN operations
- Backward and forward lineage

---

### `pipeline_example.ipynb`
Multi-query pipeline analysis demonstrating cross-query lineage tracking.

**Features demonstrated:**
- Multi-query pipelines
- Table-level dependencies
- Cross-query column lineage
- Pipeline execution order

---

## Advanced Features

### `metadata_and_export_example.ipynb`
Working with metadata, descriptions, and exporting lineage.

**Features demonstrated:**
- Column metadata (PII, owner, tags)
- Metadata propagation
- JSON/CSV/GraphViz export
- Custom metadata fields

---

### `metadata_comments_example.ipynb`
Extracting and using metadata from SQL comments.

**Features demonstrated:**
- SQL comment parsing
- Inline metadata extraction
- Column descriptions from comments

---

### `llm_description_generation.ipynb`
Using LLMs to generate column descriptions.

**Features demonstrated:**
- LLM-powered description generation
- Automated documentation
- Description propagation

**Note:** Requires Ollama or LLM API access.

---

### `pipeline_execution_example.ipynb`
Pipeline execution and orchestration.

**Features demonstrated:**
- Synchronous pipeline execution
- Asynchronous pipeline execution
- Airflow DAG generation
- Error handling and recovery

---

## Set Operations

### `set_operations_example.ipynb`
Comprehensive examples of UNION, INTERSECT, and EXCEPT operations.

**Features demonstrated:**
- UNION ALL - combining datasets
- UNION DISTINCT - deduplication
- Three-way and multi-way UNIONs
- INTERSECT - finding common elements
- EXCEPT - finding differences
- Set operations with CTEs
- Set operations with subqueries

---

## PIVOT Operations

### `pivot_example.ipynb`
Comprehensive examples of PIVOT operations for transforming rows to columns.

**Features demonstrated:**
- Basic PIVOT - quarterly data transformation
- PIVOT from base tables
- Multiple aggregation functions
- PIVOT with CTEs
- PIVOT with filters and JOINs
- Real-world financial reporting

---

## UNPIVOT Operations

### `unpivot_example.ipynb`
Comprehensive examples of UNPIVOT operations for normalizing data.

**Features demonstrated:**
- Basic UNPIVOT - quarterly normalization
- Multiple measure columns
- NULL handling with INCLUDE NULLS
- UNPIVOT with CTEs
- Real-world survey data analysis

---

## Running Notebooks

### Using Jupyter

Open notebooks directly in Jupyter Lab or Jupyter Notebook:

```bash
# Install Jupyter if needed
pip install jupyterlab

# Start Jupyter
jupyter lab examples/
```

### Running All Notebooks (Testing)

To verify all notebooks run correctly:

```bash
# From the clgraph directory
python run_all_notebooks.py --skip-llm

# Include LLM examples (requires Ollama)
python run_all_notebooks.py
```

Or using make:

```bash
make check-examples        # Skip LLM examples
make check-examples-llm    # Include LLM examples
```

---

## SQL Files

The `sql_files/` directory contains:
- Example SQL queries for pipeline analysis
- Notebooks demonstrating file-based pipeline workflows:
  - `run_lineage.ipynb` - Lineage analysis from SQL files
  - `run_metadata.ipynb` - Metadata management example
  - `run_with_duckdb.ipynb` - Execute pipeline with DuckDB

---

## Example Output

Each notebook includes:
- Clear SQL query examples
- Markdown explanations for each step
- Lineage analysis results
- Node and edge counts
- Output column listings
- Source table identification

---

## What's Next?

After exploring these examples, check out:

1. **Documentation**: Visit the full documentation for API reference
2. **Tests**: Look at `tests/` for more usage patterns
3. **Your own queries**: Try analyzing your own SQL queries!

---

## Support

For questions or issues:
- GitHub Issues: https://github.com/mingjerli/clgraph/issues

---

## Contributing

Have a good example to share? Contributions are welcome!
1. Fork the repository
2. Add your notebook to this directory
3. Update this README
4. Submit a pull request
