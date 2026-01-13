"""
Tests for Kestra orchestrator integration.

Tests the to_kestra_flow() method and KestraOrchestrator class.
"""

import yaml

from clgraph import Pipeline


def mock_executor(sql: str) -> None:
    """Mock executor for testing."""
    pass


class TestToKestraFlowBasic:
    """Basic tests for to_kestra_flow method."""

    def test_basic_flow_generation(self):
        """Test basic Kestra flow YAML generation."""
        pipeline = Pipeline(
            [
                ("staging", "CREATE TABLE staging AS SELECT 1 as id"),
                ("analytics", "CREATE TABLE analytics AS SELECT * FROM staging"),
            ]
        )

        yaml_content = pipeline.to_kestra_flow(flow_id="test_flow", namespace="clgraph.test")

        flow = yaml.safe_load(yaml_content)

        assert flow["id"] == "test_flow"
        assert flow["namespace"] == "clgraph.test"
        assert len(flow["tasks"]) == 2

    def test_flow_with_description(self):
        """Test flow with custom description."""
        pipeline = Pipeline([("query1", "CREATE TABLE table1 AS SELECT 1")])

        yaml_content = pipeline.to_kestra_flow(
            flow_id="desc_flow",
            namespace="clgraph.test",
            description="Custom description for testing",
        )

        flow = yaml.safe_load(yaml_content)
        assert flow["description"] == "Custom description for testing"

    def test_flow_auto_description(self):
        """Test auto-generated description."""
        pipeline = Pipeline(
            [
                ("q1", "CREATE TABLE t1 AS SELECT 1"),
                ("q2", "CREATE TABLE t2 AS SELECT * FROM t1"),
            ]
        )

        yaml_content = pipeline.to_kestra_flow(flow_id="auto_desc", namespace="clgraph.test")

        flow = yaml.safe_load(yaml_content)
        assert "2 queries" in flow["description"]
        assert "clgraph" in flow["description"]

    def test_task_structure(self):
        """Test task structure contains required fields."""
        pipeline = Pipeline([("query1", "CREATE TABLE table1 AS SELECT 1")])

        yaml_content = pipeline.to_kestra_flow(flow_id="task_test", namespace="clgraph.test")

        flow = yaml.safe_load(yaml_content)
        task = flow["tasks"][0]

        assert task["id"] == "query1"
        assert task["type"] == "io.kestra.plugin.jdbc.clickhouse.Query"
        assert "url" in task
        assert "username" in task
        assert "sql" in task
        assert "retry" in task

    def test_task_sql_content(self):
        """Test that SQL is correctly embedded in tasks."""
        sql = "CREATE TABLE test_table AS SELECT 42 as value"
        pipeline = Pipeline([("query1", sql)])

        yaml_content = pipeline.to_kestra_flow(flow_id="sql_test", namespace="clgraph.test")

        flow = yaml.safe_load(yaml_content)
        assert flow["tasks"][0]["sql"] == sql


class TestToKestraFlowDependencies:
    """Test Kestra flow dependency handling via topological ordering."""

    def test_linear_dependencies(self):
        """Test linear dependency chain - tasks should be in topological order."""
        pipeline = Pipeline(
            [
                ("step1", "CREATE TABLE step1 AS SELECT 1"),
                ("step2", "CREATE TABLE step2 AS SELECT * FROM step1"),
                ("step3", "CREATE TABLE step3 AS SELECT * FROM step2"),
            ]
        )

        yaml_content = pipeline.to_kestra_flow(flow_id="linear_deps", namespace="clgraph.test")

        flow = yaml.safe_load(yaml_content)
        task_ids = [t["id"] for t in flow["tasks"]]

        # Tasks should be in topological order: step1 before step2 before step3
        assert task_ids.index("step1") < task_ids.index("step2")
        assert task_ids.index("step2") < task_ids.index("step3")

    def test_diamond_dependencies(self):
        """Test diamond pattern dependencies - tasks should be in valid topological order."""
        pipeline = Pipeline(
            [
                ("source", "CREATE TABLE source AS SELECT 1 as id"),
                ("left_branch", "CREATE TABLE left_branch AS SELECT * FROM source"),
                ("right_branch", "CREATE TABLE right_branch AS SELECT * FROM source"),
                (
                    "final",
                    "CREATE TABLE final AS SELECT * FROM left_branch, right_branch",
                ),
            ]
        )

        yaml_content = pipeline.to_kestra_flow(flow_id="diamond_deps", namespace="clgraph.test")

        flow = yaml.safe_load(yaml_content)
        task_ids = [t["id"] for t in flow["tasks"]]

        # source should come before left and right branches
        assert task_ids.index("source") < task_ids.index("left_branch")
        assert task_ids.index("source") < task_ids.index("right_branch")
        # final should come after both branches
        assert task_ids.index("left_branch") < task_ids.index("final")
        assert task_ids.index("right_branch") < task_ids.index("final")

    def test_no_depends_on_field(self):
        """Test that tasks don't have dependsOn field (Kestra uses sequential execution)."""
        pipeline = Pipeline(
            [
                ("source1", "CREATE TABLE source1 AS SELECT 1"),
                ("source2", "CREATE TABLE source2 AS SELECT 2"),
                (
                    "combined",
                    "CREATE TABLE combined AS SELECT * FROM source1, source2",
                ),
            ]
        )

        yaml_content = pipeline.to_kestra_flow(flow_id="multi_source", namespace="clgraph.test")

        flow = yaml.safe_load(yaml_content)

        # No task should have dependsOn field
        for task in flow["tasks"]:
            assert "dependsOn" not in task


class TestToKestraFlowScheduling:
    """Test Kestra flow scheduling."""

    def test_flow_with_cron(self):
        """Test flow with cron schedule."""
        pipeline = Pipeline([("query1", "CREATE TABLE table1 AS SELECT 1")])

        yaml_content = pipeline.to_kestra_flow(
            flow_id="scheduled_flow", namespace="clgraph.test", cron="0 0 * * *"
        )

        flow = yaml.safe_load(yaml_content)

        assert "triggers" in flow
        assert len(flow["triggers"]) == 1
        trigger = flow["triggers"][0]
        assert trigger["type"] == "io.kestra.plugin.core.trigger.Schedule"
        assert trigger["cron"] == "0 0 * * *"

    def test_flow_without_schedule(self):
        """Test flow without schedule has no triggers."""
        pipeline = Pipeline([("query1", "CREATE TABLE table1 AS SELECT 1")])

        yaml_content = pipeline.to_kestra_flow(flow_id="unscheduled_flow", namespace="clgraph.test")

        flow = yaml.safe_load(yaml_content)
        assert "triggers" not in flow

    def test_hourly_cron(self):
        """Test hourly cron schedule."""
        pipeline = Pipeline([("query1", "CREATE TABLE table1 AS SELECT 1")])

        yaml_content = pipeline.to_kestra_flow(
            flow_id="hourly_flow", namespace="clgraph.test", cron="0 * * * *"
        )

        flow = yaml.safe_load(yaml_content)
        assert flow["triggers"][0]["cron"] == "0 * * * *"


class TestToKestraFlowConfiguration:
    """Test Kestra flow configuration options."""

    def test_custom_connection_config(self):
        """Test custom connection configuration."""
        pipeline = Pipeline([("query1", "CREATE TABLE table1 AS SELECT 1")])

        yaml_content = pipeline.to_kestra_flow(
            flow_id="custom_conn",
            namespace="clgraph.test",
            connection_config={
                "url": "jdbc:clickhouse://custom-host:8123/mydb",
                "username": "myuser",
                "password": "mypassword",
            },
        )

        flow = yaml.safe_load(yaml_content)
        task = flow["tasks"][0]

        assert task["url"] == "jdbc:clickhouse://custom-host:8123/mydb"
        assert task["username"] == "myuser"
        assert task["password"] == "mypassword"

    def test_default_connection_config(self):
        """Test default connection configuration."""
        pipeline = Pipeline([("query1", "CREATE TABLE table1 AS SELECT 1")])

        yaml_content = pipeline.to_kestra_flow(flow_id="default_conn", namespace="clgraph.test")

        flow = yaml.safe_load(yaml_content)
        task = flow["tasks"][0]

        assert task["url"] == "jdbc:clickhouse://clickhouse:8123/default"
        assert task["username"] == "default"
        assert task["password"] == ""

    def test_custom_retry_config(self):
        """Test custom retry configuration."""
        pipeline = Pipeline([("query1", "CREATE TABLE table1 AS SELECT 1")])

        yaml_content = pipeline.to_kestra_flow(
            flow_id="retry_test", namespace="clgraph.test", retry_attempts=5
        )

        flow = yaml.safe_load(yaml_content)
        task = flow["tasks"][0]

        assert task["retry"]["maxAttempt"] == 5

    def test_custom_labels(self):
        """Test custom labels."""
        pipeline = Pipeline([("query1", "CREATE TABLE table1 AS SELECT 1")])

        yaml_content = pipeline.to_kestra_flow(
            flow_id="labels_test",
            namespace="clgraph.test",
            labels={"env": "production", "team": "analytics"},
        )

        flow = yaml.safe_load(yaml_content)

        assert flow["labels"]["env"] == "production"
        assert flow["labels"]["team"] == "analytics"

    def test_default_labels(self):
        """Test default labels include generator tag."""
        pipeline = Pipeline([("query1", "CREATE TABLE table1 AS SELECT 1")])

        yaml_content = pipeline.to_kestra_flow(flow_id="default_labels", namespace="clgraph.test")

        flow = yaml.safe_load(yaml_content)
        assert flow["labels"]["generator"] == "clgraph"


class TestToKestraFlowOutput:
    """Test Kestra flow output format."""

    def test_valid_yaml_output(self):
        """Test that output is valid, parseable YAML."""
        pipeline = Pipeline(
            [
                ("q1", "CREATE TABLE t1 AS SELECT 1"),
                ("q2", "CREATE TABLE t2 AS SELECT * FROM t1"),
            ]
        )

        yaml_content = pipeline.to_kestra_flow(flow_id="valid_yaml", namespace="clgraph.test")

        # Should not raise
        flow = yaml.safe_load(yaml_content)
        assert isinstance(flow, dict)

    def test_yaml_is_string(self):
        """Test that to_kestra_flow returns a string."""
        pipeline = Pipeline([("query1", "CREATE TABLE table1 AS SELECT 1")])

        result = pipeline.to_kestra_flow(flow_id="string_test", namespace="clgraph.test")

        assert isinstance(result, str)

    def test_yaml_has_required_fields(self):
        """Test YAML has all required Kestra fields."""
        pipeline = Pipeline([("query1", "CREATE TABLE table1 AS SELECT 1")])

        yaml_content = pipeline.to_kestra_flow(flow_id="required_fields", namespace="clgraph.test")

        flow = yaml.safe_load(yaml_content)

        # Required Kestra flow fields
        assert "id" in flow
        assert "namespace" in flow
        assert "tasks" in flow


class TestKestraOrchestrator:
    """Test KestraOrchestrator class directly."""

    def test_orchestrator_initialization(self):
        """Test KestraOrchestrator initialization."""
        from clgraph.orchestrators import KestraOrchestrator

        pipeline = Pipeline([("query1", "CREATE TABLE table1 AS SELECT 1")])
        orchestrator = KestraOrchestrator(pipeline)

        assert orchestrator.pipeline == pipeline
        assert orchestrator.table_graph == pipeline.table_graph

    def test_to_flow_dict(self):
        """Test to_flow_dict returns dictionary."""
        from clgraph.orchestrators import KestraOrchestrator

        pipeline = Pipeline([("query1", "CREATE TABLE table1 AS SELECT 1")])
        orchestrator = KestraOrchestrator(pipeline)

        result = orchestrator.to_flow_dict(flow_id="dict_test", namespace="clgraph.test")

        assert isinstance(result, dict)
        assert result["id"] == "dict_test"
        assert result["namespace"] == "clgraph.test"

    def test_to_flow_with_triggers(self):
        """Test to_flow_with_triggers adds triggers."""
        from clgraph.orchestrators import KestraOrchestrator

        pipeline = Pipeline([("query1", "CREATE TABLE table1 AS SELECT 1")])
        orchestrator = KestraOrchestrator(pipeline)

        yaml_content = orchestrator.to_flow_with_triggers(
            flow_id="trigger_test", namespace="clgraph.test", cron="0 0 * * *"
        )

        flow = yaml.safe_load(yaml_content)
        assert "triggers" in flow
        assert flow["triggers"][0]["cron"] == "0 0 * * *"


class TestKestraFlowComplexPipeline:
    """Test Kestra flow generation with complex pipelines."""

    def test_enterprise_like_pipeline(self):
        """Test with a pipeline similar to enterprise demo."""
        pipeline = Pipeline(
            [
                (
                    "raw_sales",
                    """
                    CREATE TABLE raw_sales AS
                    SELECT
                        toDate('2024-01-01') + number as date,
                        number % 100 as product_id,
                        number % 10 as region_id,
                        rand() % 1000 as amount
                    FROM numbers(1000)
                """,
                ),
                (
                    "raw_products",
                    """
                    CREATE TABLE raw_products AS
                    SELECT
                        number as product_id,
                        concat('Product ', toString(number)) as product_name,
                        rand() % 5 as category_id
                    FROM numbers(100)
                """,
                ),
                (
                    "sales_with_products",
                    """
                    CREATE TABLE sales_with_products AS
                    SELECT
                        s.date,
                        s.product_id,
                        p.product_name,
                        s.region_id,
                        s.amount
                    FROM raw_sales s
                    JOIN raw_products p ON s.product_id = p.product_id
                """,
                ),
                (
                    "daily_summary",
                    """
                    CREATE TABLE daily_summary AS
                    SELECT
                        date,
                        count() as num_sales,
                        sum(amount) as total_amount
                    FROM sales_with_products
                    GROUP BY date
                """,
                ),
            ],
            dialect="clickhouse",
        )

        yaml_content = pipeline.to_kestra_flow(
            flow_id="enterprise_pipeline",
            namespace="clgraph.enterprise",
            labels={"env": "production", "source": "clgraph"},
        )

        flow = yaml.safe_load(yaml_content)

        assert flow["id"] == "enterprise_pipeline"
        assert flow["namespace"] == "clgraph.enterprise"
        assert len(flow["tasks"]) == 4

        task_ids = [t["id"] for t in flow["tasks"]]

        # Verify topological ordering (no dependsOn field, just correct order)
        # raw_sales and raw_products should come before sales_with_products
        assert task_ids.index("raw_sales") < task_ids.index("sales_with_products")
        assert task_ids.index("raw_products") < task_ids.index("sales_with_products")
        # sales_with_products should come before daily_summary
        assert task_ids.index("sales_with_products") < task_ids.index("daily_summary")

    def test_many_queries_pipeline(self):
        """Test with many queries."""
        queries = []
        for i in range(10):
            if i == 0:
                queries.append((f"step_{i}", f"CREATE TABLE step_{i} AS SELECT {i}"))
            else:
                queries.append(
                    (
                        f"step_{i}",
                        f"CREATE TABLE step_{i} AS SELECT * FROM step_{i - 1}",
                    )
                )

        pipeline = Pipeline(queries)

        yaml_content = pipeline.to_kestra_flow(flow_id="many_queries", namespace="clgraph.test")

        flow = yaml.safe_load(yaml_content)
        assert len(flow["tasks"]) == 10

        # Verify topological ordering (each step should come after its predecessor)
        task_ids = [t["id"] for t in flow["tasks"]]
        for i in range(1, 10):
            assert task_ids.index(f"step_{i - 1}") < task_ids.index(f"step_{i}")
