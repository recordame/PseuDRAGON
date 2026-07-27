"""
Database Utility Module for PseuDRAGON Framework

This module provides utilities for interacting with databases (DuckDB, Oracle, MariaDB),
including connection management, schema extraction with COMMENT support, and data sampling.
"""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

try:
    import duckdb
except ImportError:
    print("Warning: duckdb not installed. Install with: pip install duckdb")
    duckdb = None


class DatabaseError(Exception):
    """
    Custom exception for database operations
    """


class DatabaseConnector(ABC):
    """
    Abstract base class for database connectors

    Provides a unified interface for different database types.
    """

    @abstractmethod
    def connect(self):
        """Establish database connection"""

    @abstractmethod
    def get_table_names(self) -> List[str]:
        """Get all table names"""

    @abstractmethod
    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """Get table schema"""

    @abstractmethod
    def get_column_comments(self, table_name: str) -> Dict[str, str]:
        """Get column comments"""

    @abstractmethod
    def get_table_sample(self, table_name: str, limit: int = 1) -> Optional[Dict[str, Any]]:
        """Get sample row from table"""

    @abstractmethod
    def close(self):
        """Close database connection"""


class DuckDBConnector(DatabaseConnector):
    """
    DuckDB database connector with COMMENT support
    """

    SYSTEM_TABLES = {"information_schema", "pg_catalog"}

    def __init__(self, db_path: str):
        """
        Initialize DuckDB connector

        Args:
            db_path: Path to DuckDB database file
        """
        if duckdb is None:
            raise DatabaseError("DuckDB is not installed. Install with: pip install duckdb")

        self.db_path = db_path
        self.conn = None

    def connect(self):
        """Establish DuckDB connection"""
        try:
            self.conn = duckdb.connect(self.db_path)
            return self.conn
        except Exception as e:
            raise DatabaseError(f"Failed to connect to DuckDB: {e}")

    def get_table_names(self) -> List[str]:
        """Get all user table names from DuckDB"""
        if not self.conn:
            self.connect()

        try:
            result = self.conn.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
                  AND table_type = 'BASE TABLE'
                """
            ).fetchall()

            return [row[0] for row in result]
        except Exception as e:
            raise DatabaseError(f"Failed to retrieve table names: {e}")

    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """Get table schema from DuckDB"""
        if not self.conn:
            self.connect()

        try:
            result = self.conn.execute(
                f"""
                SELECT
                    column_name,
                    data_type,
                    is_nullable,
                    column_default,
                    ordinal_position,
                    column_comment
                FROM information_schema.columns
                WHERE table_name = '{table_name}'
                ORDER BY ordinal_position
            """
            ).fetchall()

            schema = []
            for row in result:
                schema.append({"name": row[0], "type": row[1], "nullable": row[2] == "YES", "default": row[3], "position": row[4], "comment": row[5] if row[5] else None, })

            return schema
        except Exception as e:
            raise DatabaseError(f"Failed to retrieve schema for table '{table_name}': {e}")

    def get_column_comments(self, table_name: str) -> Dict[str, str]:
        """
        Get column comments from DuckDB

        Note: DuckDB stores comments via COMMENT ON COLUMN syntax

        Args:
            table_name: Name of the table

        Returns:
            Dictionary mapping column names to their comments
        """
        if not self.conn:
            self.connect()

        try:
            # DuckDB doesn't have a standard way to query comments yet
            # We'll try to get them from the catalog if available
            # For now, return empty dict - comments are set but not easily queryable
            # TODO: Update when DuckDB adds comment query support

            # Workaround: Parse from table description
            comments = {}

            # Try to get comments from information_schema
            try:
                result = self.conn.execute(
                    f"""
                    SELECT column_name, column_comment
                    FROM information_schema.columns
                    WHERE table_name = '{table_name}'
                """
                ).fetchall()

                for row in result:
                    if row[1]:  # If comment exists
                        comments[row[0]] = row[1]
            except:
                # If query fails, return empty
                pass

            return comments
        except Exception:
            # If comment retrieval fails, return empty dict
            return {}

    def get_table_sample(self, table_name: str, limit: int = 1) -> Optional[Dict[str, Any]]:
        """Get sample row from DuckDB table"""
        if not self.conn:
            self.connect()

        try:
            result = self.conn.execute(f"SELECT * FROM {table_name} LIMIT {limit}").fetchone()

            if result:
                columns = [desc[0] for desc in self.conn.description]
                return dict(zip(columns, result))
            return None
        except Exception as e:
            raise DatabaseError(f"Failed to retrieve sample from table '{table_name}': {e}")

    def close(self):
        """Close DuckDB connection"""
        if self.conn:
            self.conn.close()
            self.conn = None


class OracleConnector(DatabaseConnector):
    """
    Oracle database connector
    """

    def __init__(self, host: str, port: int, service_name: str, user: str, password: str):
        """Initialize Oracle connector"""
        try:
            import cx_Oracle

            self.cx_Oracle = cx_Oracle
        except ImportError:
            raise DatabaseError("cx_Oracle is not installed. Install with: pip install cx_Oracle")

        self.host = host
        self.port = port
        self.service_name = service_name
        self.user = user
        self.password = password
        self.conn = None

    def connect(self):
        """Establish Oracle connection"""
        try:
            dsn = self.cx_Oracle.makedsn(self.host, self.port, service_name=self.service_name)
            self.conn = self.cx_Oracle.connect(user=self.user, password=self.password, dsn=dsn)
            return self.conn
        except Exception as e:
            raise DatabaseError(f"Failed to connect to Oracle: {e}")

    def get_table_names(self) -> List[str]:
        """Get all user table names from Oracle"""
        if not self.conn:
            self.connect()

        cursor = self.conn.cursor()
        cursor.execute("SELECT table_name FROM user_tables")
        return [row[0] for row in cursor.fetchall()]

    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """Get table schema from Oracle"""
        if not self.conn:
            self.connect()

        cursor = self.conn.cursor()
        cursor.execute(
            f"""
            SELECT column_name, data_type, nullable, data_default, column_id
            FROM user_tab_columns
            WHERE table_name = '{table_name.upper()}'
            ORDER BY column_id
        """
        )

        schema = []
        for row in cursor.fetchall():
            schema.append({"name": row[0], "type": row[1], "nullable": row[2] == "Y", "default": row[3], "position": row[4], })

        return schema

    def get_column_comments(self, table_name: str) -> Dict[str, str]:
        """Get column comments from Oracle"""
        if not self.conn:
            self.connect()

        cursor = self.conn.cursor()
        cursor.execute(
            f"""
            SELECT column_name, comments
            FROM user_col_comments
            WHERE table_name = '{table_name.upper()}'
            AND comments IS NOT NULL
        """
        )

        return {row[0]: row[1] for row in cursor.fetchall()}

    def get_table_sample(self, table_name: str, limit: int = 1) -> Optional[Dict[str, Any]]:
        """Get sample row from Oracle table"""
        if not self.conn:
            self.connect()

        cursor = self.conn.cursor()
        cursor.execute(f"SELECT * FROM {table_name} WHERE ROWNUM <= {limit}")

        row = cursor.fetchone()
        if row:
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, row))
        return None

    def close(self):
        """Close Oracle connection"""
        if self.conn:
            self.conn.close()
            self.conn = None


class MariaDBConnector(DatabaseConnector):
    """
    MariaDB/MySQL database connector
    """

    def __init__(self, host: str, port: int, database: str, user: str, password: str):
        """Initialize MariaDB connector"""
        try:
            import pymysql

            self.pymysql = pymysql
        except ImportError:
            raise DatabaseError("pymysql is not installed. Install with: pip install pymysql")

        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.conn = None

    def connect(self):
        """Establish MariaDB connection"""
        try:
            self.conn = self.pymysql.connect(host=self.host, port=self.port, database=self.database, user=self.user, password=self.password, charset="utf8mb4", )
            return self.conn
        except Exception as e:
            raise DatabaseError(f"Failed to connect to MariaDB: {e}")

    def get_table_names(self) -> List[str]:
        """Get all table names from MariaDB"""
        if not self.conn:
            self.connect()

        cursor = self.conn.cursor()
        cursor.execute("SHOW TABLES")
        return [row[0] for row in cursor.fetchall()]

    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """Get table schema from MariaDB"""
        if not self.conn:
            self.connect()

        cursor = self.conn.cursor()
        cursor.execute(
            f"""
            SELECT 
                COLUMN_NAME, 
                DATA_TYPE, 
                IS_NULLABLE, 
                COLUMN_DEFAULT, 
                ORDINAL_POSITION
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = '{self.database}'
            AND TABLE_NAME = '{table_name}'
            ORDER BY ORDINAL_POSITION
        """
        )

        schema = []
        for row in cursor.fetchall():
            schema.append({"name": row[0], "type": row[1], "nullable": row[2] == "YES", "default": row[3], "position": row[4], })

        return schema

    def get_column_comments(self, table_name: str) -> Dict[str, str]:
        """Get column comments from MariaDB"""
        if not self.conn:
            self.connect()

        cursor = self.conn.cursor()
        cursor.execute(
            f"""
            SELECT COLUMN_NAME, COLUMN_COMMENT
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = '{self.database}'
            AND TABLE_NAME = '{table_name}'
            AND COLUMN_COMMENT != ''
        """
        )

        return {row[0]: row[1] for row in cursor.fetchall()}

    def get_table_sample(self, table_name: str, limit: int = 1) -> Optional[Dict[str, Any]]:
        """Get sample row from MariaDB table"""
        if not self.conn:
            self.connect()

        cursor = self.conn.cursor()
        cursor.execute(f"SELECT * FROM {table_name} LIMIT {limit}")

        row = cursor.fetchone()
        if row:
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, row))
        return None

    def close(self):
        """Close MariaDB connection"""
        if self.conn:
            self.conn.close()
            self.conn = None


class DatabaseManager:
    """
    Database Manager with multi-database support

    This class provides a centralized interface for database operations,
    supporting DuckDB, Oracle, and MariaDB with COMMENT extraction.
    """

    def __init__(self, db_config):
        """
        Initialize DatabaseManager with configuration

        Args:
            db_config: Database path (str) for DuckDB or config dict for others
        """
        self.connector = self._create_connector(db_config)

    def _create_connector(self, db_config) -> DatabaseConnector:
        """Create appropriate database connector based on configuration"""
        if isinstance(db_config, str):
            # DuckDB path
            return DuckDBConnector(db_config)
        elif isinstance(db_config, dict):
            db_type = db_config.get("type", "duckdb").lower()

            if db_type == "duckdb":
                return DuckDBConnector(db_config.get("path", "sample.duckdb"))
            elif db_type == "oracle":
                return OracleConnector(
                    host=db_config["host"],
                    port=int(db_config["port"]),
                    service_name=db_config.get("service_name", db_config["database"]),
                    user=db_config["user"],
                    password=db_config["password"], )
            elif db_type in ("mariadb", "mysql"):
                return MariaDBConnector(host=db_config["host"], port=int(db_config["port"]), database=db_config["database"], user=db_config["user"], password=db_config["password"], )
            else:
                raise DatabaseError(f"Unsupported database type: {db_type}")
        else:
            raise DatabaseError("Invalid database configuration")

    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections

        Ensures proper connection cleanup even if errors occur.
        """
        try:
            conn = self.connector.connect()
            yield conn
        finally:
            self.connector.close()

    def get_table_names(self) -> List[str]:
        """Get all user table names"""
        return self.connector.get_table_names()

    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """Get table schema"""
        return self.connector.get_table_schema(table_name)

    def get_column_comments(self, table_name: str) -> Dict[str, str]:
        """
        Get column comments for a table

        Args:
            table_name: Name of the table

        Returns:
            Dictionary mapping column names to their comments
        """
        return self.connector.get_column_comments(table_name)

    def get_enhanced_schema(self, table_name: str) -> Dict[str, Dict[str, Any]]:
        """
        Get enhanced schema with column names, types, and comments

        This is the key method for PII detection with COMMENT support.

        Args:
            table_name: Name of the table

        Returns:
            Dictionary mapping column names to their metadata (type, comment, etc.)
        """
        schema = self.get_table_schema(table_name)
        comments = self.get_column_comments(table_name)

        enhanced = {}
        for col in schema:
            col_name = col["name"]
            enhanced[col_name] = {"type": col["type"], "nullable": col.get("nullable", True), "comment": comments.get(col_name, ""), "position": col.get("position", 0), }

        return enhanced

    def get_all_schemas(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get schema information for all tables"""
        tables = self.get_table_names()
        schemas = {}

        for table in tables:
            schemas[table] = self.get_table_schema(table)

        return schemas

    def get_table_sample(self, table_name: str, limit: int = 1) -> Optional[Dict[str, Any]]:
        """Get sample row from table"""
        return self.connector.get_table_sample(table_name, limit)

    def close(self):
        """Close database connection"""
        self.connector.close()


# Legacy functions for backward compatibility


def get_db_connection(db_path: str):
    """
    Legacy function: Returns a connection to the database

    Note: Consider using DatabaseManager.get_connection() context manager instead
    """
    db_manager = DatabaseManager(db_path)
    return db_manager.connector.connect()


def get_db_schema(db_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """Legacy function: Returns the schema of all tables"""
    db_manager = DatabaseManager(db_path)
    return db_manager.get_all_schemas()


def get_table_names(db_path: str) -> List[str]:
    """Legacy function: Returns a list of all table names"""
    db_manager = DatabaseManager(db_path)
    return db_manager.get_table_names()


def get_table_sample(db_path: str, table_name: str) -> Dict[str, Any]:
    """Legacy function: Retrieves a single row from the specified table"""
    try:
        db_manager = DatabaseManager(db_path)
        result = db_manager.get_table_sample(table_name)
        return result if result else {}
    except DatabaseError as e:
        print(f"Error: {e}")
        return {}
